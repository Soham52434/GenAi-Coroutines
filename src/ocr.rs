use anyhow::{Context, Result as AnyhowResult};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, error, info, warn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rand::Rng;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// API response from initial upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChandraApiResponse {
    pub request_check_url: String,
    pub request_id: Option<String>,
}

/// Result for a single OCR task - matches input index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub index: usize,
    pub success: bool,
    pub json_response: Option<serde_json::Value>,
    pub error: Option<String>,
    pub processing_time_secs: f64,
    pub cost_breakdown: Option<serde_json::Value>,
}

/// Configuration
#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub api_key: String,
    pub api_url: String,
    pub output_format: String,
    pub mode: String,
    pub max_concurrent_requests: usize,
    pub poll_interval_secs: u64,
    pub max_poll_attempts: u32,
    pub max_retries: u32,
    pub base_retry_delay_secs: u64,
    pub jitter_percent: u64,
    pub page_schema: Option<String>,
    pub paginate: bool,
    pub page_range: Option<String>,
    pub max_pages: Option<u32>,
    pub disable_image_extraction: bool,
    pub extras: Option<String>,
    pub webhook_url: Option<String>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_url: "https://www.datalab.to/api/v1/marker".to_string(),
            output_format: "json".to_string(),
            mode: "accurate".to_string(),
            max_concurrent_requests: 10,
            poll_interval_secs: 2,
            max_poll_attempts: 60,
            max_retries: 5,
            base_retry_delay_secs: 5,
            jitter_percent: 200,
            page_schema: None,
            paginate: false,
            page_range: None,
            max_pages: None,
            disable_image_extraction: false,
            extras: None,
            webhook_url: None,
        }
    }
}

// ============================================================================
// ASYNC OCR PROCESSOR
// ============================================================================

pub struct AsyncOcrProcessor {
    config: Arc<OcrConfig>,
    client: reqwest::Client,
}

impl AsyncOcrProcessor {
    pub fn new(config: OcrConfig) -> AnyhowResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(config.max_concurrent_requests)
            .build()
            .context("Failed to create HTTP client")?;

        info!(
            "[chandra] processor_init | url={} format={} mode={} concurrency={} retries={} page_schema={}",
            config.api_url,
            config.output_format,
            config.mode,
            config.max_concurrent_requests,
            config.max_retries,
            config.page_schema.is_some()
        );

        Ok(Self {
            config: Arc::new(config),
            client,
        })
    }

    pub async fn process_multipart_batch(&self, multiparts: Vec<Vec<u8>>) -> Vec<OcrResult> {
        let total = multiparts.len();
        info!(
            "[chandra] batch_start | total={} concurrency={}",
            total, self.config.max_concurrent_requests
        );

        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            self.config.max_concurrent_requests,
        ));

        let mut tasks = FuturesUnordered::new();

        for (index, multipart_data) in multiparts.into_iter().enumerate() {
            let processor = self.clone();
            let permit = semaphore.clone();

            let task = tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                debug!("[chandra] task_acquired_permit | index={}", index);
                processor.process_single_multipart(index, multipart_data).await
            });

            tasks.push(task);
        }

        let mut results = Vec::with_capacity(total);
        while let Some(task_result) = tasks.next().await {
            match task_result {
                Ok(ocr_result) => results.push(ocr_result),
                Err(e) => {
                    error!("[chandra] task_panic | error={}", e);
                    results.push(OcrResult {
                        index: 0,
                        success: false,
                        json_response: None,
                        error: Some(format!("Task panic: {}", e)),
                        processing_time_secs: 0.0,
                        cost_breakdown: None,
                    });
                }
            }
        }

        results.sort_by_key(|r| r.index);

        let success_count = results.iter().filter(|r| r.success).count();
        let avg_time: f64 = results.iter().map(|r| r.processing_time_secs).sum::<f64>()
            / total as f64;

        info!(
            "[chandra] batch_done | success={}/{} avg_time={:.2}s",
            success_count, total, avg_time
        );

        results
    }

    async fn process_single_multipart(&self, index: usize, multipart_data: Vec<u8>) -> OcrResult {
        let start = std::time::Instant::now();
        let data_size = multipart_data.len();

        info!(
            "[chandra] task_start | index={} size_bytes={}",
            index, data_size
        );

        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                let delay = self.calculate_retry_delay(attempt);
                warn!(
                    "[chandra] retry | index={} attempt={}/{} delay={:.1}s",
                    index,
                    attempt,
                    self.config.max_retries,
                    delay.as_secs_f64()
                );
                sleep(delay).await;
            }

            match self.execute_ocr(&multipart_data).await {
                Ok(json_response) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    let cost_breakdown = json_response.get("cost_breakdown").cloned();
                    info!(
                        "[chandra] task_success | index={} time={:.2}s attempts={}",
                        index,
                        elapsed,
                        attempt + 1
                    );
                    return OcrResult {
                        index,
                        success: true,
                        json_response: Some(json_response),
                        error: None,
                        processing_time_secs: elapsed,
                        cost_breakdown,
                    };
                }
                Err(e) => {
                    let error_str = format!("{:?}", e);

                    if !self.is_retriable_error(&error_str) {
                        let elapsed = start.elapsed().as_secs_f64();
                        error!(
                            "[chandra] task_fail_fatal | index={} error={} time={:.2}s",
                            index, error_str, elapsed
                        );
                        return OcrResult {
                            index,
                            success: false,
                            json_response: None,
                            error: Some(error_str),
                            processing_time_secs: elapsed,
                            cost_breakdown: None,
                        };
                    }

                    if self.is_rate_limit_error(&e) {
                        warn!(
                            "[chandra] rate_limit | index={} attempt={}/{}",
                            index,
                            attempt + 1,
                            self.config.max_retries
                        );
                    }

                    if attempt == self.config.max_retries - 1 {
                        let elapsed = start.elapsed().as_secs_f64();
                        error!(
                            "[chandra] task_fail_exhausted | index={} retries={} error={} time={:.2}s",
                            index, self.config.max_retries, error_str, elapsed
                        );
                    }
                }
            }
        }

        OcrResult {
            index,
            success: false,
            json_response: None,
            error: Some("Max retries exceeded".to_string()),
            processing_time_secs: start.elapsed().as_secs_f64(),
            cost_breakdown: None,
        }
    }

    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let base_delay_ms =
            self.config.base_retry_delay_secs * 1000 * (2_u64.pow(attempt.saturating_sub(1)));
        let jitter_range = (base_delay_ms * self.config.jitter_percent) / 100;

        let mut rng = rand::thread_rng();
        let jitter: i64 =
            rng.gen_range(-(jitter_range as i64)..=(jitter_range as i64));

        let final_delay_ms = (base_delay_ms as i64 + jitter).max(0) as u64;
        Duration::from_millis(final_delay_ms)
    }

    fn is_rate_limit_error(&self, error: &anyhow::Error) -> bool {
        let error_str = format!("{:?}", error).to_lowercase();
        error_str.contains("429")
            || error_str.contains("rate limit")
            || error_str.contains("too many requests")
    }

    fn is_retriable_error(&self, error_str: &str) -> bool {
        let lower = error_str.to_lowercase();

        if lower.contains("400")
            || lower.contains("bad request")
            || lower.contains("401")
            || lower.contains("unauthorized")
            || lower.contains("403")
            || lower.contains("forbidden")
            || lower.contains("404")
            || lower.contains("not found")
        {
            return false;
        }

        lower.contains("429")
            || lower.contains("500")
            || lower.contains("502")
            || lower.contains("503")
            || lower.contains("504")
            || lower.contains("rate limit")
            || lower.contains("too many requests")
            || lower.contains("timeout")
            || lower.contains("connection")
    }

    async fn execute_ocr(&self, multipart_data: &[u8]) -> AnyhowResult<serde_json::Value> {
        let check_url = self.upload_multipart(multipart_data).await?;
        let raw_json = self.poll_until_complete(&check_url).await?;
        Ok(raw_json)
    }

    async fn upload_multipart(&self, data: &[u8]) -> AnyhowResult<String> {
        let part = if let Some(kind) = infer::get(data) {
            debug!(
                "[chandra] upload_detect | mime={} ext={}",
                kind.mime_type(),
                kind.extension()
            );
            Part::bytes(data.to_vec())
                .file_name(format!("file.{}", kind.extension()))
                .mime_str(kind.mime_type())?
        } else {
            debug!("[chandra] upload_detect | mime=application/octet-stream (fallback)");
            Part::bytes(data.to_vec())
                .file_name("file.bin")
                .mime_str("application/octet-stream")?
        };

        let mut form = Form::new()
            .part("file", part)
            .text("output_format", self.config.output_format.clone())
            .text("mode", self.config.mode.clone());

        if let Some(ref schema) = self.config.page_schema {
            form = form.text("page_schema", schema.clone());
            debug!("[chandra] upload_param | page_schema=set");
        }
        if self.config.paginate {
            form = form.text("paginate", "true".to_string());
        }
        if let Some(ref range) = self.config.page_range {
            form = form.text("page_range", range.clone());
        }
        if let Some(max) = self.config.max_pages {
            form = form.text("max_pages", max.to_string());
        }
        if self.config.disable_image_extraction {
            form = form.text("disable_image_extraction", "true".to_string());
        }
        if let Some(ref extras) = self.config.extras {
            form = form.text("extras", extras.clone());
        }
        if let Some(ref webhook) = self.config.webhook_url {
            form = form.text("webhook_url", webhook.clone());
        }

        debug!("[chandra] upload_start | url={}", self.config.api_url);

        let response = self
            .client
            .post(&self.config.api_url)
            .header("X-API-Key", &self.config.api_key)
            .multipart(form)
            .send()
            .await
            .context("Upload request failed")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let error_msg = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.as_str())
                        .map(String::from)
                        .or_else(|| {
                            v.get("detail")
                                .and_then(|d| d.as_str())
                                .map(String::from)
                        })
                })
                .unwrap_or_else(|| body.clone());
            error!(
                "[chandra] api_error | status={} error={} raw_body={}",
                status.as_u16(),
                error_msg,
                body
            );
            anyhow::bail!("Datalab API error [{}]: {}", status.as_u16(), error_msg);
        }

        let api_response: ChandraApiResponse = response.json().await?;
        let request_id = api_response.request_id.as_deref().unwrap_or("unknown");
        info!(
            "[chandra] upload_ok | request_id={} check_url={}",
            request_id, api_response.request_check_url
        );
        Ok(api_response.request_check_url)
    }

    async fn poll_until_complete(&self, check_url: &str) -> AnyhowResult<serde_json::Value> {
        for attempt in 0..self.config.max_poll_attempts {
            sleep(Duration::from_secs(self.config.poll_interval_secs)).await;

            let response = self
                .client
                .get(check_url)
                .header("X-API-Key", &self.config.api_key)
                .send()
                .await?;

            if !response.status().is_success() {
                let poll_status = response.status();
                let poll_body = response.text().await.unwrap_or_default();
                let poll_error = serde_json::from_str::<serde_json::Value>(&poll_body)
                    .ok()
                    .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
                    .unwrap_or_else(|| poll_body.clone());
                error!(
                    "[chandra] poll_http_error | status={} attempt={}/{} error={}",
                    poll_status.as_u16(),
                    attempt + 1,
                    self.config.max_poll_attempts,
                    poll_error
                );
                anyhow::bail!("Poll failed [{}]: {}", poll_status.as_u16(), poll_error);
            }

            let json_resp: serde_json::Value = response.json().await?;
            let status_str = json_resp
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown");

            match status_str {
                "complete" => {
                    let quality = json_resp
                        .get("parse_quality_score")
                        .and_then(|q| q.as_f64())
                        .unwrap_or(0.0);
                    let pages = json_resp
                        .get("page_count")
                        .and_then(|p| p.as_u64())
                        .unwrap_or(0);
                    info!(
                        "[chandra] poll_complete | pages={} quality={:.1}",
                        pages, quality
                    );
                    return Ok(json_resp);
                }
                "failed" => {
                    let err = json_resp
                        .get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("Unknown");
                    error!("[chandra] poll_failed | error={}", err);
                    anyhow::bail!("Processing failed: {}", err);
                }
                "processing" | "queued" => {
                    debug!(
                        "[chandra] poll_waiting | status={} attempt={}/{}",
                        status_str,
                        attempt + 1,
                        self.config.max_poll_attempts
                    );
                }
                other => {
                    warn!("[chandra] poll_unknown_status | status={}", other);
                }
            }
        }

        error!(
            "[chandra] poll_timeout | attempts={}",
            self.config.max_poll_attempts
        );
        anyhow::bail!(
            "Polling timeout after {} attempts",
            self.config.max_poll_attempts
        )
    }
}

impl Clone for AsyncOcrProcessor {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            client: self.client.clone(),
        }
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

#[pyclass(name = "DocumentConfig")]
#[derive(Clone)]
pub struct PyDocumentConfig {
    inner: OcrConfig,
}

#[pymethods]
impl PyDocumentConfig {
    #[new]
    #[pyo3(signature = (
        api_key,
        api_url=None,
        output_format=None,
        mode=None,
        max_concurrent_requests=None,
        poll_interval_secs=None,
        max_poll_attempts=None,
        max_retries=None,
        base_retry_delay_secs=None,
        jitter_percent=None,
        page_schema=None,
        paginate=None,
        page_range=None,
        max_pages=None,
        disable_image_extraction=None,
        extras=None,
        webhook_url=None
    ))]
    fn new(
        api_key: String,
        api_url: Option<String>,
        output_format: Option<String>,
        mode: Option<String>,
        max_concurrent_requests: Option<usize>,
        poll_interval_secs: Option<u64>,
        max_poll_attempts: Option<u32>,
        max_retries: Option<u32>,
        base_retry_delay_secs: Option<u64>,
        jitter_percent: Option<u64>,
        page_schema: Option<String>,
        paginate: Option<bool>,
        page_range: Option<String>,
        max_pages: Option<u32>,
        disable_image_extraction: Option<bool>,
        extras: Option<String>,
        webhook_url: Option<String>,
    ) -> PyResult<Self> {
        let mut config = OcrConfig::default();
        config.api_key = api_key;

        if let Some(url) = api_url {
            config.api_url = url;
        }

        if let Some(fmt) = output_format {
            match fmt.as_str() {
                "json" | "html" | "markdown" | "chunks" => config.output_format = fmt,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid output_format '{}'. Must be one of: 'json', 'html', 'markdown', 'chunks'",
                        fmt
                    )))
                }
            }
        }

        if let Some(m) = mode {
            match m.as_str() {
                "fast" | "balanced" | "accurate" => config.mode = m,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid mode '{}'. Must be one of: 'fast', 'balanced', 'accurate'",
                        m
                    )))
                }
            }
        }

        if let Some(c) = max_concurrent_requests {
            config.max_concurrent_requests = c;
        }
        if let Some(p) = poll_interval_secs {
            config.poll_interval_secs = p;
        }
        if let Some(m) = max_poll_attempts {
            config.max_poll_attempts = m;
        }
        if let Some(r) = max_retries {
            config.max_retries = r;
        }
        if let Some(d) = base_retry_delay_secs {
            config.base_retry_delay_secs = d;
        }
        if let Some(j) = jitter_percent {
            config.jitter_percent = j;
        }

        if let Some(schema) = page_schema {
            serde_json::from_str::<serde_json::Value>(&schema).map_err(|e| {
                PyValueError::new_err(format!("page_schema is not valid JSON: {}", e))
            })?;
            config.page_schema = Some(schema);
        }
        if let Some(p) = paginate {
            config.paginate = p;
        }
        if let Some(r) = page_range {
            config.page_range = Some(r);
        }
        if let Some(m) = max_pages {
            config.max_pages = Some(m);
        }
        if let Some(d) = disable_image_extraction {
            config.disable_image_extraction = d;
        }
        if let Some(e) = extras {
            serde_json::from_str::<serde_json::Value>(&e).map_err(|e| {
                PyValueError::new_err(format!("extras is not valid JSON: {}", e))
            })?;
            config.extras = Some(e);
        }
        if let Some(w) = webhook_url {
            config.webhook_url = Some(w);
        }

        Ok(Self { inner: config })
    }

    fn __repr__(&self) -> String {
        format!(
            "DocumentConfig(url='{}', format='{}', mode='{}', concurrency={}, retries={}, page_schema={})",
            self.inner.api_url,
            self.inner.output_format,
            self.inner.mode,
            self.inner.max_concurrent_requests,
            self.inner.max_retries,
            self.inner.page_schema.is_some()
        )
    }
}

#[pyclass(name = "DocumentProcessor")]
pub struct PyDocumentProcessor {
    processor: AsyncOcrProcessor,
}

#[pymethods]
impl PyDocumentProcessor {
    #[new]
    fn new(config: PyDocumentConfig) -> PyResult<Self> {
        let processor = AsyncOcrProcessor::new(config.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Init failed: {}", e)))?;
        Ok(Self { processor })
    }

    fn process_multiparts<'py>(
        &self,
        py: Python<'py>,
        multiparts: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut rust_multiparts = Vec::new();

        for (idx, item) in multiparts.iter().enumerate() {
            let bytes_obj = item.downcast::<PyBytes>().map_err(|_| {
                PyValueError::new_err(format!("Item {} is not bytes object", idx))
            })?;
            rust_multiparts.push(bytes_obj.as_bytes().to_vec());
        }

        let processor = self.processor.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let results = processor.process_multipart_batch(rust_multiparts).await;

            Python::with_gil(|py| -> PyResult<PyObject> {
                let py_results = PyList::empty_bound(py);

                for result in results {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("index", result.index)?;
                    dict.set_item("success", result.success)?;
                    dict.set_item("processing_time_secs", result.processing_time_secs)?;

                    if let Some(json) = result.json_response {
                        let json_str = serde_json::to_string(&json).map_err(|e| {
                            PyRuntimeError::new_err(format!("JSON error: {}", e))
                        })?;
                        dict.set_item("json_response", json_str)?;
                    }

                    if let Some(ref cost) = result.cost_breakdown {
                        let cost_py = json_value_to_py(py, cost).map_err(|e| {
                            PyRuntimeError::new_err(format!("Cost JSON error: {}", e))
                        })?;
                        dict.set_item("cost_breakdown", cost_py)?;
                    }

                    if let Some(error) = result.error {
                        dict.set_item("error", error)?;
                    }

                    py_results.append(dict)?;
                }

                Ok(py_results.into())
            })
        })
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to schedule async task. Is an event loop running? Error: {}",
                e
            ))
        })
    }

    fn __repr__(&self) -> String {
        "DocumentProcessor()".to_string()
    }
}

// ============================================================================
// HELPER: serde_json::Value -> Python object
// ============================================================================

fn json_value_to_py(py: Python, v: &serde_json::Value) -> PyResult<PyObject> {
    let s = serde_json::to_string(v)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(py.import_bound("json")?.call_method1("loads", (s,))?.to_object(py))
}