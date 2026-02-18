use log::{debug, error, info, warn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::Rng;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// ============================================================================
// PUBLIC API
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIChatBatchRequest {
    pub api_key: String,
    pub system_prompt: String,
    pub user_prompts: Vec<String>,
    pub model: String,
    pub response_format: Value,
    pub timeout_secs: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_output_tokens: Option<u32>,
    pub store: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub reasoning_summary: Option<String>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub previous_response_id: Option<String>,
    pub include: Option<Vec<String>>,
    pub truncation: Option<String>,
    pub metadata: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub service_tier: Option<String>,
    pub stream: Option<bool>,
    pub max_retries: u32,
    pub retry_delay_min_ms: u64,
    pub retry_delay_max_ms: u64,
    pub max_concurrent_requests: usize,
}

// ============================================================================
// ERROR
// ============================================================================

#[derive(Debug, Clone)]
struct OpenAIErrorDetail {
    message: String,
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}

fn extract_error_detail(json_resp: &Value) -> OpenAIErrorDetail {
    let error_obj = json_resp.get("error");
    OpenAIErrorDetail {
        message: error_obj
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error")
            .to_string(),
        error_type: error_obj
            .and_then(|e| e.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("unknown")
            .to_string(),
        param: error_obj
            .and_then(|e| e.get("param"))
            .and_then(|p| p.as_str())
            .map(String::from),
        code: error_obj
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_str())
            .map(String::from),
    }
}

fn format_error_string(status: u16, d: &OpenAIErrorDetail) -> String {
    let mut s = format!("HTTP {} [{}]", status, d.error_type);
    if let Some(ref p) = d.param {
        s.push_str(&format!(": param={}", p));
    }
    if let Some(ref c) = d.code {
        s.push_str(&format!(": code={}", c));
    }
    s.push_str(&format!(": {}", d.message));
    s
}

// ============================================================================
// BATCH RESULT
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum BatchItemResult {
    Success {
        output: Value,
        usage: Option<Value>,
    },
    Error {
        message: String,
        error_type: String,
        param: Option<String>,
        code: Option<String>,
        is_retriable: bool,
        attempts: u32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIChatBatchResponse {
    pub results: Vec<BatchItemResult>,
    pub total_success: usize,
    pub total_errors: usize,
}

// ============================================================================
// INTERNAL
// ============================================================================

const OPENAI_URL: &str = "https://api.openai.com/v1/responses";

#[derive(Debug)]
enum ApiError {
    Retriable { detail: OpenAIErrorDetail, status: u16 },
    Fatal { detail: OpenAIErrorDetail, status: u16 },
}

impl ApiError {
    fn is_retriable(&self) -> bool {
        matches!(self, ApiError::Retriable { .. })
    }
    fn detail(&self) -> &OpenAIErrorDetail {
        match self {
            ApiError::Retriable { detail, .. } | ApiError::Fatal { detail, .. } => detail,
        }
    }
    fn status(&self) -> u16 {
        match self {
            ApiError::Retriable { status, .. } | ApiError::Fatal { status, .. } => *status,
        }
    }
}

// ============================================================================
// ENTRY POINT
// ============================================================================

pub async fn openai_chat_batch(
    req: OpenAIChatBatchRequest,
) -> Result<OpenAIChatBatchResponse, String> {
    let n = req.user_prompts.len();
    info!(
        "[llm] batch_start | prompts={} model={} timeout={}s",
        n, req.model, req.timeout_secs
    );

    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(10)
        .build()
        .map_err(|e| format!("HTTP client init failed: {e}"))?;

    let shared = Arc::new(SharedRequestState {
        api_key: req.api_key,
        system_prompt: req.system_prompt,
        model: req.model,
        response_format: req.response_format,
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_output_tokens,
        store: req.store,
        reasoning_effort: req.reasoning_effort,
        reasoning_summary: req.reasoning_summary,
        tools: req.tools,
        tool_choice: req.tool_choice,
        previous_response_id: req.previous_response_id,
        include: req.include,
        truncation: req.truncation,
        metadata: req.metadata,
        parallel_tool_calls: req.parallel_tool_calls,
        service_tier: req.service_tier,
        stream: req.stream,
        max_retries: req.max_retries,
        retry_delay_min_ms: req.retry_delay_min_ms,
        retry_delay_max_ms: req.retry_delay_max_ms,
    });
    let timeout = Duration::from_secs(req.timeout_secs);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(req.max_concurrent_requests));

    let mut handles = Vec::with_capacity(n);
    for (i, prompt) in req.user_prompts.into_iter().enumerate() {
        let c = client.clone();
        let s = Arc::clone(&shared);
        let sem = Arc::clone(&semaphore);
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            (i, call_with_retry(&c, &s, &prompt, timeout, i).await)
        }));
    }

    let mut ordered: Vec<Option<BatchItemResult>> = vec![None; n];
    let (mut ok, mut err) = (0usize, 0usize);

    for h in handles {
        let (i, result) = h.await.map_err(|e| format!("Join error: {e}"))?;
        ordered[i] = Some(match result {
            Ok(((json, usage), attempts)) => {
                ok += 1;
                info!("[llm] call_ok | index={} attempts={}", i, attempts);
                BatchItemResult::Success { output: json, usage }
            }
            Err((detail, status, retriable, attempts)) => {
                err += 1;
                error!(
                    "[llm] call_fail | index={} status={} type={} param={:?} code={:?} msg={} attempts={}",
                    i, status, detail.error_type, detail.param, detail.code, detail.message, attempts
                );
                BatchItemResult::Error {
                    message: format_error_string(status, &detail),
                    error_type: detail.error_type,
                    param: detail.param,
                    code: detail.code,
                    is_retriable: retriable,
                    attempts,
                }
            }
        });
    }

    let results: Vec<BatchItemResult> = ordered
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            opt.unwrap_or_else(|| {
                error!("[llm] task_missing | index={}", i);
                BatchItemResult::Error {
                    message: format!("Task {} did not complete", i),
                    error_type: "internal_error".into(),
                    param: None,
                    code: None,
                    is_retriable: false,
                    attempts: 0,
                }
            })
        })
        .collect();

    info!("[llm] batch_done | ok={}/{} errors={}", ok, n, err);
    Ok(OpenAIChatBatchResponse {
        results,
        total_success: ok,
        total_errors: err,
    })
}

struct SharedRequestState {
    api_key: String,
    system_prompt: String,
    model: String,
    response_format: Value,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_output_tokens: Option<u32>,
    store: Option<bool>,
    reasoning_effort: Option<String>,
    reasoning_summary: Option<String>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    previous_response_id: Option<String>,
    include: Option<Vec<String>>,
    truncation: Option<String>,
    metadata: Option<Value>,
    parallel_tool_calls: Option<bool>,
    service_tier: Option<String>,
    stream: Option<bool>,
    max_retries: u32,
    retry_delay_min_ms: u64,
    retry_delay_max_ms: u64,
}

// ============================================================================
// RETRY
// ============================================================================

async fn call_with_retry(
    client: &Client,
    state: &SharedRequestState,
    user_prompt: &str,
    timeout: Duration,
    index: usize,
) -> Result<((Value, Option<Value>), u32), (OpenAIErrorDetail, u16, bool, u32)> {
    let max_retries = state.max_retries;
    let min_ms = state.retry_delay_min_ms;
    let max_ms = state.retry_delay_max_ms;

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match call_openai(client, state, user_prompt, timeout).await {
            Ok((resp, usage)) => return Ok(((resp, usage), attempt)),
            Err(err) => {
                if !err.is_retriable() || attempt >= max_retries {
                    let retriable = err.is_retriable();
                    let status = err.status();
                    let detail = match err {
                        ApiError::Retriable { detail, .. } | ApiError::Fatal { detail, .. } => {
                            detail
                        }
                    };
                    return Err((detail, status, retriable, attempt));
                }
                let base = min_ms * 2u64.pow(attempt - 1);
                let jitter_range = base / 2;
                let jitter: i64 = rand::thread_rng()
                    .gen_range(-(jitter_range as i64)..=(jitter_range as i64));
                let delay =
                    ((base as i64 + jitter).max(min_ms as i64) as u64).min(max_ms);
                warn!(
                    "[llm] retry | index={} attempt={}/{} delay={}ms type={} msg={}",
                    index,
                    attempt,
                    max_retries,
                    delay,
                    err.detail().error_type,
                    err.detail().message
                );
                sleep(Duration::from_millis(delay)).await;
            }
        }
    }
}

// ============================================================================
// SINGLE API CALL
// ============================================================================

async fn call_openai(
    client: &Client,
    s: &SharedRequestState,
    user_prompt: &str,
    timeout: Duration,
) -> Result<(Value, Option<Value>), ApiError> {
    let mut input = user_prompt.to_string();

    if let Some(t) = s
        .response_format
        .as_object()
        .and_then(|o| o.get("type"))
        .and_then(|v| v.as_str())
    {
        if t == "json_object" && !input.to_lowercase().contains("json") {
            input.push_str(" Respond in JSON format.");
        }
    }

    let mut body = json!({
        "model": s.model,
        "instructions": s.system_prompt,
        "input": input,
    });

    if !s.response_format.is_null() {
        if let Some(obj) = s.response_format.as_object() {
            if let Some(type_val) = obj.get("type").and_then(|v| v.as_str()) {
                match type_val {
                    "json_schema" => {
                        if let Some(schema_obj) = obj.get("json_schema") {
                            let name = schema_obj
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("response");
                            if let Some(schema) = schema_obj.get("schema") {
                                let strict = schema_obj
                                    .get("strict")
                                    .and_then(|s| s.as_bool())
                                    .unwrap_or(true);
                                body["text"] = json!({"format": {
                                    "type": "json_schema",
                                    "name": name,
                                    "schema": schema.clone(),
                                    "strict": strict
                                }});
                            } else {
                                body["text"] = json!({"format": {"type": "json_object"}});
                            }
                        } else {
                            body["text"] = json!({"format": {"type": "json_object"}});
                        }
                    }
                    "json_object" => {
                        body["text"] = json!({"format": {"type": "json_object"}});
                    }
                    _ => {
                        body["text"] = json!({"format": {"type": "text"}});
                    }
                }
            }
        }
    }

    if let Some(v) = s.temperature {
        body["temperature"] = json!(v);
    }
    if let Some(v) = s.top_p {
        body["top_p"] = json!(v);
    }
    if let Some(v) = s.max_output_tokens {
        body["max_output_tokens"] = json!(v);
    }
    if let Some(v) = s.store {
        body["store"] = json!(v);
    }
    if let Some(v) = s.stream {
        body["stream"] = json!(v);
    }
    if let Some(ref v) = s.truncation {
        body["truncation"] = json!(v);
    }
    if let Some(ref v) = s.service_tier {
        body["service_tier"] = json!(v);
    }
    if let Some(v) = s.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(v);
    }

    if s.reasoning_effort.is_some() || s.reasoning_summary.is_some() {
        let mut r = json!({});
        if let Some(ref e) = s.reasoning_effort {
            r["effort"] = json!(e);
        }
        if let Some(ref su) = s.reasoning_summary {
            r["summary"] = json!(su);
        }
        body["reasoning"] = r;
    }

    if let Some(ref v) = s.tools {
        body["tools"] = v.clone();
    }
    if let Some(ref v) = s.tool_choice {
        body["tool_choice"] = v.clone();
    }
    if let Some(ref v) = s.previous_response_id {
        body["previous_response_id"] = json!(v);
    }
    if let Some(ref v) = s.include {
        body["include"] = json!(v);
    }
    if let Some(ref v) = s.metadata {
        body["metadata"] = v.clone();
    }

    debug!("[llm] request_send | model={}", s.model);

    let response = tokio::time::timeout(
        timeout,
        client
            .post(OPENAI_URL)
            .bearer_auth(&s.api_key)
            .json(&body)
            .send(),
    )
    .await
    .map_err(|_| ApiError::Retriable {
        detail: OpenAIErrorDetail {
            message: format!("Timeout after {}s", timeout.as_secs()),
            error_type: "timeout".into(),
            param: None,
            code: None,
        },
        status: 0,
    })?
    .map_err(|e| ApiError::Retriable {
        detail: OpenAIErrorDetail {
            message: format!("Network error: {e}"),
            error_type: "network_error".into(),
            param: None,
            code: None,
        },
        status: 0,
    })?;

    let status = response.status();
    let json_resp: Value = response.json().await.map_err(|e| ApiError::Fatal {
        detail: OpenAIErrorDetail {
            message: format!("Invalid JSON: {e}"),
            error_type: "parse_error".into(),
            param: None,
            code: None,
        },
        status: status.as_u16(),
    })?;

    match status {
        StatusCode::OK => {
            debug!("[llm] response_ok | status=200");
            let usage = json_resp.get("usage").cloned();
            Ok((json_resp, usage))
        }
        StatusCode::TOO_MANY_REQUESTS
        | StatusCode::INTERNAL_SERVER_ERROR
        | StatusCode::SERVICE_UNAVAILABLE
        | StatusCode::GATEWAY_TIMEOUT => {
            let d = extract_error_detail(&json_resp);
            error!(
                "[llm] api_error | status={} type={} param={:?} code={:?} msg={}",
                status.as_u16(),
                d.error_type,
                d.param,
                d.code,
                d.message
            );
            Err(ApiError::Retriable {
                detail: d,
                status: status.as_u16(),
            })
        }
        _ => {
            let d = extract_error_detail(&json_resp);
            error!(
                "[llm] api_error | status={} type={} param={:?} code={:?} msg={}",
                status.as_u16(),
                d.error_type,
                d.param,
                d.code,
                d.message
            );
            Err(ApiError::Fatal {
                detail: d,
                status: status.as_u16(),
            })
        }
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

#[pyclass(name = "ResponsesRequest")]
#[derive(Clone)]
pub struct PyResponsesRequest {
    inner: OpenAIChatBatchRequest,
}

#[pymethods]
impl PyResponsesRequest {
    #[new]
    #[pyo3(signature = (
        api_key,
        system_prompt,
        user_prompts,
        model,
        response_format,
        timeout_secs=60,
        temperature=None,
        top_p=None,
        max_output_tokens=None,
        store=None,
        reasoning_effort=None,
        reasoning_summary=None,
        tools=None,
        tool_choice=None,
        previous_response_id=None,
        include=None,
        truncation=None,
        metadata=None,
        parallel_tool_calls=None,
        service_tier=None,
        stream=None,
        max_retries=5,
        retry_delay_min_ms=1000,
        retry_delay_max_ms=60000,
        max_concurrent_requests=10
    ))]
    fn new(
        api_key: String,
        system_prompt: String,
        user_prompts: Vec<String>,
        model: String,
        response_format: &Bound<'_, PyAny>,
        timeout_secs: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        max_output_tokens: Option<u32>,
        store: Option<bool>,
        reasoning_effort: Option<String>,
        reasoning_summary: Option<String>,
        tools: Option<&Bound<'_, PyAny>>,
        tool_choice: Option<&Bound<'_, PyAny>>,
        previous_response_id: Option<String>,
        include: Option<Vec<String>>,
        truncation: Option<String>,
        metadata: Option<&Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        service_tier: Option<String>,
        stream: Option<bool>,
        max_retries: u32,
        retry_delay_min_ms: u64,
        retry_delay_max_ms: u64,
        max_concurrent_requests: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OpenAIChatBatchRequest {
                api_key,
                system_prompt,
                user_prompts,
                model,
                response_format: py_to_json(response_format, "response_format")?,
                timeout_secs,
                temperature,
                top_p,
                max_output_tokens,
                store,
                reasoning_effort,
                reasoning_summary,
                tools: py_opt_to_json(tools, "tools")?,
                tool_choice: py_opt_to_json(tool_choice, "tool_choice")?,
                previous_response_id,
                include,
                truncation,
                metadata: py_opt_to_json(metadata, "metadata")?,
                parallel_tool_calls,
                service_tier,
                stream,
                max_retries,
                retry_delay_min_ms,
                retry_delay_max_ms,
                max_concurrent_requests,
            },
        })
    }

    #[getter] fn user_prompts(&self) -> Vec<String> { self.inner.user_prompts.clone() }
    #[getter] fn model(&self) -> String { self.inner.model.clone() }
    #[getter] fn system_prompt(&self) -> String { self.inner.system_prompt.clone() }
    #[getter] fn timeout_secs(&self) -> u64 { self.inner.timeout_secs }
    #[getter] fn temperature(&self) -> Option<f64> { self.inner.temperature }
    #[getter] fn top_p(&self) -> Option<f64> { self.inner.top_p }
    #[getter] fn max_output_tokens(&self) -> Option<u32> { self.inner.max_output_tokens }
    #[getter] fn store(&self) -> Option<bool> { self.inner.store }
    #[getter] fn reasoning_effort(&self) -> Option<String> { self.inner.reasoning_effort.clone() }
    #[getter] fn reasoning_summary(&self) -> Option<String> { self.inner.reasoning_summary.clone() }
    #[getter] fn previous_response_id(&self) -> Option<String> { self.inner.previous_response_id.clone() }
    #[getter] fn include(&self) -> Option<Vec<String>> { self.inner.include.clone() }
    #[getter] fn truncation(&self) -> Option<String> { self.inner.truncation.clone() }
    #[getter] fn parallel_tool_calls(&self) -> Option<bool> { self.inner.parallel_tool_calls }
    #[getter] fn service_tier(&self) -> Option<String> { self.inner.service_tier.clone() }
    #[getter] fn stream(&self) -> Option<bool> { self.inner.stream }
    #[getter] fn max_retries(&self) -> u32 { self.inner.max_retries }
    #[getter] fn retry_delay_min_ms(&self) -> u64 { self.inner.retry_delay_min_ms }
    #[getter] fn retry_delay_max_ms(&self) -> u64 { self.inner.retry_delay_max_ms }
    #[getter] fn max_concurrent_requests(&self) -> usize { self.inner.max_concurrent_requests }

    #[getter]
    fn response_format(&self, py: Python) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.response_format)
    }
    #[getter]
    fn tools(&self, py: Python) -> PyResult<Option<PyObject>> {
        json_opt_to_py(py, &self.inner.tools)
    }
    #[getter]
    fn tool_choice(&self, py: Python) -> PyResult<Option<PyObject>> {
        json_opt_to_py(py, &self.inner.tool_choice)
    }
    #[getter]
    fn metadata(&self, py: Python) -> PyResult<Option<PyObject>> {
        json_opt_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "ResponsesRequest(model='{}', prompts={}, timeout={}s)",
            self.inner.model,
            self.inner.user_prompts.len(),
            self.inner.timeout_secs,
        )
    }
}

#[pyclass(name = "ResponsesProcessor")]
pub struct PyResponsesProcessor;

#[pymethods]
impl PyResponsesProcessor {
    #[new]
    fn new() -> Self {
        Self
    }

    fn process_batch<'py>(
        &self,
        py: Python<'py>,
        request: PyResponsesRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let req = request.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = openai_chat_batch(req).await;
            Python::with_gil(|py| -> PyResult<PyObject> {
                match result {
                    Ok(resp) => {
                        let dict = PyDict::new(py);
                        dict.set_item("total_success", resp.total_success)?;
                        dict.set_item("total_errors", resp.total_errors)?;
                        let py_results = PyList::empty(py);
                        for item in resp.results {
                            let d = PyDict::new(py);
                            match item {
                                BatchItemResult::Success { output, usage } => {
                                    d.set_item("success", true)?;
                                    d.set_item(
                                        "raw_response",
                                        serde_json::to_string(&output).map_err(|e| {
                                            PyRuntimeError::new_err(format!("JSON: {}", e))
                                        })?,
                                    )?;
                                    if let Some(ref usage_val) = usage {
                                        let usage_py = json_to_py(py, usage_val)?;
                                        d.set_item("usage", usage_py)?;
                                    }
                                }
                                BatchItemResult::Error {
                                    message,
                                    error_type,
                                    param,
                                    code,
                                    is_retriable,
                                    attempts,
                                } => {
                                    d.set_item("success", false)?;
                                    d.set_item("error", message)?;
                                    d.set_item("error_type", error_type)?;
                                    d.set_item("param", param)?;
                                    d.set_item("code", code)?;
                                    d.set_item("is_retriable", is_retriable)?;
                                    d.set_item("attempts", attempts)?;
                                }
                            }
                            py_results.append(d)?;
                        }
                        dict.set_item("results", py_results)?;
                        Ok(dict.into())
                    }
                    Err(e) => Err(PyRuntimeError::new_err(format!("Batch failed: {}", e))),
                }
            })
        })
        .map_err(|e| PyRuntimeError::new_err(format!("No event loop? {}", e)))
    }

    fn __repr__(&self) -> String {
        "ResponsesProcessor()".into()
    }
}

// ============================================================================
// HELPERS: Python <-> JSON
// ============================================================================

fn py_to_json(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Value> {
    let s: String = obj
        .py()
        .import("json")?
        .call_method1("dumps", (obj,))?
        .extract()?;
    serde_json::from_str(&s)
        .map_err(|e| PyValueError::new_err(format!("Invalid {} JSON: {}", name, e)))
}

fn py_opt_to_json(obj: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Option<Value>> {
    match obj {
        Some(o) => Ok(Some(py_to_json(o, name)?)),
        None => Ok(None),
    }
}

fn json_to_py(py: Python, v: &Value) -> PyResult<PyObject> {
    let s = serde_json::to_string(v)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(py
        .import("json")?
        .call_method1("loads", (s,))?
        .unbind())
}

fn json_opt_to_py(py: Python, v: &Option<Value>) -> PyResult<Option<PyObject>> {
    match v {
        Some(val) => Ok(Some(json_to_py(py, val)?)),
        None => Ok(None),
    }
}