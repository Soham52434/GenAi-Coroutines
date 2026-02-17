pub mod ocr;
pub mod responses;

use pyo3::prelude::*;

#[pymodule]
fn genai_coroutines(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize pyo3-log bridge: forwards all Rust `log` records
    // into Python's `logging` module under logger "genai_coroutines".
    //
    // Python usage:
    //   import logging
    //   logging.basicConfig(level=logging.INFO)
    //
    // Or per-module control:
    //   logging.getLogger("genai_coroutines.ocr").setLevel(logging.DEBUG)
    //   logging.getLogger("genai_coroutines.responses").setLevel(logging.WARNING)
    pyo3_log::init();

    // Initialize a multi-threaded Tokio runtime for async support.
    // pyo3-async-runtimes 0.21 uses get_runtime() — there is no
    // init_multi_thread_once(). The runtime is lazily created on first
    // access, so calling get_runtime() here eagerly boots it at import time.
    let _ = pyo3_async_runtimes::tokio::get_runtime();

    let py = m.py();

    // ── OCR submodule ──────────────────────────────────────────────────────
    let ocr_module = PyModule::new(py, "ocr")?;
    ocr_module.add_class::<ocr::PyDocumentConfig>()?;
    ocr_module.add_class::<ocr::PyDocumentProcessor>()?;
    ocr_module.add("__version__", "1.0.0")?;
    m.add_submodule(&ocr_module)?;

    // Register in sys.modules so `from genai_coroutines.ocr import ...` works
    py.import("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines.ocr", &ocr_module)?;

    // ── Responses submodule ────────────────────────────────────────────────
    let responses_module = PyModule::new(py, "responses")?;
    responses_module.add_class::<responses::PyResponsesRequest>()?;
    responses_module.add_class::<responses::PyResponsesProcessor>()?;
    responses_module.add("__version__", "1.0.0")?;
    m.add_submodule(&responses_module)?;

    // Register in sys.modules so `from genai_coroutines.responses import ...` works
    py.import("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines.responses", &responses_module)?;

    m.add("__version__", "1.0.0")?;
    Ok(())
}