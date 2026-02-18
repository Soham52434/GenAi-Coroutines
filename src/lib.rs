pub mod ocr;
pub mod responses;

use pyo3::prelude::*;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    // Eagerly boot the Tokio runtime at import time.
    let _ = pyo3_async_runtimes::tokio::get_runtime();

    let py = m.py();

    // ── OCR submodule ──────────────────────────────────────────────────────
    let ocr_module = PyModule::new(py, "ocr")?;
    ocr_module.add_class::<ocr::PyDocumentConfig>()?;
    ocr_module.add_class::<ocr::PyDocumentProcessor>()?;
    ocr_module.add("__version__", "1.0.0")?;
    m.add_submodule(&ocr_module)?;

    // Register in sys.modules so `from genai_coroutines._internal.ocr import ...` works
    py.import("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines._internal.ocr", &ocr_module)?;

    // ── Responses submodule ────────────────────────────────────────────────
    let responses_module = PyModule::new(py, "responses")?;
    responses_module.add_class::<responses::PyResponsesRequest>()?;
    responses_module.add_class::<responses::PyResponsesProcessor>()?;
    responses_module.add("__version__", "1.0.0")?;
    m.add_submodule(&responses_module)?;

    // Register in sys.modules so `from genai_coroutines._internal.responses import ...` works
    py.import("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines._internal.responses", &responses_module)?;

    m.add("__version__", "1.0.0")?;
    Ok(())
}