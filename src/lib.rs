pub mod ocr;
pub mod responses;

use pyo3::prelude::*;

#[pymodule]
fn genai_coroutines(py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize pyo3-log bridge: forwards all Rust `log` records
    // into Python's `logging` module under logger "genai_coroutines".
    //
    // Python usage:
    //   import logging
    //   logging.basicConfig(level=logging.INFO)
    //   # All Rust logs now visible!
    //
    // Or per-module control:
    //   logging.getLogger("genai_coroutines.ocr").setLevel(logging.DEBUG)
    //   logging.getLogger("genai_coroutines.responses").setLevel(logging.WARNING)
    pyo3_log::init();
    
    // Initialize the Tokio runtime for async support
    pyo3_asyncio::tokio::init_multi_thread_once();

    // Create ocr submodule
    let ocr_module = PyModule::new(py, "ocr")?;
    ocr_module.add_class::<ocr::PyDocumentConfig>()?;
    ocr_module.add_class::<ocr::PyDocumentProcessor>()?;
    ocr_module.add("__version__", "1.0.0")?;
    m.add_submodule(ocr_module)?;
    
    // Register submodule in sys.modules for proper Python import
    py.import("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines.ocr", ocr_module)?;

    // Create responses submodule
    let responses_module = PyModule::new(py, "responses")?;
    responses_module.add_class::<responses::PyResponsesRequest>()?;
    responses_module.add_class::<responses::PyResponsesProcessor>()?;
    responses_module.add("__version__", "1.0.0")?;
    m.add_submodule(responses_module)?;
    
    // Register submodule in sys.modules for proper Python import
    py.import("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines.responses", responses_module)?;

    m.add("__version__", "1.0.0")?;
    Ok(())
}
