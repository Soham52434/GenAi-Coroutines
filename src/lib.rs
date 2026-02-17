pub mod ocr;
pub mod responses;

use pyo3::prelude::*;

#[pymodule]
fn genai_coroutines(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize pyo3-log bridge
    pyo3_log::init();
    
    // Initialize the Tokio runtime for async support
    pyo3_async_runtimes::tokio::init_multi_thread_once();

    let py = m.py();

    // Create ocr submodule
    let ocr_module = PyModule::new_bound(py, "ocr")?;
    ocr_module.add_class::<ocr::PyDocumentConfig>()?;
    ocr_module.add_class::<ocr::PyDocumentProcessor>()?;
    ocr_module.add("__version__", "1.0.0")?;
    m.add_submodule(&ocr_module)?;
    
    // Register submodule in sys.modules for proper Python import
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines.ocr", &ocr_module)?;

    // Create responses submodule
    let responses_module = PyModule::new_bound(py, "responses")?;
    responses_module.add_class::<responses::PyResponsesRequest>()?;
    responses_module.add_class::<responses::PyResponsesProcessor>()?;
    responses_module.add("__version__", "1.0.0")?;
    m.add_submodule(&responses_module)?;
    
    // Register submodule in sys.modules for proper Python import
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("genai_coroutines.responses", &responses_module)?;

    m.add("__version__", "1.0.0")?;
    Ok(())
}
