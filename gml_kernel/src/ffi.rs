use crate::gh05t3::core_loop::create_gh05t3_agent;
use crate::kernel::{
    executor::execute_block,
    payload::{KernelRunSummary, ModelCallPayload, MultiModelCallPayload},
    KernelState,
};
use std::collections::HashMap;

#[no_mangle]
pub extern "C" fn gh05t3_run_core_loop() -> *mut std::os::raw::c_char {
    let mut kernel = KernelState::new();
    let mut agent = create_gh05t3_agent();

    let core_loop = agent.core_loop.clone();
    execute_block(&core_loop, &mut agent, &mut kernel);

    let summary = format!(
        "ticks={},short_term={:?}",
        kernel.tick, agent.memory.short_term
    );

    let c_string = std::ffi::CString::new(summary).unwrap();
    c_string.into_raw()
}

/// Same run as gh05t3_run_core_loop, but returns a real JSON envelope
/// (KernelRunSummary) instead of a Debug-formatted string, so Python can
/// actually parse it instead of guessing at Rust's {:?} format.
#[no_mangle]
pub extern "C" fn gh05t3_run_core_loop_json() -> *mut std::os::raw::c_char {
    let mut kernel = KernelState::new();
    let mut agent = create_gh05t3_agent();

    let core_loop = agent.core_loop.clone();
    execute_block(&core_loop, &mut agent, &mut kernel);

    let summary = KernelRunSummary {
        tick: kernel.tick,
        short_term: agent.memory.short_term,
    };

    std::ffi::CString::new(summary.to_json()).unwrap().into_raw()
}

/// v2 MODEL_CALL contract: builds the JSON envelope shared by the exported
/// FFI symbol below and by `kernel::executor::model_call` — same crate, so
/// no need to round-trip through the C ABI to call this from within Rust.
pub fn model_call_summary(
    backend: &str,
    prompt: &str,
    version: &str,
    meta: HashMap<String, String>,
) -> String {
    ModelCallPayload {
        backend: backend.to_string(),
        prompt: prompt.to_string(),
        version: version.to_string(),
        meta,
    }
    .to_json()
}

/// v4 multi-model MODEL_CALL contract: same in-crate-shared pattern as
/// model_call_summary above, one struct per contract version.
pub fn model_call_blend_summary(
    backends: Vec<String>,
    prompt: &str,
    version: &str,
    blend_strategy: &str,
    meta: HashMap<String, String>,
) -> String {
    MultiModelCallPayload {
        backends,
        prompt: prompt.to_string(),
        version: version.to_string(),
        blend_strategy: blend_strategy.to_string(),
        meta,
    }
    .to_json()
}

#[no_mangle]
pub extern "C" fn gh05t3_model_call(
    backend_ptr: *const std::os::raw::c_char,
    prompt_ptr: *const std::os::raw::c_char,
    version_ptr: *const std::os::raw::c_char,
) -> *mut std::os::raw::c_char {
    let (backend, prompt, version) = unsafe {
        (
            std::ffi::CStr::from_ptr(backend_ptr).to_string_lossy().into_owned(),
            std::ffi::CStr::from_ptr(prompt_ptr).to_string_lossy().into_owned(),
            std::ffi::CStr::from_ptr(version_ptr).to_string_lossy().into_owned(),
        )
    };

    let summary = model_call_summary(&backend, &prompt, &version, HashMap::new());
    std::ffi::CString::new(summary).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn gh05t3_free_string(ptr: *mut std::os::raw::c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(std::ffi::CString::from_raw(ptr));
    }
}

/// Real SIMD inference: computes one binarized layer's forward pass
/// (see binary_inference module -- signed accumulation, not XNOR+popcount,
/// since this model's activations are continuous, only weights are +-1).
/// Reads from and writes into caller-owned buffers (numpy arrays on the
/// Python side) -- no Rust-side allocation, so there's nothing to free.
/// Returns 0 on success, a negative error code otherwise.
#[no_mangle]
pub extern "C" fn gh05t3_binary_forward_layer(
    input_ptr: *const f32,
    input_len: usize,
    packed_weights_ptr: *const u64,
    weights_len: usize,
    out_features: usize,
    k_packed: usize,
    output_ptr: *mut f32,
    output_len: usize,
) -> i32 {
    if input_ptr.is_null() || packed_weights_ptr.is_null() || output_ptr.is_null() {
        return -1;
    }
    if input_len < k_packed * 64 {
        return -2;
    }
    if weights_len != out_features * k_packed {
        return -3;
    }
    if output_len != out_features {
        return -4;
    }

    let input = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    let packed_weights = unsafe { std::slice::from_raw_parts(packed_weights_ptr, weights_len) };
    let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, output_len) };

    crate::binary_inference::forward_layer(input, packed_weights, out_features, k_packed, output);
    0
}
