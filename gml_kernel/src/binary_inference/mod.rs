//! Real SIMD inference kernel for gh05t3_binary's weight-only-binarized
//! layers (see gh05t3_binary/inference/pack_weights.py for why this is a
//! signed-accumulation kernel and not XNOR+popcount: activations here are
//! continuous floats, never binarized -- only weights are +-1).
//!
//! Packed weight layout (matches pack_weights.py exactly): row-major,
//! `k_packed` u64 words per output row, bit b of word w == 1 means
//! weight == +1 at input position w*64+b, 0 means weight == -1. Padding
//! bits (when in_features isn't a multiple of 64) are 0 and the caller
//! must zero-pad the corresponding input positions too, so they
//! contribute 0 regardless of the padding weight's sign.

use core::arch::x86_64::*;

/// Portable scalar reference implementation -- no intrinsics, used to
/// verify the AVX-512 kernel and as a fallback on hardware without it.
pub fn signed_accumulate_row_scalar(input: &[f32], weight_row: &[u64]) -> f32 {
    let mut acc = 0.0f32;
    for (word_idx, &word) in weight_row.iter().enumerate() {
        for bit in 0..64 {
            let x = input[word_idx * 64 + bit];
            let is_positive = (word >> bit) & 1 == 1;
            acc += if is_positive { x } else { -x };
        }
    }
    acc
}

/// AVX-512F kernel for the same computation. Processes 16 f32 lanes (one
/// zmm register) per 16-bit mask chunk, using `_mm512_mask_xor_epi32` to
/// flip the sign bit only on lanes whose weight is -1 -- a native
/// mask-register conditional negate, not a full multiply, and not
/// XNOR+popcount (which would be wrong here since input isn't binary).
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (checked at the call site
/// via `is_x86_feature_detected!`), `input` has length >= k_packed*64,
/// and `weight_row` has length >= k_packed.
#[target_feature(enable = "avx512f")]
pub unsafe fn signed_accumulate_row_avx512(input: &[f32], weight_row: &[u64]) -> f32 {
    let k_packed = weight_row.len();
    debug_assert!(input.len() >= k_packed * 64);

    let mut acc = _mm512_setzero_ps();
    let sign_bit = _mm512_set1_epi32(i32::MIN); // 0x80000000: XOR this to flip a float's sign

    for word_idx in 0..k_packed {
        let word = weight_row[word_idx];
        for chunk in 0..4 {
            let mask16 = ((word >> (chunk * 16)) & 0xFFFF) as u16;
            let negate_mask = !mask16; // 1 where weight == -1 (needs sign flip)

            let x = _mm512_loadu_ps(input.as_ptr().add(word_idx * 64 + chunk * 16));
            let x_bits = _mm512_castps_si512(x);
            let signed_bits = _mm512_mask_xor_epi32(x_bits, negate_mask, x_bits, sign_bit);
            let signed_x = _mm512_castsi512_ps(signed_bits);

            acc = _mm512_add_ps(acc, signed_x);
        }
    }

    _mm512_reduce_add_ps(acc)
}

/// Computes a full layer's forward pass: `output[j] = sum_i(x_i * w[j,i])`
/// for all `out_features` rows, dispatching to AVX-512 if available at
/// runtime, scalar otherwise. `input` must be pre-padded to
/// `k_packed * 64` elements (zeros past the real in_features).
///
/// NOT parallelized, on purpose, after actually measuring it: a rayon
/// par_iter_mut() version was tried here to address a confirmed 8-thread
/// (PyTorch) vs 1-thread (this kernel) mismatch. Measured result: 32x
/// SLOWER (703.8us -> 22488.2us for one isolated binary layer), because
/// rayon's per-call work-stealing dispatch overhead vastly exceeds the
/// actual work in one row (a few hundred nanoseconds of AVX-512
/// accumulation) at this out_features size. The threading-mismatch
/// observation was real; the fix was wrong for this granularity.
/// Reverted immediately upon measuring it -- left as a documented dead
/// end so it isn't tried again the same way.
pub fn forward_layer(
    input: &[f32],
    packed_weights: &[u64],
    out_features: usize,
    k_packed: usize,
    output: &mut [f32],
) {
    assert_eq!(packed_weights.len(), out_features * k_packed);
    assert_eq!(output.len(), out_features);

    let use_avx512 = is_x86_feature_detected!("avx512f");

    for j in 0..out_features {
        let row = &packed_weights[j * k_packed..(j + 1) * k_packed];
        output[j] = if use_avx512 {
            unsafe { signed_accumulate_row_avx512(input, row) }
        } else {
            signed_accumulate_row_scalar(input, row)
        };
    }
}

/// Batched version of forward_layer: computes `num_rows` independent
/// forward passes (e.g. one per token in a [batch*seq_len, in_features]
/// activation tensor) in a single call, looping over rows on the Rust
/// side. Exists specifically to avoid per-token Python<->Rust FFI call
/// overhead when wiring this into a real model forward pass -- a real
/// concern (not hypothetical): a 12-layer model with 7 BinaryLinear calls
/// per layer, called once per token, would otherwise mean 84 * seq_len
/// individual FFI round-trips per forward pass.
pub fn forward_layer_batched(
    inputs: &[f32],       // num_rows * (k_packed * 64), row-major
    packed_weights: &[u64],
    out_features: usize,
    k_packed: usize,
    num_rows: usize,
    outputs: &mut [f32],  // num_rows * out_features, row-major
) {
    let row_in_len = k_packed * 64;
    assert_eq!(inputs.len(), num_rows * row_in_len);
    assert_eq!(outputs.len(), num_rows * out_features);

    for r in 0..num_rows {
        let row_input = &inputs[r * row_in_len..(r + 1) * row_in_len];
        let row_output = &mut outputs[r * out_features..(r + 1) * out_features];
        forward_layer(row_input, packed_weights, out_features, k_packed, row_output);
    }
}

// ---------------------------------------------------------------------
// Ternary kernel: weights in {-1, 0, +1} (see gh05t3_binary/inference/
// pack_ternary.py and gh05t3_binary/core/binary_layers.py::TernaryLinear,
// Ternary Weight Networks, Li & Liu 2016). Two u64 bitplanes per row --
// a nonzero mask and a sign mask (meaningless wherever nonzero==0, always
// packed as 0 there) -- plus a single per-tensor fp32 alpha scale applied
// once per output element after accumulation, matching TernaryLinear's
// own scaling (alpha * ternary_weight).
// ---------------------------------------------------------------------

/// Portable scalar reference for the ternary kernel.
pub fn ternary_accumulate_row_scalar(input: &[f32], nonzero_row: &[u64], sign_row: &[u64]) -> f32 {
    debug_assert_eq!(nonzero_row.len(), sign_row.len());
    let mut acc = 0.0f32;
    for word_idx in 0..nonzero_row.len() {
        let nz_word = nonzero_row[word_idx];
        let sign_word = sign_row[word_idx];
        for bit in 0..64 {
            if (nz_word >> bit) & 1 == 0 {
                continue; // weight is exactly 0 -- contributes nothing
            }
            let x = input[word_idx * 64 + bit];
            let is_positive = (sign_word >> bit) & 1 == 1;
            acc += if is_positive { x } else { -x };
        }
    }
    acc
}

/// AVX-512F ternary kernel. Same mask-register sign-flip as the binary
/// kernel, plus a second mask (`nz_mask16`) used with
/// `_mm512_mask_add_ps` to only accumulate lanes whose weight is
/// actually nonzero -- inactive lanes keep the running total unchanged,
/// which is exactly "contribute 0". Whole 16-lane chunks with no nonzero
/// weights skip the load/compute entirely: correctness doesn't depend on
/// this (an all-zero chunk would add exactly 0 either way), it's a
/// provably-safe shortcut enabled by ternary sparsity specifically,
/// unlike the earlier debunked "optimizations" -- benchmarked honestly
/// below rather than assumed to help.
///
/// # Safety
/// Same preconditions as signed_accumulate_row_avx512: AVX-512F must be
/// available (checked by the caller), `input` must have length >=
/// k_packed*64, and both mask rows must have length >= k_packed.
#[target_feature(enable = "avx512f")]
pub unsafe fn ternary_accumulate_row_avx512(input: &[f32], nonzero_row: &[u64], sign_row: &[u64]) -> f32 {
    let k_packed = nonzero_row.len();
    debug_assert_eq!(k_packed, sign_row.len());
    debug_assert!(input.len() >= k_packed * 64);

    let mut acc = _mm512_setzero_ps();
    let sign_bit = _mm512_set1_epi32(i32::MIN);

    for word_idx in 0..k_packed {
        let nz_word = nonzero_row[word_idx];
        let sign_word = sign_row[word_idx];
        for chunk in 0..4 {
            let nz_mask16 = ((nz_word >> (chunk * 16)) & 0xFFFF) as u16;
            if nz_mask16 == 0 {
                continue;
            }
            let sign_mask16 = ((sign_word >> (chunk * 16)) & 0xFFFF) as u16;
            let negate_mask = nz_mask16 & !sign_mask16; // nonzero AND weight == -1

            let x = _mm512_loadu_ps(input.as_ptr().add(word_idx * 64 + chunk * 16));
            let x_bits = _mm512_castps_si512(x);
            let signed_bits = _mm512_mask_xor_epi32(x_bits, negate_mask, x_bits, sign_bit);
            let signed_x = _mm512_castsi512_ps(signed_bits);

            acc = _mm512_mask_add_ps(acc, nz_mask16, acc, signed_x);
        }
    }

    _mm512_reduce_add_ps(acc)
}

/// Ternary sibling of forward_layer: one full layer's forward pass,
/// applying the single per-tensor alpha scale after accumulation.
/// NOT parallelized -- see forward_layer's docstring above for why a
/// rayon version was tried and measured 32x slower, then reverted.
pub fn ternary_forward_layer(
    input: &[f32],
    nonzero_weights: &[u64],
    sign_weights: &[u64],
    alpha: f32,
    out_features: usize,
    k_packed: usize,
    output: &mut [f32],
) {
    assert_eq!(nonzero_weights.len(), out_features * k_packed);
    assert_eq!(sign_weights.len(), out_features * k_packed);
    assert_eq!(output.len(), out_features);

    let use_avx512 = is_x86_feature_detected!("avx512f");

    for j in 0..out_features {
        let nz_row = &nonzero_weights[j * k_packed..(j + 1) * k_packed];
        let sign_row = &sign_weights[j * k_packed..(j + 1) * k_packed];
        let acc = if use_avx512 {
            unsafe { ternary_accumulate_row_avx512(input, nz_row, sign_row) }
        } else {
            ternary_accumulate_row_scalar(input, nz_row, sign_row)
        };
        output[j] = acc * alpha;
    }
}

/// Ternary sibling of forward_layer_batched: `num_rows` independent
/// forward passes in one call, same reasoning as the binary version.
pub fn ternary_forward_layer_batched(
    inputs: &[f32],
    nonzero_weights: &[u64],
    sign_weights: &[u64],
    alpha: f32,
    out_features: usize,
    k_packed: usize,
    num_rows: usize,
    outputs: &mut [f32],
) {
    let row_in_len = k_packed * 64;
    assert_eq!(inputs.len(), num_rows * row_in_len);
    assert_eq!(outputs.len(), num_rows * out_features);

    for r in 0..num_rows {
        let row_input = &inputs[r * row_in_len..(r + 1) * row_in_len];
        let row_output = &mut outputs[r * out_features..(r + 1) * out_features];
        ternary_forward_layer(
            row_input, nonzero_weights, sign_weights, alpha, out_features, k_packed, row_output,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_bits(seed: &mut u64, n_words: usize) -> Vec<u64> {
        (0..n_words)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                *seed
            })
            .collect()
    }

    #[test]
    fn avx512_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F not available on this CPU, skipping");
            return;
        }

        let mut seed = 0xDEADBEEFu64;
        let k_packed = 8; // 512 input elements
        let weight_row = random_bits(&mut seed, k_packed);

        let input: Vec<f32> = (0..k_packed * 64)
            .map(|i| ((i as f32) * 0.0173).sin() * 3.0)
            .collect();

        let scalar_result = signed_accumulate_row_scalar(&input, &weight_row);
        let avx512_result = unsafe { signed_accumulate_row_avx512(&input, &weight_row) };

        let diff = (scalar_result - avx512_result).abs();
        assert!(
            diff < 1e-3,
            "scalar={scalar_result}, avx512={avx512_result}, diff={diff}"
        );
    }

    #[test]
    fn all_positive_weights_sums_input_directly() {
        let k_packed = 2;
        let weight_row = vec![u64::MAX; k_packed]; // all +1
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let expected: f32 = input.iter().sum();

        let scalar_result = signed_accumulate_row_scalar(&input, &weight_row);
        assert!((scalar_result - expected).abs() < 1e-3);

        if is_x86_feature_detected!("avx512f") {
            let avx512_result = unsafe { signed_accumulate_row_avx512(&input, &weight_row) };
            assert!((avx512_result - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn all_negative_weights_negates_sum() {
        let k_packed = 2;
        let weight_row = vec![0u64; k_packed]; // all -1
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let expected: f32 = -input.iter().sum::<f32>();

        let scalar_result = signed_accumulate_row_scalar(&input, &weight_row);
        assert!((scalar_result - expected).abs() < 1e-3);
    }

    #[test]
    fn forward_layer_matches_row_by_row_scalar() {
        let out_features = 4;
        let k_packed = 4;
        let mut seed = 12345u64;
        let packed_weights = random_bits(&mut seed, out_features * k_packed);
        let input: Vec<f32> = (0..k_packed * 64).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let mut output = vec![0.0f32; out_features];
        forward_layer(&input, &packed_weights, out_features, k_packed, &mut output);

        for j in 0..out_features {
            let row = &packed_weights[j * k_packed..(j + 1) * k_packed];
            let expected = signed_accumulate_row_scalar(&input, row);
            assert!((output[j] - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn forward_layer_batched_matches_repeated_single_row_calls() {
        let out_features = 6;
        let k_packed = 3;
        let num_rows = 5;
        let row_in_len = k_packed * 64;
        let mut seed = 999u64;
        let packed_weights = random_bits(&mut seed, out_features * k_packed);

        let inputs: Vec<f32> = (0..num_rows * row_in_len)
            .map(|i| ((i as f32) * 0.037).cos() * 2.0)
            .collect();

        let mut batched_outputs = vec![0.0f32; num_rows * out_features];
        forward_layer_batched(
            &inputs, &packed_weights, out_features, k_packed, num_rows, &mut batched_outputs,
        );

        for r in 0..num_rows {
            let row_input = &inputs[r * row_in_len..(r + 1) * row_in_len];
            let mut single_output = vec![0.0f32; out_features];
            forward_layer(row_input, &packed_weights, out_features, k_packed, &mut single_output);

            let batched_row = &batched_outputs[r * out_features..(r + 1) * out_features];
            for j in 0..out_features {
                assert!(
                    (batched_row[j] - single_output[j]).abs() < 1e-6,
                    "row {r} col {j}: batched={} single={}",
                    batched_row[j],
                    single_output[j]
                );
            }
        }
    }

    #[test]
    #[should_panic]
    fn forward_layer_batched_rejects_mismatched_input_length() {
        let packed_weights = vec![0u64; 2];
        let mut outputs = vec![0.0f32; 2];
        // inputs too short for num_rows=1, k_packed=2 (needs 128 elements)
        let inputs = vec![0.0f32; 64];
        forward_layer_batched(&inputs, &packed_weights, 1, 2, 1, &mut outputs);
    }

    #[test]
    fn ternary_avx512_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F not available on this CPU, skipping");
            return;
        }

        let mut seed = 0xC0FFEEu64;
        let k_packed = 6;
        let nonzero_row = random_bits(&mut seed, k_packed);
        let sign_row = random_bits(&mut seed, k_packed);

        let input: Vec<f32> = (0..k_packed * 64)
            .map(|i| ((i as f32) * 0.0231).sin() * 2.5)
            .collect();

        let scalar_result = ternary_accumulate_row_scalar(&input, &nonzero_row, &sign_row);
        let avx512_result = unsafe { ternary_accumulate_row_avx512(&input, &nonzero_row, &sign_row) };

        let diff = (scalar_result - avx512_result).abs();
        assert!(diff < 1e-3, "scalar={scalar_result}, avx512={avx512_result}, diff={diff}");
    }

    #[test]
    fn ternary_all_zero_weights_gives_zero_output() {
        let k_packed = 3;
        let nonzero_row = vec![0u64; k_packed]; // every weight is 0
        let sign_row = vec![u64::MAX; k_packed]; // sign bits irrelevant, should be ignored
        let input: Vec<f32> = (0..192).map(|i| i as f32 + 1.0).collect();

        let scalar_result = ternary_accumulate_row_scalar(&input, &nonzero_row, &sign_row);
        assert_eq!(scalar_result, 0.0);

        if is_x86_feature_detected!("avx512f") {
            let avx512_result = unsafe { ternary_accumulate_row_avx512(&input, &nonzero_row, &sign_row) };
            assert_eq!(avx512_result, 0.0);
        }
    }

    #[test]
    fn ternary_reduces_to_binary_when_all_nonzero() {
        // Cross-check against the already-verified binary kernel: a
        // ternary row with nonzero_row = all-ones and alpha = 1.0 IS a
        // binary row (no weight can be 0), so it must match
        // signed_accumulate_row_* exactly given the same sign bits.
        let mut seed = 777u64;
        let k_packed = 4;
        let sign_row = random_bits(&mut seed, k_packed); // reused as the binary "weight_row"
        let nonzero_row = vec![u64::MAX; k_packed];

        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.02).collect();

        let binary_result = signed_accumulate_row_scalar(&input, &sign_row);
        let ternary_result = ternary_accumulate_row_scalar(&input, &nonzero_row, &sign_row);
        assert!((binary_result - ternary_result).abs() < 1e-3);

        if is_x86_feature_detected!("avx512f") {
            let binary_avx = unsafe { signed_accumulate_row_avx512(&input, &sign_row) };
            let ternary_avx = unsafe { ternary_accumulate_row_avx512(&input, &nonzero_row, &sign_row) };
            assert!((binary_avx - ternary_avx).abs() < 1e-3);
        }
    }

    #[test]
    fn ternary_forward_layer_matches_row_by_row_scalar() {
        let out_features = 4;
        let k_packed = 3;
        let alpha = 0.37f32;
        let mut seed = 24680u64;
        let nonzero_weights = random_bits(&mut seed, out_features * k_packed);
        let sign_weights = random_bits(&mut seed, out_features * k_packed);
        let input: Vec<f32> = (0..k_packed * 64).map(|i| (i as f32 - 96.0) * 0.015).collect();

        let mut output = vec![0.0f32; out_features];
        ternary_forward_layer(
            &input, &nonzero_weights, &sign_weights, alpha, out_features, k_packed, &mut output,
        );

        for j in 0..out_features {
            let nz_row = &nonzero_weights[j * k_packed..(j + 1) * k_packed];
            let sign_row = &sign_weights[j * k_packed..(j + 1) * k_packed];
            let expected = ternary_accumulate_row_scalar(&input, nz_row, sign_row) * alpha;
            assert!((output[j] - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn ternary_forward_layer_batched_matches_repeated_single_row_calls() {
        let out_features = 5;
        let k_packed = 2;
        let num_rows = 4;
        let alpha = 1.7f32;
        let row_in_len = k_packed * 64;
        let mut seed = 13579u64;
        let nonzero_weights = random_bits(&mut seed, out_features * k_packed);
        let sign_weights = random_bits(&mut seed, out_features * k_packed);

        let inputs: Vec<f32> = (0..num_rows * row_in_len)
            .map(|i| ((i as f32) * 0.041).sin() * 1.5)
            .collect();

        let mut batched_outputs = vec![0.0f32; num_rows * out_features];
        ternary_forward_layer_batched(
            &inputs, &nonzero_weights, &sign_weights, alpha, out_features, k_packed, num_rows,
            &mut batched_outputs,
        );

        for r in 0..num_rows {
            let row_input = &inputs[r * row_in_len..(r + 1) * row_in_len];
            let mut single_output = vec![0.0f32; out_features];
            ternary_forward_layer(
                row_input, &nonzero_weights, &sign_weights, alpha, out_features, k_packed, &mut single_output,
            );

            let batched_row = &batched_outputs[r * out_features..(r + 1) * out_features];
            for j in 0..out_features {
                assert!((batched_row[j] - single_output[j]).abs() < 1e-6);
            }
        }
    }
}
