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
}
