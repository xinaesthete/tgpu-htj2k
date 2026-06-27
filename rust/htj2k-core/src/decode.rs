//! Full single-component decode: reassemble code-blocks into subbands and run
//! the inverse 5/3 (reversible) DWT to reconstruct the image. Lossless path.
//!
//! The inverse 5/3 DWT is a faithful port of OpenJPH's `gen_rev_horz_syn`
//! (`ojph_transform.cpp`): deinterleaved low/high buffers, per-step **clamp**
//! (replicate) boundary extension, an origin-parity `even` flag, and the
//! aug/oth buffer swap. Pass order is horizontal-then-vertical to match
//! `resolution::pull_line`. Bit-exact vs OpenJPH for high-detail content across
//! odd / non-power-of-two / multi-level sizes (see `test/decode.test.ts`).
//!
//! Known limitation: *over-decomposed* images — more decomposition levels than
//! the image naturally supports, which produces two consecutive 1×1
//! resolutions — are not yet bit-exact. The DWT itself is correct there (the
//! 1×1 passthrough is unit-tested); the gap is upstream code-block coverage at
//! those degenerate resolutions. Realistic encoders never over-decompose.

use crate::block_decoder;
use crate::geometry::{compute_component_layout, Orientation};

/// One subband's coefficients (row-major, `w * h`).
struct Band {
    w: usize,
    h: usize,
    data: Vec<i32>,
}

/// Clamp (replicate) read of `buf[idx]` into `[0, len)` — OpenJPH extends the
/// deinterleaved high/low buffers by replicating the edge sample (`oth[-1] =
/// oth[0]`, `oth[len] = oth[len-1]`), *not* by whole-sample mirroring.
#[inline]
fn clamp_get(buf: &[i32], idx: isize, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    if idx < 0 {
        buf[0]
    } else if idx as usize >= len {
        buf[len - 1]
    } else {
        buf[idx as usize]
    }
}

/// Inverse 1D 5/3 (reversible) synthesis — a faithful port of OpenJPH's
/// `gen_rev_horz_syn32` (`ojph_transform.cpp`). Used for both the horizontal
/// (row) and vertical (column) passes, since the 1D math is identical.
///
/// `low`/`high` are the deinterleaved low-pass / high-pass subband samples.
/// `even` is the parity of this resolution's coordinate origin on the canvas
/// (`(org & 1) == 0`): when `true` the first reconstructed sample (`out[0]`) is
/// a low-pass sample, otherwise it is a high-pass sample. This parity — skipped
/// in the earlier mirror-based version — is what makes the boundary extension
/// match OpenJPH for odd dimensions and non-even origins.
///
/// The two 5/3 synthesis lifting steps (T.801 `init_rev53`) are applied on the
/// deinterleaved buffers with per-step clamp extension and an aug/oth buffer
/// swap, exactly as OpenJPH does, then the result is re-interleaved into `out`.
/// Integer `>>` is an arithmetic (floor) shift, matching OpenJPH's signed `>>`.
fn idwt_1d_53(low: &[i32], high: &[i32], out: &mut [i32], even: bool) {
    let width = low.len() + high.len();
    if width == 0 {
        return;
    }
    if width == 1 {
        // OpenJPH: lone sample — low passes through, lone high is halved.
        out[0] = if even { low[0] } else { high[0] >> 1 };
        return;
    }

    // `aug` starts as the low-pass buffer, `oth` as the high-pass buffer.
    let mut aug = low.to_vec();
    let mut oth = high.to_vec();
    let mut aug_width = (width + if even { 1 } else { 0 }) >> 1; // low-pass count
    let mut oth_width = (width + if even { 0 } else { 1 }) >> 1; // high-pass count
    let mut ev = even;

    // 5/3 reversible synthesis steps: (A=1, B=2, E=2) update, then
    // (A=-1, B=1, E=1) predict. `synthesis` direction per gen_rev_vert_step.
    for &(a, b, e) in &[(1i32, 2i32, 2u32), (-1i32, 1i32, 1u32)] {
        let off: isize = if ev { 0 } else { 1 };
        for i in 0..aug_width {
            let ii = i as isize + off;
            let s = clamp_get(&oth, ii - 1, oth_width) + clamp_get(&oth, ii, oth_width);
            if a == 1 {
                aug[i] -= (b + s) >> e; // update (synthesis subtracts)
            } else {
                aug[i] += s >> e; // 5/3 predict (a=-1, b=1, e=1)
            }
        }
        std::mem::swap(&mut aug, &mut oth);
        std::mem::swap(&mut aug_width, &mut oth_width);
        ev = !ev;
    }
    // Two swaps later, `aug` holds the final low samples, `oth` the final high.

    // Re-interleave: a leading lone high sample when the origin is odd, then
    // alternating low/high, then a trailing lone low sample.
    let (lo, hi) = (&aug, &oth);
    let mut li = 0usize;
    let mut hi_i = 0usize;
    let mut dp = 0usize;
    let mut w = width;
    if !even {
        out[dp] = hi[hi_i];
        dp += 1;
        hi_i += 1;
        w -= 1;
    }
    while w > 1 {
        out[dp] = lo[li];
        out[dp + 1] = hi[hi_i];
        dp += 2;
        li += 1;
        hi_i += 1;
        w -= 2;
    }
    if w == 1 {
        out[dp] = lo[li];
    }
}

/// Inverse one DWT level: combine `ll` (low-low) with detail bands `hl`, `lh`,
/// `hh` into a reconstructed band of size `out_w * out_h`.
///
/// Order matches OpenJPH's `resolution::pull_line`: **horizontal first, then
/// vertical**. Integer 5/3 lifting does not commute, so the pass order must be
/// the reference's to stay bit-exact across decomposition levels.
///
/// `even_x`/`even_y` are the parity of this resolution's canvas origin
/// (`(org & 1) == 0`), threaded into the 1D synthesis so boundary extension
/// matches OpenJPH (see `idwt_1d_53`).
fn idwt_level(
    ll: &Band,
    hl: &Band,
    lh: &Band,
    hh: &Band,
    out_w: usize,
    out_h: usize,
    even_x: bool,
    even_y: bool,
) -> Band {
    let rw0 = ll.w; // low horizontal width  (== lh.w)
    let rw1 = hl.w; // high horizontal width (== hh.w)
    let rh0 = ll.h; // low vertical height   (== hl.h)
    let rh1 = lh.h; // high vertical height  (== hh.h)
    debug_assert_eq!(rw0 + rw1, out_w);
    debug_assert_eq!(rh0 + rh1, out_h);

    // Horizontal inverse first:
    //   A = combine(LL, HL) per row → vertically-low, full-width rows  (rh0 × out_w)
    //   B = combine(LH, HH) per row → vertically-high, full-width rows (rh1 × out_w)
    let mut a = vec![0i32; out_w * rh0];
    let mut b = vec![0i32; out_w * rh1];
    let mut row_out = vec![0i32; out_w];

    for y in 0..rh0 {
        let lo = &ll.data[y * rw0..y * rw0 + rw0];
        let hi = &hl.data[y * rw1..y * rw1 + rw1];
        idwt_1d_53(lo, hi, &mut row_out[..out_w], even_x);
        a[y * out_w..y * out_w + out_w].copy_from_slice(&row_out[..out_w]);
    }
    for y in 0..rh1 {
        let lo = &lh.data[y * rw0..y * rw0 + rw0];
        let hi = &hh.data[y * rw1..y * rw1 + rw1];
        idwt_1d_53(lo, hi, &mut row_out[..out_w], even_x);
        b[y * out_w..y * out_w + out_w].copy_from_slice(&row_out[..out_w]);
    }

    // Vertical inverse: combine A (low) and B (high) per column → output.
    let mut out = vec![0i32; out_w * out_h];
    let mut col_low = vec![0i32; rh0.max(1)];
    let mut col_high = vec![0i32; rh1.max(1)];
    let mut col_out = vec![0i32; out_h];
    for x in 0..out_w {
        for y in 0..rh0 {
            col_low[y] = a[y * out_w + x];
        }
        for y in 0..rh1 {
            col_high[y] = b[y * out_w + x];
        }
        idwt_1d_53(&col_low[..rh0], &col_high[..rh1], &mut col_out[..out_h], even_y);
        for y in 0..out_h {
            out[y * out_w + x] = col_out[y];
        }
    }
    Band { w: out_w, h: out_h, data: out }
}

// ---------------------------------------------------------------------------
// Irreversible 9/7 (lossy) inverse DWT — float analogue of the 5/3 path above.
// Same deinterleaved/clamp/parity/swap structure (a port of OpenJPH
// `gen_irv_horz_syn`), but float, with a K pre-scale and four lifting steps.
// ---------------------------------------------------------------------------

/// 9/7 irreversible scaling constant K and the four lifting coefficients in
/// synthesis order (T.801 `init_irv97`).
const IRV97_K: f32 = 1.230_174_104_914_001;
const IRV97_STEPS: [f32; 4] = [
    0.443_506_852_043_971,
    0.882_911_075_530_934,
    -0.052_980_118_572_961,
    -1.586_134_342_059_924,
];

/// One subband's dequantized float coefficients (row-major, `w * h`).
struct FBand {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

#[inline]
fn clamp_get_f32(buf: &[f32], idx: isize, len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }
    if idx < 0 {
        buf[0]
    } else if idx as usize >= len {
        buf[len - 1]
    } else {
        buf[idx as usize]
    }
}

/// Inverse 1D 9/7 (irreversible) synthesis — a port of OpenJPH's
/// `gen_irv_horz_syn`. Mirrors `idwt_1d_53` (clamp extension, origin-parity
/// `even` flag, aug/oth swap, re-interleave) but operates on floats, applies
/// the K pre-scale (low `*= K`, high `*= 1/K`) before lifting, and runs four
/// `aug -= a * (l + r)` steps.
fn idwt_1d_97(low: &[f32], high: &[f32], out: &mut [f32], even: bool) {
    let width = low.len() + high.len();
    if width == 0 {
        return;
    }
    if width == 1 {
        out[0] = if even { low[0] } else { high[0] * 0.5 };
        return;
    }

    let mut aug = low.to_vec();
    let mut oth = high.to_vec();
    let mut aug_width = (width + if even { 1 } else { 0 }) >> 1;
    let mut oth_width = (width + if even { 0 } else { 1 }) >> 1;

    // K pre-scale on the original low/high buffers (before any swap).
    let k_inv = 1.0f32 / IRV97_K;
    for v in aug.iter_mut() {
        *v *= IRV97_K;
    }
    for v in oth.iter_mut() {
        *v *= k_inv;
    }

    let mut ev = even;
    for &a in &IRV97_STEPS {
        let off: isize = if ev { 0 } else { 1 };
        for i in 0..aug_width {
            let ii = i as isize + off;
            let s = clamp_get_f32(&oth, ii - 1, oth_width) + clamp_get_f32(&oth, ii, oth_width);
            aug[i] -= a * s;
        }
        std::mem::swap(&mut aug, &mut oth);
        std::mem::swap(&mut aug_width, &mut oth_width);
        ev = !ev;
    }

    let (lo, hi) = (&aug, &oth);
    let mut li = 0usize;
    let mut hi_i = 0usize;
    let mut dp = 0usize;
    let mut w = width;
    if !even {
        out[dp] = hi[hi_i];
        dp += 1;
        hi_i += 1;
        w -= 1;
    }
    while w > 1 {
        out[dp] = lo[li];
        out[dp + 1] = hi[hi_i];
        dp += 2;
        li += 1;
        hi_i += 1;
        w -= 2;
    }
    if w == 1 {
        out[dp] = lo[li];
    }
}

/// Inverse one 9/7 DWT level — float analogue of `idwt_level` (horizontal then
/// vertical, per-axis origin parity).
fn idwt_level_f32(
    ll: &FBand,
    hl: &FBand,
    lh: &FBand,
    hh: &FBand,
    out_w: usize,
    out_h: usize,
    even_x: bool,
    even_y: bool,
) -> FBand {
    let rw0 = ll.w;
    let rw1 = hl.w;
    let rh0 = ll.h;
    let rh1 = lh.h;
    debug_assert_eq!(rw0 + rw1, out_w);
    debug_assert_eq!(rh0 + rh1, out_h);

    let mut a = vec![0f32; out_w * rh0];
    let mut b = vec![0f32; out_w * rh1];
    let mut row_out = vec![0f32; out_w];

    for y in 0..rh0 {
        let lo = &ll.data[y * rw0..y * rw0 + rw0];
        let hi = &hl.data[y * rw1..y * rw1 + rw1];
        idwt_1d_97(lo, hi, &mut row_out[..out_w], even_x);
        a[y * out_w..y * out_w + out_w].copy_from_slice(&row_out[..out_w]);
    }
    for y in 0..rh1 {
        let lo = &lh.data[y * rw0..y * rw0 + rw0];
        let hi = &hh.data[y * rw1..y * rw1 + rw1];
        idwt_1d_97(lo, hi, &mut row_out[..out_w], even_x);
        b[y * out_w..y * out_w + out_w].copy_from_slice(&row_out[..out_w]);
    }

    let mut out = vec![0f32; out_w * out_h];
    let mut col_low = vec![0f32; rh0.max(1)];
    let mut col_high = vec![0f32; rh1.max(1)];
    let mut col_out = vec![0f32; out_h];
    for x in 0..out_w {
        for y in 0..rh0 {
            col_low[y] = a[y * out_w + x];
        }
        for y in 0..rh1 {
            col_high[y] = b[y * out_w + x];
        }
        idwt_1d_97(&col_low[..rh0], &col_high[..rh1], &mut col_out[..out_h], even_y);
        for y in 0..out_h {
            out[y * out_w + x] = col_out[y];
        }
    }
    FBand { w: out_w, h: out_h, data: out }
}

/// Reconstruct a single reversible component from its parsed code-blocks.
/// Returns the level-shifted (pixel-domain) samples, row-major `width * height`.
pub fn reconstruct_reversible(
    data: &[u8],
    width: u32,
    height: u32,
    num_decompositions: u32,
    code_block_width: u32,
    code_block_height: u32,
    guard_bits: u32,
    subband_exponents: &[u8],
    code_blocks: &[block_decoder_input::CbInput],
) -> Option<Vec<i32>> {
    let layout = compute_component_layout(
        0,
        0,
        width as i64,
        height as i64,
        num_decompositions,
        code_block_width,
        code_block_height,
    );

    // Allocate a Band per (resolution, orientation) and fill it from code-blocks.
    // bands[res] = (LL?, HL, LH, HH) — res 0 holds only LL.
    let n_res = layout.resolutions.len();
    let mut ll0: Option<Band> = None;
    // detail[res-1] = [HL, LH, HH]
    let mut detail: Vec<[Option<Band>; 3]> = (0..n_res).map(|_| [None, None, None]).collect();

    for res in &layout.resolutions {
        for sb in &res.subbands {
            let band = Band {
                w: sb.width() as usize,
                h: sb.height() as usize,
                data: vec![0i32; (sb.width() * sb.height()) as usize],
            };
            match (res.index, sb.orientation) {
                (0, Orientation::LL) => ll0 = Some(band),
                (_, Orientation::HL) => detail[res.index as usize][0] = Some(band),
                (_, Orientation::LH) => detail[res.index as usize][1] = Some(band),
                (_, Orientation::HH) => detail[res.index as usize][2] = Some(band),
                _ => {}
            }
        }
    }

    // Fill bands from decoded code-blocks.
    let cbw = code_block_width as usize;
    let cbh = code_block_height as usize;
    for cb in code_blocks {
        // Locate the target band and its grid.
        let band: &mut Band = match (cb.resolution, cb.orientation) {
            (0, Orientation::LL) => ll0.as_mut()?,
            (r, Orientation::HL) => detail[r as usize][0].as_mut()?,
            (r, Orientation::LH) => detail[r as usize][1].as_mut()?,
            (r, Orientation::HH) => detail[r as usize][2].as_mut()?,
            _ => return None,
        };
        let x0 = cb.x as usize * cbw;
        let y0 = cb.y as usize * cbh;
        let bw = cbw.min(band.w - x0);
        let bh = cbh.min(band.h - y0);

        // K_max for this subband.
        let sb_idx = if cb.resolution == 0 {
            0usize
        } else {
            ((cb.resolution - 1) * 3) as usize
                + match cb.orientation {
                    Orientation::HL => 0,
                    Orientation::LH => 1,
                    Orientation::HH => 2,
                    Orientation::LL => 0,
                }
                + 1
        };
        let exp = *subband_exponents.get(sb_idx).unwrap_or(&0) as u32;
        let k_max = exp.saturating_sub(1) + guard_bits;

        let coded = data.get(cb.offset as usize..(cb.offset + cb.length_cleanup) as usize)?;
        let decoded = block_decoder::decode_cleanup(coded, cb.missing_msbs, cb.length_cleanup, bw as u32, bh as u32)?;
        let coeffs = block_decoder::reversible_to_i32(&decoded, k_max);
        for y in 0..bh {
            for x in 0..bw {
                band.data[(y0 + y) * band.w + (x0 + x)] = coeffs[y * bw + x];
            }
        }
    }

    // Inverse DWT from the coarsest level up.
    let mut cur = ll0?;
    for r in 1..n_res {
        let res = &layout.resolutions[r];
        let out_w = res.width() as usize;
        let out_h = res.height() as usize;
        let hl = detail[r][0].take()?;
        let lh = detail[r][1].take()?;
        let hh = detail[r][2].take()?;
        // Boundary parity: whether this resolution's canvas origin is even
        // (OpenJPH `horz_even = (org.x & 1) == 0`, `vert_even = (org.y & 1) == 0`).
        let even_x = (res.x0 & 1) == 0;
        let even_y = (res.y0 & 1) == 0;
        cur = idwt_level(&cur, &hl, &lh, &hh, out_w, out_h, even_x, even_y);
    }
    Some(cur.data)
}

/// Quantization step size `delta` for an irreversible (9/7) subband, including
/// the `2^-(31 - K_max)` fixed-point scale folded in — a port of OpenJPH
/// `get_irrev_delta` (`ojph_params.cpp`) plus `ojph_subband.cpp`'s
/// `d /= 2^(31 - K_max)`. `orient` is 0=LL, 1=HL, 2=LH, 3=HH.
fn irrev_delta(exp: u32, mantissa: u16, orient: usize, k_max: u32) -> f32 {
    // arr = sub-band energy gain by orientation.
    const ARR: [f32; 4] = [1.0, 2.0, 2.0, 4.0];
    let delta_b = ((mantissa as u32 | 0x800) as f32 * ARR[orient]) / 2048.0
        / ((1u64 << exp) as f32);
    delta_b / ((1u64 << (31 - k_max)) as f32)
}

/// Reconstruct a single irreversible (9/7, lossy) component from its parsed
/// code-blocks. Returns the inverse-DWT float samples (normalized, ~[-0.5,
/// 0.5)); the caller applies the level shift + rounding to pixel values.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_irreversible(
    data: &[u8],
    width: u32,
    height: u32,
    num_decompositions: u32,
    code_block_width: u32,
    code_block_height: u32,
    guard_bits: u32,
    subband_exponents: &[u8],
    subband_mantissas: &[u16],
    code_blocks: &[block_decoder_input::CbInput],
) -> Option<Vec<f32>> {
    let layout = compute_component_layout(
        0,
        0,
        width as i64,
        height as i64,
        num_decompositions,
        code_block_width,
        code_block_height,
    );

    let n_res = layout.resolutions.len();
    let mut ll0: Option<FBand> = None;
    let mut detail: Vec<[Option<FBand>; 3]> = (0..n_res).map(|_| [None, None, None]).collect();

    for res in &layout.resolutions {
        for sb in &res.subbands {
            let band = FBand {
                w: sb.width() as usize,
                h: sb.height() as usize,
                data: vec![0f32; (sb.width() * sb.height()) as usize],
            };
            match (res.index, sb.orientation) {
                (0, Orientation::LL) => ll0 = Some(band),
                (_, Orientation::HL) => detail[res.index as usize][0] = Some(band),
                (_, Orientation::LH) => detail[res.index as usize][1] = Some(band),
                (_, Orientation::HH) => detail[res.index as usize][2] = Some(band),
                _ => {}
            }
        }
    }

    let cbw = code_block_width as usize;
    let cbh = code_block_height as usize;
    for cb in code_blocks {
        let band: &mut FBand = match (cb.resolution, cb.orientation) {
            (0, Orientation::LL) => ll0.as_mut()?,
            (r, Orientation::HL) => detail[r as usize][0].as_mut()?,
            (r, Orientation::LH) => detail[r as usize][1].as_mut()?,
            (r, Orientation::HH) => detail[r as usize][2].as_mut()?,
            _ => return None,
        };
        let x0 = cb.x as usize * cbw;
        let y0 = cb.y as usize * cbh;
        let bw = cbw.min(band.w - x0);
        let bh = cbh.min(band.h - y0);

        // K_max and quant index (identical mapping to the reversible path).
        let sb_idx = if cb.resolution == 0 {
            0usize
        } else {
            ((cb.resolution - 1) * 3) as usize
                + match cb.orientation {
                    Orientation::HL => 0,
                    Orientation::LH => 1,
                    Orientation::HH => 2,
                    Orientation::LL => 0,
                }
                + 1
        };
        let exp = *subband_exponents.get(sb_idx).unwrap_or(&0) as u32;
        let mantissa = *subband_mantissas.get(sb_idx).unwrap_or(&0);
        let k_max = exp.saturating_sub(1) + guard_bits;
        let orient = cb.orientation as usize;
        let delta = irrev_delta(exp, mantissa, orient, k_max);

        let coded = data.get(cb.offset as usize..(cb.offset + cb.length_cleanup) as usize)?;
        let decoded = block_decoder::decode_cleanup(coded, cb.missing_msbs, cb.length_cleanup, bw as u32, bh as u32)?;
        let coeffs = block_decoder::irreversible_to_f32(&decoded, delta);
        for y in 0..bh {
            for x in 0..bw {
                band.data[(y0 + y) * band.w + (x0 + x)] = coeffs[y * bw + x];
            }
        }
    }

    let mut cur = ll0?;
    for r in 1..n_res {
        let res = &layout.resolutions[r];
        let out_w = res.width() as usize;
        let out_h = res.height() as usize;
        let hl = detail[r][0].take()?;
        let lh = detail[r][1].take()?;
        let hh = detail[r][2].take()?;
        let even_x = (res.x0 & 1) == 0;
        let even_y = (res.y0 & 1) == 0;
        cur = idwt_level_f32(&cur, &hl, &lh, &hh, out_w, out_h, even_x, even_y);
    }
    Some(cur.data)
}

#[cfg(test)]
mod tests {
    use super::{clamp_get, idwt_1d_53};

    /// Forward 5/3 analysis — the exact inverse of `idwt_1d_53` for a given
    /// `even` parity. Deinterleaves `x` into the (already-lifted) low/high
    /// samples present at even/odd output positions, then undoes the predict
    /// and update lifting steps with the same clamp extension.
    fn fwd_1d_53(x: &[i32], even: bool) -> (Vec<i32>, Vec<i32>) {
        let width = x.len();
        let nl = (width + if even { 1 } else { 0 }) >> 1;
        let nh = (width + if even { 0 } else { 1 }) >> 1;
        if width == 1 {
            return if even {
                (vec![x[0]], vec![])
            } else {
                (vec![], vec![x[0] << 1])
            };
        }
        // Deinterleave: low/high are the lifted samples as they appear in x.
        let mut low = vec![0i32; nl];
        let mut high = vec![0i32; nh];
        {
            let mut li = 0;
            let mut hi = 0;
            let mut dp = 0;
            if !even {
                high[hi] = x[dp];
                hi += 1;
                dp += 1;
            }
            while dp + 1 < width {
                low[li] = x[dp];
                high[hi] = x[dp + 1];
                li += 1;
                hi += 1;
                dp += 2;
            }
            if dp < width {
                low[li] = x[dp];
            }
        }
        // Undo predict (step1: synthesis added `s >> 1`), then undo update
        // (step0: synthesis subtracted `(2 + s) >> 2`). Reverse order, with the
        // same aug/oth roles: predict acts on `high` reading `low`; update acts
        // on `low` reading `high`.
        let off_predict: isize = if even { 1 } else { 0 }; // ev after one swap
        for i in 0..nh {
            let ii = i as isize + off_predict;
            let s = clamp_get(&low, ii - 1, nl) + clamp_get(&low, ii, nl);
            high[i] -= s >> 1;
        }
        let off_update: isize = if even { 0 } else { 1 };
        for i in 0..nl {
            let ii = i as isize + off_update;
            let s = clamp_get(&high, ii - 1, nh) + clamp_get(&high, ii, nh);
            low[i] += (2 + s) >> 2;
        }
        (low, high)
    }

    #[test]
    fn idwt_level_1x1_passthrough() {
        use super::{idwt_level, Band};
        let ll = Band { w: 1, h: 1, data: vec![1234] };
        let empty_col = Band { w: 0, h: 1, data: vec![] }; // HL: 0 wide, 1 tall
        let empty_row = Band { w: 1, h: 0, data: vec![] }; // LH: 1 wide, 0 tall
        let empty_hh = Band { w: 0, h: 0, data: vec![] };
        let out = idwt_level(&ll, &empty_col, &empty_row, &empty_hh, 1, 1, true, true);
        assert_eq!(out.data, vec![1234]);
    }

    #[test]
    fn idwt_53_round_trips_for_all_lengths_and_parities() {
        for &even in &[true, false] {
            for n in 1..40usize {
                let x: Vec<i32> = (0..n as i32).map(|i| (i * 37 + 5) % 101 - 50).collect();
                let (low, high) = fwd_1d_53(&x, even);
                let mut out = vec![0i32; n];
                idwt_1d_53(&low, &high, &mut out, even);
                assert_eq!(out, x, "round-trip failed for length {} even={}", n, even);
            }
        }
    }
}

/// Minimal per-code-block inputs needed for reconstruction.
pub mod block_decoder_input {
    use crate::geometry::Orientation;
    #[derive(Clone, Copy)]
    pub struct CbInput {
        pub resolution: u32,
        pub orientation: Orientation,
        pub x: u32,
        pub y: u32,
        pub offset: u32,
        pub length_cleanup: u32,
        pub missing_msbs: u32,
    }
}
