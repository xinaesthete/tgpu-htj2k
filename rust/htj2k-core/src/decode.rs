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

/// Decode all code-blocks into per-(resolution, orientation) `Band`s for the
/// reversible path: the LL base band plus `detail[r] = [HL, LH, HH]`. This is
/// the entropy-decode + reassembly stage shared by the CPU inverse DWT and the
/// GPU DWT input packer (`dwt_input_53`).
fn fill_bands_53(
    data: &[u8],
    width: u32,
    height: u32,
    num_decompositions: u32,
    code_block_width: u32,
    code_block_height: u32,
    guard_bits: u32,
    subband_exponents: &[u8],
    code_blocks: &[block_decoder_input::CbInput],
) -> Option<(crate::geometry::ComponentLayout, Band, Vec<[Option<Band>; 3]>)> {
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

    Some((layout, ll0?, detail))
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
    let (layout, ll0, mut detail) = fill_bands_53(
        data, width, height, num_decompositions, code_block_width, code_block_height,
        guard_bits, subband_exponents, code_blocks,
    )?;

    // Inverse DWT from the coarsest level up.
    let mut cur = ll0;
    for r in 1..layout.resolutions.len() {
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

/// Per-level descriptor record length (u32 words) in the GPU DWT input header.
pub const DWT_LEVEL_REC: usize = 12;

/// Pack the reversible subband coefficients + geometry into a flat layout the
/// GPU inverse DWT consumes, so the GPU does only the DWT (and the caller the
/// level shift). Returns `(descriptor, coeffs)`:
///
/// `descriptor` (u32): `[kernel=0, n_levels, ll0_w, ll0_h, ll0_off]` followed
/// by `n_levels` records of [`DWT_LEVEL_REC`] words for r=1..=N:
/// `[rw0, rh0, rw1, rh1, out_w, out_h, even_x, even_y, hl_off, lh_off, hh_off, _pad]`.
/// All `*_off` are element offsets into `coeffs` (row-major band data).
pub fn dwt_input_53(
    data: &[u8],
    width: u32,
    height: u32,
    num_decompositions: u32,
    code_block_width: u32,
    code_block_height: u32,
    guard_bits: u32,
    subband_exponents: &[u8],
    code_blocks: &[block_decoder_input::CbInput],
) -> Option<(Vec<u32>, Vec<i32>)> {
    let (layout, ll0, mut detail) = fill_bands_53(
        data, width, height, num_decompositions, code_block_width, code_block_height,
        guard_bits, subband_exponents, code_blocks,
    )?;

    let n_res = layout.resolutions.len();
    let n_levels = (n_res - 1) as u32;
    let mut coeffs: Vec<i32> = Vec::new();
    let mut header: Vec<u32> = vec![0, n_levels, ll0.w as u32, ll0.h as u32, 0];

    // ll0 at offset 0.
    let ll0_off = coeffs.len() as u32;
    header[4] = ll0_off;
    coeffs.extend_from_slice(&ll0.data);

    for r in 1..n_res {
        let res = &layout.resolutions[r];
        let prev = &layout.resolutions[r - 1];
        let hl = detail[r][0].take()?;
        let lh = detail[r][1].take()?;
        let hh = detail[r][2].take()?;
        let hl_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hl.data);
        let lh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&lh.data);
        let hh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hh.data);
        header.extend_from_slice(&[
            prev.width(),
            prev.height(),
            hl.w as u32,
            lh.h as u32,
            res.width(),
            res.height(),
            ((res.x0 & 1) == 0) as u32,
            ((res.y0 & 1) == 0) as u32,
            hl_off,
            lh_off,
            hh_off,
            0,
        ]);
    }
    Some((header, coeffs))
}

/// Run *only* the inverse 5/3 DWT on a packed `(descriptor, coeffs)` pair
/// (as produced by `dwt_input_53`). This is the CPU counterpart of the GPU
/// `idwt53Gpu` over identical input — used to benchmark the DWT stage in
/// isolation from entropy decode.
pub fn idwt53_from_packed(descriptor: &[u32], coeffs: &[i32]) -> Option<Vec<i32>> {
    let n_levels = *descriptor.get(1)? as usize;
    let ll0_w = *descriptor.get(2)? as usize;
    let ll0_h = *descriptor.get(3)? as usize;
    let ll0_off = *descriptor.get(4)? as usize;
    let mut cur = Band {
        w: ll0_w,
        h: ll0_h,
        data: coeffs.get(ll0_off..ll0_off + ll0_w * ll0_h)?.to_vec(),
    };
    for lvl in 0..n_levels {
        let o = 5 + lvl * DWT_LEVEL_REC;
        let rec = descriptor.get(o..o + DWT_LEVEL_REC)?;
        let (rw0, rh0, rw1, rh1) = (rec[0] as usize, rec[1] as usize, rec[2] as usize, rec[3] as usize);
        let (out_w, out_h) = (rec[4] as usize, rec[5] as usize);
        let (even_x, even_y) = (rec[6] == 1, rec[7] == 1);
        let (hl_off, lh_off, hh_off) = (rec[8] as usize, rec[9] as usize, rec[10] as usize);
        let hl = Band { w: rw1, h: rh0, data: coeffs.get(hl_off..hl_off + rw1 * rh0)?.to_vec() };
        let lh = Band { w: rw0, h: rh1, data: coeffs.get(lh_off..lh_off + rw0 * rh1)?.to_vec() };
        let hh = Band { w: rw1, h: rh1, data: coeffs.get(hh_off..hh_off + rw1 * rh1)?.to_vec() };
        cur = idwt_level(&cur, &hl, &lh, &hh, out_w, out_h, even_x, even_y);
    }
    Some(cur.data)
}

// ---------------------------------------------------------------------------
// Forward (analysis) 5/3 DWT — the encode-direction transform, and a reusable
// primitive in its own right. Exact inverse of the synthesis above: it
// decomposes a spatial image into LL + per-level (HL, LH, HH) subbands in the
// same packed layout the inverse consumes, so forward → inverse is identity and
// forward output is bit-exact vs OpenJPH's analysis.
// ---------------------------------------------------------------------------

/// Forward 1D 5/3 analysis — exact inverse of `idwt_1d_53`. Splits a spatial
/// line `x` into deinterleaved low/high subband samples (clamp extension,
/// origin-parity `even`). Predict then update (reverse of the synthesis order).
fn fwd_1d_53(x: &[i32], even: bool) -> (Vec<i32>, Vec<i32>) {
    let width = x.len();
    let nl = (width + if even { 1 } else { 0 }) >> 1;
    let nh = (width + if even { 0 } else { 1 }) >> 1;
    let mut low = vec![0i32; nl];
    let mut high = vec![0i32; nh];
    if width == 0 {
        return (low, high);
    }
    if width == 1 {
        if even {
            low[0] = x[0];
        } else {
            high[0] = x[0] << 1;
        }
        return (low, high);
    }
    // Deinterleave: low samples sit at the parity-`even` positions.
    let low_phase = if even { 0usize } else { 1 };
    for p in 0..width {
        let idx = p >> 1;
        if (p & 1) == low_phase {
            low[idx] = x[p];
        } else {
            high[idx] = x[p];
        }
    }
    // Predict (analysis): high[i] -= (low[i+off-1] + low[i+off]) >> 1.
    let offp: isize = if even { 1 } else { 0 };
    for i in 0..nh {
        let ii = i as isize + offp;
        let s = clamp_get(&low, ii - 1, nl) + clamp_get(&low, ii, nl);
        high[i] -= s >> 1;
    }
    // Update (analysis): low[i] += (2 + high[i+off-1] + high[i+off]) >> 2.
    let offu: isize = if even { 0 } else { 1 };
    for i in 0..nl {
        let ii = i as isize + offu;
        let s = clamp_get(&high, ii - 1, nh) + clamp_get(&high, ii, nh);
        low[i] += (2 + s) >> 2;
    }
    (low, high)
}

/// Forward one 5/3 level: split an image band into (LL, HL, LH, HH). Vertical
/// analysis first, then horizontal — the reverse of `idwt_level`'s order.
#[allow(clippy::too_many_arguments)]
fn fwd_level_53(
    input: &Band,
    rw0: usize,
    rh0: usize,
    rw1: usize,
    rh1: usize,
    even_x: bool,
    even_y: bool,
) -> (Band, Band, Band, Band) {
    let w = input.w;
    let h = input.h;
    debug_assert_eq!(rw0 + rw1, w);
    debug_assert_eq!(rh0 + rh1, h);

    // Vertical analysis: each column → low (rh0 rows, A) + high (rh1 rows, B).
    let mut a = vec![0i32; w * rh0];
    let mut b = vec![0i32; w * rh1];
    let mut col = vec![0i32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = input.data[y * w + x];
        }
        let (lo, hi) = fwd_1d_53(&col, even_y);
        for y in 0..rh0 {
            a[y * w + x] = lo[y];
        }
        for y in 0..rh1 {
            b[y * w + x] = hi[y];
        }
    }

    // Horizontal analysis: A rows → LL + HL; B rows → LH + HH.
    let mut ll = vec![0i32; rw0 * rh0];
    let mut hl = vec![0i32; rw1 * rh0];
    let mut lh = vec![0i32; rw0 * rh1];
    let mut hh = vec![0i32; rw1 * rh1];
    for y in 0..rh0 {
        let (lo, hi) = fwd_1d_53(&a[y * w..y * w + w], even_x);
        ll[y * rw0..y * rw0 + rw0].copy_from_slice(&lo[..rw0]);
        hl[y * rw1..y * rw1 + rw1].copy_from_slice(&hi[..rw1]);
    }
    for y in 0..rh1 {
        let (lo, hi) = fwd_1d_53(&b[y * w..y * w + w], even_x);
        lh[y * rw0..y * rw0 + rw0].copy_from_slice(&lo[..rw0]);
        hh[y * rw1..y * rw1 + rw1].copy_from_slice(&hi[..rw1]);
    }
    (
        Band { w: rw0, h: rh0, data: ll },
        Band { w: rw1, h: rh0, data: hl },
        Band { w: rw0, h: rh1, data: lh },
        Band { w: rw1, h: rh1, data: hh },
    )
}

/// Forward 5/3 DWT of a full (level-shifted) image → packed `(descriptor,
/// coeffs)` in the same layout `dwt_input_53` / `idwt53_from_packed` consume, so
/// `dwt_forward_53` then `idwt53_from_packed` is the identity. Decomposes
/// finest→coarsest; the running LL feeds the next (coarser) level.
pub fn dwt_forward_53(
    samples: &[i32],
    width: u32,
    height: u32,
    num_decompositions: u32,
    code_block_width: u32,
    code_block_height: u32,
) -> Option<(Vec<u32>, Vec<i32>)> {
    let layout = compute_component_layout(
        0, 0, width as i64, height as i64, num_decompositions, code_block_width, code_block_height,
    );
    let n_res = layout.resolutions.len();
    let top = layout.resolutions.last()?;
    let (w, h) = (top.width() as usize, top.height() as usize);
    if samples.len() != w * h {
        return None;
    }

    let mut cur = Band { w, h, data: samples.to_vec() };
    let mut detail: Vec<Option<(Band, Band, Band)>> = (0..n_res).map(|_| None).collect();
    for r in (1..n_res).rev() {
        let res = &layout.resolutions[r];
        let prev = &layout.resolutions[r - 1];
        let rw0 = prev.width() as usize;
        let rh0 = prev.height() as usize;
        let rw1 = res.width() as usize - rw0;
        let rh1 = res.height() as usize - rh0;
        let even_x = (res.x0 & 1) == 0;
        let even_y = (res.y0 & 1) == 0;
        let (ll, hl, lh, hh) = fwd_level_53(&cur, rw0, rh0, rw1, rh1, even_x, even_y);
        detail[r] = Some((hl, lh, hh));
        cur = ll;
    }

    let n_levels = (n_res - 1) as u32;
    let mut coeffs: Vec<i32> = Vec::new();
    let mut header: Vec<u32> = vec![0, n_levels, cur.w as u32, cur.h as u32, 0];
    coeffs.extend_from_slice(&cur.data); // ll0 at offset 0
    for r in 1..n_res {
        let res = &layout.resolutions[r];
        let prev = &layout.resolutions[r - 1];
        let (hl, lh, hh) = detail[r].take()?;
        let hl_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hl.data);
        let lh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&lh.data);
        let hh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hh.data);
        header.extend_from_slice(&[
            prev.width(), prev.height(), hl.w as u32, lh.h as u32, res.width(), res.height(),
            ((res.x0 & 1) == 0) as u32, ((res.y0 & 1) == 0) as u32, hl_off, lh_off, hh_off, 0,
        ]);
    }
    Some((header, coeffs))
}

// ---------------------------------------------------------------------------
// Forward (analysis) 9/7 DWT — float, the irreversible analysis direction, and
// the float half of the reusable transform pair. Exact inverse of `idwt_1d_97`.
// ---------------------------------------------------------------------------

/// Forward 1D 9/7 analysis — exact inverse of `idwt_1d_97`: deinterleave, run
/// the four lifting steps in reverse with `aug += a*(l+r)`, then undo the K
/// pre-scale (low /= K, high *= K).
fn fwd_1d_97(x: &[f32], even: bool) -> (Vec<f32>, Vec<f32>) {
    let width = x.len();
    let nl = (width + if even { 1 } else { 0 }) >> 1;
    let nh = (width + if even { 0 } else { 1 }) >> 1;
    let mut low = vec![0f32; nl];
    let mut high = vec![0f32; nh];
    if width == 0 {
        return (low, high);
    }
    if width == 1 {
        if even {
            low[0] = x[0];
        } else {
            high[0] = x[0] * 2.0;
        }
        return (low, high);
    }
    let low_phase = if even { 0usize } else { 1 };
    for p in 0..width {
        let idx = p >> 1;
        if (p & 1) == low_phase {
            low[idx] = x[p];
        } else {
            high[idx] = x[p];
        }
    }
    // Undo the synthesis lifting: reverse step order, opposite sign.
    for jj in 0..4 {
        let j = 3 - jj;
        let a = IRV97_STEPS[j];
        let aug_low = (j & 1) == 0;
        let ev = if aug_low { even } else { !even };
        let off: isize = if ev { 0 } else { 1 };
        if aug_low {
            for i in 0..nl {
                let ii = i as isize + off;
                let s = clamp_get_f32(&high, ii - 1, nh) + clamp_get_f32(&high, ii, nh);
                low[i] += a * s;
            }
        } else {
            for i in 0..nh {
                let ii = i as isize + off;
                let s = clamp_get_f32(&low, ii - 1, nl) + clamp_get_f32(&low, ii, nl);
                high[i] += a * s;
            }
        }
    }
    let k_inv = 1.0f32 / IRV97_K;
    for v in low.iter_mut() {
        *v *= k_inv;
    }
    for v in high.iter_mut() {
        *v *= IRV97_K;
    }
    (low, high)
}

/// Forward one 9/7 level (float): vertical analysis then horizontal split.
#[allow(clippy::too_many_arguments)]
fn fwd_level_97(
    input: &FBand,
    rw0: usize,
    rh0: usize,
    rw1: usize,
    rh1: usize,
    even_x: bool,
    even_y: bool,
) -> (FBand, FBand, FBand, FBand) {
    let w = input.w;
    let h = input.h;
    let mut a = vec![0f32; w * rh0];
    let mut b = vec![0f32; w * rh1];
    let mut col = vec![0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = input.data[y * w + x];
        }
        let (lo, hi) = fwd_1d_97(&col, even_y);
        for y in 0..rh0 {
            a[y * w + x] = lo[y];
        }
        for y in 0..rh1 {
            b[y * w + x] = hi[y];
        }
    }
    let mut ll = vec![0f32; rw0 * rh0];
    let mut hl = vec![0f32; rw1 * rh0];
    let mut lh = vec![0f32; rw0 * rh1];
    let mut hh = vec![0f32; rw1 * rh1];
    for y in 0..rh0 {
        let (lo, hi) = fwd_1d_97(&a[y * w..y * w + w], even_x);
        ll[y * rw0..y * rw0 + rw0].copy_from_slice(&lo[..rw0]);
        hl[y * rw1..y * rw1 + rw1].copy_from_slice(&hi[..rw1]);
    }
    for y in 0..rh1 {
        let (lo, hi) = fwd_1d_97(&b[y * w..y * w + w], even_x);
        lh[y * rw0..y * rw0 + rw0].copy_from_slice(&lo[..rw0]);
        hh[y * rw1..y * rw1 + rw1].copy_from_slice(&hi[..rw1]);
    }
    (
        FBand { w: rw0, h: rh0, data: ll },
        FBand { w: rw1, h: rh0, data: hl },
        FBand { w: rw0, h: rh1, data: lh },
        FBand { w: rw1, h: rh1, data: hh },
    )
}

/// Forward 9/7 DWT of a float image → packed `(descriptor, coeffs)` matching the
/// inverse path (`idwt97_from_packed`). `kernel` field of the descriptor is 1.
pub fn dwt_forward_97(
    samples: &[f32],
    width: u32,
    height: u32,
    num_decompositions: u32,
    code_block_width: u32,
    code_block_height: u32,
) -> Option<(Vec<u32>, Vec<f32>)> {
    let layout = compute_component_layout(
        0, 0, width as i64, height as i64, num_decompositions, code_block_width, code_block_height,
    );
    let n_res = layout.resolutions.len();
    let top = layout.resolutions.last()?;
    let (w, h) = (top.width() as usize, top.height() as usize);
    if samples.len() != w * h {
        return None;
    }
    let mut cur = FBand { w, h, data: samples.to_vec() };
    let mut detail: Vec<Option<(FBand, FBand, FBand)>> = (0..n_res).map(|_| None).collect();
    for r in (1..n_res).rev() {
        let res = &layout.resolutions[r];
        let prev = &layout.resolutions[r - 1];
        let rw0 = prev.width() as usize;
        let rh0 = prev.height() as usize;
        let rw1 = res.width() as usize - rw0;
        let rh1 = res.height() as usize - rh0;
        let (ll, hl, lh, hh) = fwd_level_97(&cur, rw0, rh0, rw1, rh1, (res.x0 & 1) == 0, (res.y0 & 1) == 0);
        detail[r] = Some((hl, lh, hh));
        cur = ll;
    }
    let n_levels = (n_res - 1) as u32;
    let mut coeffs: Vec<f32> = Vec::new();
    let mut header: Vec<u32> = vec![1, n_levels, cur.w as u32, cur.h as u32, 0];
    coeffs.extend_from_slice(&cur.data);
    for r in 1..n_res {
        let res = &layout.resolutions[r];
        let prev = &layout.resolutions[r - 1];
        let (hl, lh, hh) = detail[r].take()?;
        let hl_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hl.data);
        let lh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&lh.data);
        let hh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hh.data);
        header.extend_from_slice(&[
            prev.width(), prev.height(), hl.w as u32, lh.h as u32, res.width(), res.height(),
            ((res.x0 & 1) == 0) as u32, ((res.y0 & 1) == 0) as u32, hl_off, lh_off, hh_off, 0,
        ]);
    }
    Some((header, coeffs))
}

/// Inverse 9/7 from a packed `(descriptor, coeffs)` pair (float) — counterpart
/// of `idwt53_from_packed`, for round-tripping the forward and as a reusable
/// primitive. Returns the reconstructed float samples.
pub fn idwt97_from_packed(descriptor: &[u32], coeffs: &[f32]) -> Option<Vec<f32>> {
    let n_levels = *descriptor.get(1)? as usize;
    let ll0_w = *descriptor.get(2)? as usize;
    let ll0_h = *descriptor.get(3)? as usize;
    let ll0_off = *descriptor.get(4)? as usize;
    let mut cur = FBand {
        w: ll0_w,
        h: ll0_h,
        data: coeffs.get(ll0_off..ll0_off + ll0_w * ll0_h)?.to_vec(),
    };
    for lvl in 0..n_levels {
        let o = 5 + lvl * DWT_LEVEL_REC;
        let rec = descriptor.get(o..o + DWT_LEVEL_REC)?;
        let (rw0, rh0, rw1, rh1) = (rec[0] as usize, rec[1] as usize, rec[2] as usize, rec[3] as usize);
        let (out_w, out_h) = (rec[4] as usize, rec[5] as usize);
        let (even_x, even_y) = (rec[6] == 1, rec[7] == 1);
        let (hl_off, lh_off, hh_off) = (rec[8] as usize, rec[9] as usize, rec[10] as usize);
        let hl = FBand { w: rw1, h: rh0, data: coeffs.get(hl_off..hl_off + rw1 * rh0)?.to_vec() };
        let lh = FBand { w: rw0, h: rh1, data: coeffs.get(lh_off..lh_off + rw0 * rh1)?.to_vec() };
        let hh = FBand { w: rw1, h: rh1, data: coeffs.get(hh_off..hh_off + rw1 * rh1)?.to_vec() };
        cur = idwt_level_f32(&cur, &hl, &lh, &hh, out_w, out_h, even_x, even_y);
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
#[allow(clippy::too_many_arguments)]
fn fill_bands_97(
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
) -> Option<(crate::geometry::ComponentLayout, FBand, Vec<[Option<FBand>; 3]>)> {
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

    Some((layout, ll0?, detail))
}

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
    let (layout, ll0, mut detail) = fill_bands_97(
        data, width, height, num_decompositions, code_block_width, code_block_height,
        guard_bits, subband_exponents, subband_mantissas, code_blocks,
    )?;

    let mut cur = ll0;
    for r in 1..layout.resolutions.len() {
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

/// Pack the irreversible (dequantized float) subband coefficients + geometry
/// for the GPU inverse 9/7 DWT — same descriptor layout as `dwt_input_53`
/// (kernel = 1), with `coeffs` as f32.
#[allow(clippy::too_many_arguments)]
pub fn dwt_input_97(
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
) -> Option<(Vec<u32>, Vec<f32>)> {
    let (layout, ll0, mut detail) = fill_bands_97(
        data, width, height, num_decompositions, code_block_width, code_block_height,
        guard_bits, subband_exponents, subband_mantissas, code_blocks,
    )?;

    let n_res = layout.resolutions.len();
    let n_levels = (n_res - 1) as u32;
    let mut coeffs: Vec<f32> = Vec::new();
    let mut header: Vec<u32> = vec![1, n_levels, ll0.w as u32, ll0.h as u32, 0];

    header[4] = coeffs.len() as u32;
    coeffs.extend_from_slice(&ll0.data);

    for r in 1..n_res {
        let res = &layout.resolutions[r];
        let prev = &layout.resolutions[r - 1];
        let hl = detail[r][0].take()?;
        let lh = detail[r][1].take()?;
        let hh = detail[r][2].take()?;
        let hl_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hl.data);
        let lh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&lh.data);
        let hh_off = coeffs.len() as u32;
        coeffs.extend_from_slice(&hh.data);
        header.extend_from_slice(&[
            prev.width(),
            prev.height(),
            hl.w as u32,
            lh.h as u32,
            res.width(),
            res.height(),
            ((res.x0 & 1) == 0) as u32,
            ((res.y0 & 1) == 0) as u32,
            hl_off,
            lh_off,
            hh_off,
            0,
        ]);
    }
    Some((header, coeffs))
}

#[cfg(test)]
mod tests {
    use super::{
        dwt_forward_53, dwt_forward_97, fwd_1d_53, idwt53_from_packed, idwt97_from_packed, idwt_1d_53,
    };

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

    #[test]
    fn dwt_forward_then_inverse_is_identity() {
        // Full 2D forward (analysis) → inverse (synthesis) round-trips exactly,
        // across odd / non-square sizes and multiple levels.
        for &(w, h) in &[(8usize, 8usize), (9, 9), (17, 23), (32, 32), (33, 48)] {
            let max_lv = (w.min(h) as f64).log2().floor() as u32;
            for lv in 1..=max_lv.min(4) {
                let img: Vec<i32> = (0..(w * h) as i32).map(|i| (i * 37 + 11) % 4096 - 2048).collect();
                let (desc, coeffs) = dwt_forward_53(&img, w as u32, h as u32, lv, 64, 64).unwrap();
                let back = idwt53_from_packed(&desc, &coeffs).unwrap();
                assert_eq!(back, img, "round-trip failed w={} h={} lv={}", w, h, lv);
            }
        }
    }

    #[test]
    fn dwt_forward_97_round_trips_within_tolerance() {
        // Float 9/7 forward → inverse recovers the input within f32 tolerance.
        for &(w, h) in &[(8usize, 8usize), (9, 9), (17, 23), (32, 32), (33, 48)] {
            let max_lv = (w.min(h) as f64).log2().floor() as u32;
            for lv in 1..=max_lv.min(4) {
                let img: Vec<f32> = (0..(w * h) as i32)
                    .map(|i| ((i * 37 + 11) % 4096 - 2048) as f32 / 4096.0)
                    .collect();
                let (desc, coeffs) = dwt_forward_97(&img, w as u32, h as u32, lv, 64, 64).unwrap();
                let back = idwt97_from_packed(&desc, &coeffs).unwrap();
                for i in 0..back.len() {
                    assert!(
                        (back[i] - img[i]).abs() < 1e-3,
                        "round-trip drift {} at {} (w={} h={} lv={})",
                        (back[i] - img[i]).abs(), i, w, h, lv
                    );
                }
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
