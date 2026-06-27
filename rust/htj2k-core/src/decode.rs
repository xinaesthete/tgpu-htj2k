//! Full single-component decode: reassemble code-blocks into subbands and run
//! the inverse 5/3 (reversible) DWT to reconstruct the image. Lossless path.
//!
//! The inverse 5/3 lifting matches OpenJPH (predict: `+= (l+r) >> 1`,
//! update: `-= (2 + l + r) >> 2`) with whole-sample symmetric extension.

use crate::block_decoder;
use crate::geometry::{compute_component_layout, Orientation};

/// One subband's coefficients (row-major, `w * h`).
struct Band {
    w: usize,
    h: usize,
    data: Vec<i32>,
}

/// Whole-sample symmetric mirror of index `i` into `[0, n)`.
fn mirror(mut i: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// Inverse 1D 5/3 synthesis (interleaved, whole-sample symmetric extension).
/// Bit-exact vs OpenJPH for power-of-two-friendly subband sizes; see the
/// known-limitation note in `decode_image` for arbitrary non-power-of-two
/// dimensions, whose boundary parity OpenJPH handles differently.
fn idwt_1d_53(low: &[i32], high: &[i32], out: &mut [i32]) {
    let nl = low.len();
    let nh = high.len();
    let n = nl + nh;
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = if nl == 1 { low[0] } else { high[0] };
        return;
    }
    for k in 0..n {
        out[k] = if k & 1 == 0 { low[k / 2] } else { high[k / 2] };
    }
    let ni = n as isize;
    let mut k = 0;
    while k < n {
        let left = out[mirror(k as isize - 1, ni)];
        let right = out[mirror(k as isize + 1, ni)];
        out[k] -= (left + right + 2).div_euclid(4);
        k += 2;
    }
    let mut k = 1;
    while k < n {
        let left = out[k - 1];
        let right = out[mirror(k as isize + 1, ni)];
        out[k] += (left + right).div_euclid(2);
        k += 2;
    }
}

/// Inverse one DWT level: combine `ll` (low-low) with detail bands `hl`, `lh`,
/// `hh` into a reconstructed band of size `out_w * out_h`.
fn idwt_level(ll: &Band, hl: &Band, lh: &Band, hh: &Band, out_w: usize, out_h: usize) -> Band {
    let rw0 = ll.w; // low horizontal width  (== lh.w)
    let rw1 = hl.w; // high horizontal width (== hh.w)
    let rh0 = ll.h; // low vertical height   (== hl.h)
    let rh1 = lh.h; // high vertical height  (== hh.h)
    debug_assert_eq!(rw0 + rw1, out_w);
    debug_assert_eq!(rh0 + rh1, out_h);

    // Vertical inverse: A = combine(LL, LH) per column (width rw0, height out_h);
    //                   B = combine(HL, HH) per column (width rw1, height out_h).
    let mut a = vec![0i32; rw0 * out_h];
    let mut b = vec![0i32; rw1 * out_h];
    let mut col_low = vec![0i32; rh0.max(1)];
    let mut col_high = vec![0i32; rh1.max(1)];
    let mut col_out = vec![0i32; out_h];

    for x in 0..rw0 {
        for y in 0..rh0 {
            col_low[y] = ll.data[y * rw0 + x];
        }
        for y in 0..rh1 {
            col_high[y] = lh.data[y * rw0 + x];
        }
        idwt_1d_53(&col_low[..rh0], &col_high[..rh1], &mut col_out[..out_h]);
        for y in 0..out_h {
            a[y * rw0 + x] = col_out[y];
        }
    }
    for x in 0..rw1 {
        for y in 0..rh0 {
            col_low[y] = hl.data[y * rw1 + x];
        }
        for y in 0..rh1 {
            col_high[y] = hh.data[y * rw1 + x];
        }
        idwt_1d_53(&col_low[..rh0], &col_high[..rh1], &mut col_out[..out_h]);
        for y in 0..out_h {
            b[y * rw1 + x] = col_out[y];
        }
    }

    // Horizontal inverse: combine A (low) and B (high) per row → output.
    let mut out = vec![0i32; out_w * out_h];
    let mut row_out = vec![0i32; out_w];
    for y in 0..out_h {
        let lo = &a[y * rw0..y * rw0 + rw0];
        let hi = &b[y * rw1..y * rw1 + rw1];
        idwt_1d_53(lo, hi, &mut row_out[..out_w]);
        out[y * out_w..y * out_w + out_w].copy_from_slice(&row_out[..out_w]);
    }
    Band { w: out_w, h: out_h, data: out }
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
        cur = idwt_level(&cur, &hl, &lh, &hh, out_w, out_h);
    }
    Some(cur.data)
}

#[cfg(test)]
mod tests {
    use super::idwt_1d_53;

    use super::mirror;

    // Forward 5/3 analysis (interleaved, whole-sample symmetric) for round-trip.
    fn fwd_1d_53(x: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let n = x.len();
        let mut t = x.to_vec();
        if n >= 2 {
            let ni = n as isize;
            let mut k = 1;
            while k < n {
                let right = t[mirror(k as isize + 1, ni)];
                t[k] -= (t[k - 1] + right).div_euclid(2);
                k += 2;
            }
            let mut k = 0;
            while k < n {
                let left = t[mirror(k as isize - 1, ni)];
                let right = t[mirror(k as isize + 1, ni)];
                t[k] += (left + right + 2).div_euclid(4);
                k += 2;
            }
        }
        let nl = n.div_ceil(2);
        let nh = n / 2;
        let low: Vec<i32> = (0..nl).map(|i| t[2 * i]).collect();
        let high: Vec<i32> = (0..nh).map(|i| t[2 * i + 1]).collect();
        (low, high)
    }

    #[test]
    fn idwt_53_round_trips_for_all_lengths() {
        for n in 1..40usize {
            let x: Vec<i32> = (0..n as i32).map(|i| (i * 37 + 5) % 101 - 50).collect();
            let (low, high) = fwd_1d_53(&x);
            let mut out = vec![0i32; n];
            idwt_1d_53(&low, &high, &mut out);
            assert_eq!(out, x, "round-trip failed for length {}", n);
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
