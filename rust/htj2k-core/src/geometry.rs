//! Resolution / subband / code-block geometry for a tile-component.
//!
//! Pure integer geometry derived from the codestream parameters (image size,
//! decomposition levels, code-block size). No bitstream involved — this is the
//! structure the inverse DWT and the HT block decoder iterate over. Formulas
//! follow ITU-T T.800 (JPEG 2000 Part 1), Annex B.

/// Subband orientation. LL only exists at the coarsest resolution (level 0).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Orientation {
    LL = 0,
    HL = 1, // horizontally high-pass
    LH = 2, // vertically high-pass
    HH = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct Subband {
    pub orientation: Orientation,
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
    pub num_cblks_x: u32,
    pub num_cblks_y: u32,
}

impl Subband {
    pub fn width(&self) -> u32 {
        (self.x1 - self.x0).max(0) as u32
    }
    pub fn height(&self) -> u32 {
        (self.y1 - self.y0).max(0) as u32
    }
    pub fn num_code_blocks(&self) -> u32 {
        self.num_cblks_x * self.num_cblks_y
    }
}

#[derive(Clone, Debug)]
pub struct Resolution {
    /// Resolution index, 0 = coarsest (the LL pyramid base).
    pub index: u32,
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
    pub subbands: Vec<Subband>,
}

impl Resolution {
    pub fn width(&self) -> u32 {
        (self.x1 - self.x0).max(0) as u32
    }
    pub fn height(&self) -> u32 {
        (self.y1 - self.y0).max(0) as u32
    }
}

#[derive(Clone, Debug)]
pub struct ComponentLayout {
    pub resolutions: Vec<Resolution>,
}

impl ComponentLayout {
    pub fn total_code_blocks(&self) -> u32 {
        self.resolutions
            .iter()
            .flat_map(|r| r.subbands.iter())
            .map(Subband::num_code_blocks)
            .sum()
    }
}

/// Ceiling division for q > 0, defined for negative numerators (T.800 uses this).
fn ceil_div(p: i64, q: i64) -> i64 {
    debug_assert!(q > 0);
    (p + q - 1).div_euclid(q)
}

fn floor_div(p: i64, q: i64) -> i64 {
    debug_assert!(q > 0);
    p.div_euclid(q)
}

/// Number of code-blocks spanning `[a0, a1)` against a partition grid of pitch
/// `cb` anchored at the origin.
fn cblk_count(a0: i64, a1: i64, cb: i64) -> u32 {
    if a1 <= a0 {
        return 0;
    }
    (ceil_div(a1, cb) - floor_div(a0, cb)) as u32
}

/// Compute the resolution/subband/code-block layout for a single tile-component.
///
/// `(tcx0, tcy0)` is the tile-component top-left (the image offset for a
/// single-tile codestream); `(tcx1, tcy1)` is the bottom-right (the image
/// extent). `num_decompositions` is the DWT depth N; `cbw`/`cbh` are the
/// nominal code-block dimensions (powers of two).
pub fn compute_component_layout(
    tcx0: i64,
    tcy0: i64,
    tcx1: i64,
    tcy1: i64,
    num_decompositions: u32,
    cbw: u32,
    cbh: u32,
) -> ComponentLayout {
    let n = num_decompositions as i64;
    let cbw = cbw as i64;
    let cbh = cbh as i64;
    let mut resolutions = Vec::with_capacity(num_decompositions as usize + 1);

    for r in 0..=num_decompositions {
        // Resolution coordinates (T.800 B-14): divide by 2^(N - r).
        let denom_shift = n - r as i64;
        let rdiv = 1i64 << denom_shift;
        let rx0 = ceil_div(tcx0, rdiv);
        let ry0 = ceil_div(tcy0, rdiv);
        let rx1 = ceil_div(tcx1, rdiv);
        let ry1 = ceil_div(tcy1, rdiv);

        let mut subbands = Vec::new();
        if r == 0 {
            subbands.push(make_subband(Orientation::LL, tcx0, tcy0, tcx1, tcy1, n, cbw, cbh));
        } else {
            // Detail subbands at decomposition level nb = N - r + 1.
            let nb = n - r as i64 + 1;
            for orientation in [Orientation::HL, Orientation::LH, Orientation::HH] {
                subbands.push(make_subband(orientation, tcx0, tcy0, tcx1, tcy1, nb, cbw, cbh));
            }
        }

        resolutions.push(Resolution {
            index: r,
            x0: rx0,
            y0: ry0,
            x1: rx1,
            y1: ry1,
            subbands,
        });
    }

    ComponentLayout { resolutions }
}

fn make_subband(
    orientation: Orientation,
    tcx0: i64,
    tcy0: i64,
    tcx1: i64,
    tcy1: i64,
    nb: i64,
    cbw: i64,
    cbh: i64,
) -> Subband {
    let (xob, yob): (i64, i64) = match orientation {
        Orientation::LL => (0, 0),
        Orientation::HL => (1, 0),
        Orientation::LH => (0, 1),
        Orientation::HH => (1, 1),
    };
    let div = 1i64 << nb;
    let half = 1i64 << (nb - 1);
    // Subband coordinates (T.800 B-15).
    let x0 = ceil_div(tcx0 - half * xob, div);
    let x1 = ceil_div(tcx1 - half * xob, div);
    let y0 = ceil_div(tcy0 - half * yob, div);
    let y1 = ceil_div(tcy1 - half * yob, div);
    Subband {
        orientation,
        x0,
        y0,
        x1,
        y1,
        num_cblks_x: cblk_count(x0, x1, cbw),
        num_cblks_y: cblk_count(y0, y1, cbh),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_count_and_full_size() {
        let layout = compute_component_layout(0, 0, 64, 48, 5, 64, 64);
        assert_eq!(layout.resolutions.len(), 6); // N + 1
        let top = layout.resolutions.last().unwrap();
        assert_eq!((top.width(), top.height()), (64, 48));
        let base = &layout.resolutions[0];
        assert_eq!(base.subbands.len(), 1);
        assert_eq!(base.subbands[0].orientation, Orientation::LL);
    }

    #[test]
    fn each_detail_resolution_has_three_subbands() {
        let layout = compute_component_layout(0, 0, 100, 70, 4, 32, 32);
        for r in 1..layout.resolutions.len() {
            assert_eq!(layout.resolutions[r].subbands.len(), 3);
            let orients: Vec<_> = layout.resolutions[r]
                .subbands
                .iter()
                .map(|s| s.orientation)
                .collect();
            assert_eq!(orients, vec![Orientation::HL, Orientation::LH, Orientation::HH]);
        }
    }

    #[test]
    fn resolution_dims_halve_each_step() {
        // res[r-1].w == ceil(res[r].w / 2), and likewise for height.
        let layout = compute_component_layout(0, 0, 333, 211, 5, 64, 64);
        for r in 1..layout.resolutions.len() {
            let cur = &layout.resolutions[r];
            let prev = &layout.resolutions[r - 1];
            assert_eq!(prev.width(), cur.width().div_ceil(2));
            assert_eq!(prev.height(), cur.height().div_ceil(2));
        }
    }

    #[test]
    fn code_blocks_cover_each_subband() {
        // Image offset 0 => code-block grid is anchored at the subband origin,
        // so the count is simply ceil(dim / cb).
        let layout = compute_component_layout(0, 0, 256, 256, 5, 64, 64);
        for res in &layout.resolutions {
            for sb in &res.subbands {
                let expect_x = if sb.width() > 0 { sb.width().div_ceil(64) } else { 0 };
                let expect_y = if sb.height() > 0 { sb.height().div_ceil(64) } else { 0 };
                assert_eq!(sb.num_cblks_x, expect_x);
                assert_eq!(sb.num_cblks_y, expect_y);
            }
        }
        assert!(layout.total_code_blocks() > 0);
    }
}
