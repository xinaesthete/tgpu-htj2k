//! HT cleanup-pass block decoder — a faithful Rust port of OpenJPH's
//! `ojph_decode_codeblock32` (BSD-2-Clause). Decodes one code-block's HT
//! cleanup pass (MEL + VLC + MagSgn) into sign-magnitude coefficients.
//!
//! Scope: the cleanup pass only (`num_passes == 1`), which is what OpenJPH
//! produces for pure-HT lossless/lossy. SigProp/MagRef are not yet ported.
//!
//! The bit readers are ported with 4-byte little-endian word reads over a
//! zero-padded buffer; OpenJPH's pointer-alignment handling is an optimization
//! that does not affect the decoded bitstream, so we mirror the structure
//! using buffer indices.

use crate::block_tables::{UVLC_TBL0, UVLC_TBL1, VLC_TBL0, VLC_TBL1};

const PAD: usize = 16;

#[inline]
fn load_le_u32(buf: &[u8], idx: usize) -> u32 {
    u32::from_le_bytes([buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]])
}

// ---------------------------------------------------------------------------
// MEL decoder (runs of zero events), MSB-first with byte-after-0xFF unstuffing.
// ---------------------------------------------------------------------------
struct Mel<'a> {
    buf: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: i32,
    size: i32,
    unstuff: bool,
    k: i32,
    num_runs: i32,
    runs: u64,
}

const MEL_EXP: [i32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

impl<'a> Mel<'a> {
    fn init(buf: &'a [u8], base: usize, lcup: i32, scup: i32) -> Self {
        let mut m = Mel {
            buf,
            pos: base + (lcup - scup) as usize,
            tmp: 0,
            bits: 0,
            size: scup - 1,
            unstuff: false,
            k: 0,
            num_runs: 0,
            runs: 0,
        };
        let num = 4 - (m.pos & 3) as i32;
        for _ in 0..num {
            let mut d: u64 = if m.size > 0 { m.buf[m.pos] as u64 } else { 0xFF };
            if m.size == 1 {
                d |= 0xF;
            }
            let inc = if m.size > 0 { 1 } else { 0 };
            m.size -= 1;
            m.pos += inc;
            let d_bits = 8 - m.unstuff as i32;
            m.tmp = (m.tmp << d_bits) | d;
            m.bits += d_bits;
            m.unstuff = (d & 0xFF) == 0xFF;
        }
        m.tmp <<= 64 - m.bits;
        m
    }

    fn read(&mut self) {
        if self.bits > 32 {
            return;
        }
        let mut val: u32 = 0xFFFFFFFF;
        if self.size > 4 {
            val = load_le_u32(self.buf, self.pos);
            self.pos += 4;
            self.size -= 4;
        } else if self.size > 0 {
            let mut i = 0;
            while self.size > 1 {
                let v = self.buf[self.pos] as u32;
                self.pos += 1;
                let m = !(0xFFu32 << i);
                val = (val & m) | (v << i);
                self.size -= 1;
                i += 8;
            }
            let v = (self.buf[self.pos] as u32) | 0xF;
            self.pos += 1;
            let m = !(0xFFu32 << i);
            val = (val & m) | (v << i);
            self.size -= 1;
        }
        let mut bits = 32 - self.unstuff as i32;
        let mut t = val & 0xFF;
        let mut un = (val & 0xFF) == 0xFF;
        bits -= un as i32;
        t <<= 8 - un as i32;
        t |= (val >> 8) & 0xFF;
        un = ((val >> 8) & 0xFF) == 0xFF;
        bits -= un as i32;
        t <<= 8 - un as i32;
        t |= (val >> 16) & 0xFF;
        un = ((val >> 16) & 0xFF) == 0xFF;
        bits -= un as i32;
        t <<= 8 - un as i32;
        t |= (val >> 24) & 0xFF;
        self.unstuff = ((val >> 24) & 0xFF) == 0xFF;
        self.tmp |= (t as u64) << (64 - bits - self.bits);
        self.bits += bits;
    }

    fn decode(&mut self) {
        if self.bits < 6 {
            self.read();
        }
        while self.bits >= 6 && self.num_runs < 8 {
            let eval = MEL_EXP[self.k as usize];
            let run;
            if self.tmp & (1u64 << 63) != 0 {
                let mut r = 1 << eval;
                r -= 1;
                self.k = if self.k + 1 < 12 { self.k + 1 } else { 12 };
                self.tmp <<= 1;
                self.bits -= 1;
                run = r << 1;
            } else {
                let r = (self.tmp >> (63 - eval)) as i32 & ((1 << eval) - 1);
                self.k = if self.k - 1 > 0 { self.k - 1 } else { 0 };
                self.tmp <<= eval + 1;
                self.bits -= eval + 1;
                run = (r << 1) + 1;
            }
            let shift = self.num_runs * 7;
            self.runs &= !(0x3Fu64 << shift);
            self.runs |= (run as u64) << shift;
            self.num_runs += 1;
        }
    }

    fn get_run(&mut self) -> i32 {
        if self.num_runs == 0 {
            self.decode();
        }
        let t = (self.runs & 0x7F) as i32;
        self.runs >>= 7;
        self.num_runs -= 1;
        t
    }
}

// ---------------------------------------------------------------------------
// Reverse (VLC) reader — backward-growing, LSB-first.
// ---------------------------------------------------------------------------
struct Rev<'a> {
    buf: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: u32,
    size: i32,
    unstuff: bool,
}

impl<'a> Rev<'a> {
    fn init(buf: &'a [u8], base: usize, lcup: i32, scup: i32) -> Self {
        let mut v = Rev {
            buf,
            pos: base + (lcup - 2) as usize,
            tmp: 0,
            bits: 0,
            size: scup - 2,
            unstuff: false,
        };
        let d = v.buf[v.pos] as u32;
        v.pos = v.pos.wrapping_sub(1);
        v.tmp = (d >> 4) as u64;
        v.bits = 4 - if (v.tmp & 7) == 7 { 1 } else { 0 };
        v.unstuff = (d | 0xF) > 0x8F;
        let num = 1 + (v.pos & 3) as i32;
        let tnum = num.min(v.size);
        for _ in 0..tnum {
            let d = v.buf[v.pos] as u64;
            v.pos = v.pos.wrapping_sub(1);
            let d_bits = 8 - if v.unstuff && (d & 0x7F) == 0x7F { 1 } else { 0 };
            v.tmp |= d << v.bits;
            v.bits += d_bits;
            v.unstuff = d > 0x8F;
        }
        v.size -= tnum;
        v.read();
        v
    }

    fn read(&mut self) {
        if self.bits > 32 {
            return;
        }
        let mut val: u32 = 0;
        if self.size > 3 {
            val = load_le_u32(self.buf, self.pos - 3);
            self.pos = self.pos.wrapping_sub(4);
            self.size -= 4;
        } else if self.size > 0 {
            let mut i = 24i32;
            while self.size > 0 {
                let v = self.buf[self.pos] as u32;
                self.pos = self.pos.wrapping_sub(1);
                val |= v << i;
                self.size -= 1;
                i -= 8;
            }
        }
        let mut t = val >> 24;
        let mut bits = 8 - if self.unstuff && ((val >> 24) & 0x7F) == 0x7F { 1 } else { 0 };
        let mut un = (val >> 24) > 0x8F;
        t |= ((val >> 16) & 0xFF) << bits;
        bits += 8 - if un && ((val >> 16) & 0x7F) == 0x7F { 1 } else { 0 };
        un = ((val >> 16) & 0xFF) > 0x8F;
        t |= ((val >> 8) & 0xFF) << bits;
        bits += 8 - if un && ((val >> 8) & 0x7F) == 0x7F { 1 } else { 0 };
        un = ((val >> 8) & 0xFF) > 0x8F;
        t |= (val & 0xFF) << bits;
        bits += 8 - if un && (val & 0x7F) == 0x7F { 1 } else { 0 };
        un = (val & 0xFF) > 0x8F;
        self.tmp |= (t as u64) << self.bits;
        self.bits += bits;
        self.unstuff = un;
    }

    fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.read();
            if self.bits < 32 {
                self.read();
            }
        }
        self.tmp as u32
    }

    fn advance(&mut self, num_bits: u32) -> u32 {
        self.tmp >>= num_bits;
        self.bits -= num_bits;
        self.tmp as u32
    }
}

// ---------------------------------------------------------------------------
// Forward (MagSgn) reader — forward-growing, LSB-first. Feeds 0xFF past end.
// ---------------------------------------------------------------------------
struct Frwd<'a> {
    buf: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: u32,
    unstuff: u32,
    size: i32,
}

impl<'a> Frwd<'a> {
    fn init(buf: &'a [u8], base: usize, size: i32) -> Self {
        let mut m = Frwd { buf, pos: base, tmp: 0, bits: 0, unstuff: 0, size };
        let num = 4 - (m.pos & 3) as i32;
        for _ in 0..num {
            let old = m.size;
            m.size -= 1;
            let d: u64 = if old > 0 {
                let b = m.buf[m.pos] as u64;
                m.pos += 1;
                b
            } else {
                0xFF
            };
            m.tmp |= d << m.bits;
            m.bits += 8 - m.unstuff;
            m.unstuff = ((d & 0xFF) == 0xFF) as u32;
        }
        m.read();
        m
    }

    fn read(&mut self) {
        let mut val: u32;
        if self.size > 3 {
            val = load_le_u32(self.buf, self.pos);
            self.pos += 4;
            self.size -= 4;
        } else if self.size > 0 {
            let mut i = 0;
            val = 0xFFFFFFFF;
            while self.size > 0 {
                let v = self.buf[self.pos] as u32;
                self.pos += 1;
                let m = !(0xFFu32 << i);
                val = (val & m) | (v << i);
                self.size -= 1;
                i += 8;
            }
        } else {
            val = 0xFFFFFFFF;
        }
        let mut bits = 8 - self.unstuff;
        let mut t = val & 0xFF;
        let mut un = (val & 0xFF) == 0xFF;
        t |= ((val >> 8) & 0xFF) << bits;
        bits += 8 - un as u32;
        un = ((val >> 8) & 0xFF) == 0xFF;
        t |= ((val >> 16) & 0xFF) << bits;
        bits += 8 - un as u32;
        un = ((val >> 16) & 0xFF) == 0xFF;
        t |= ((val >> 24) & 0xFF) << bits;
        bits += 8 - un as u32;
        self.unstuff = (((val >> 24) & 0xFF) == 0xFF) as u32;
        self.tmp |= (t as u64) << self.bits;
        self.bits += bits;
    }

    fn advance(&mut self, num_bits: u32) {
        self.tmp >>= num_bits;
        self.bits -= num_bits;
    }

    fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.read();
            if self.bits < 32 {
                self.read();
            }
        }
        self.tmp as u32
    }
}

/// Decode one code-block's HT cleanup pass into sign-magnitude coefficients
/// (row-major, `stride == width`). Bit 31 is the sign, magnitude in low bits.
/// Returns None on a malformed code-block.
pub fn decode_cleanup(
    coded: &[u8],
    missing_msbs: u32,
    lengths1: u32,
    width: u32,
    height: u32,
) -> Option<Vec<u32>> {
    if missing_msbs > 30 || lengths1 < 2 {
        return None;
    }
    let lcup = lengths1 as i32;
    if (lcup as usize) > coded.len() {
        return None;
    }
    // Padded buffer so the backward/forward word reads stay in bounds.
    let mut buf = vec![0u8; PAD + lcup as usize + PAD];
    buf[PAD..PAD + lcup as usize].copy_from_slice(&coded[..lcup as usize]);
    let base = PAD;

    let scup = ((buf[base + lcup as usize - 1] as i32) << 4)
        + (buf[base + lcup as usize - 2] as i32 & 0xF);
    if scup < 2 || scup > lcup || scup > 4079 {
        return None;
    }

    let width = width as usize;
    let height = height as usize;
    let stride = width;
    let p = 30i32 - missing_msbs as i32;
    let mmsbp2 = missing_msbs + 2;
    let sstr = (((width as u32 + 2) + 7) & !7u32) as usize;

    // The quad scan processes ceil(height/2) quad-rows and writes both samples
    // of each quad, so the output buffer needs an even number of rows.
    let alloc_h = height + (height & 1);
    let mut scratch = vec![0u16; sstr * ((height + 1) / 2 + 2)];
    let mut decoded = vec![0u32; stride * alloc_h];

    // ---- Step 1: VLC + MEL -> scratch (inf, u_q per quad) ----
    {
        let mut mel = Mel::init(&buf, base, lcup, scup);
        let mut vlc = Rev::init(&buf, base, lcup, scup);
        let mut run = mel.get_run();
        let mut c_q: u32;

        // initial quad row
        let mut sp = 0usize;
        c_q = 0;
        let mut x = 0usize;
        while x < width {
            let mut vlc_val = vlc.fetch();
            let mut t0 = VLC_TBL0[(c_q + (vlc_val & 0x7F)) as usize];
            if c_q == 0 {
                run -= 2;
                t0 = if run == -1 { t0 } else { 0 };
                if run < 0 {
                    run = mel.get_run();
                }
            }
            scratch[sp] = t0;
            x += 2;
            c_q = ((t0 as u32 & 0x10) << 3) | ((t0 as u32 & 0xE0) << 2);
            vlc_val = vlc.advance(t0 as u32 & 0x7);

            let mut t1 = VLC_TBL0[(c_q + (vlc_val & 0x7F)) as usize];
            if c_q == 0 && x < width {
                run -= 2;
                t1 = if run == -1 { t1 } else { 0 };
                if run < 0 {
                    run = mel.get_run();
                }
            }
            t1 = if x < width { t1 } else { 0 };
            scratch[sp + 2] = t1;
            x += 2;
            c_q = ((t1 as u32 & 0x10) << 3) | ((t1 as u32 & 0xE0) << 2);
            vlc_val = vlc.advance(t1 as u32 & 0x7);

            let mut uvlc_mode = ((t0 as u32 & 0x8) << 3) | ((t1 as u32 & 0x8) << 4);
            if uvlc_mode == 0xc0 {
                run -= 2;
                uvlc_mode += if run == -1 { 0x40 } else { 0 };
                if run < 0 {
                    run = mel.get_run();
                }
            }
            let mut uvlc_entry = UVLC_TBL0[(uvlc_mode + (vlc_val & 0x3F)) as usize] as u32;
            vlc_val = vlc.advance(uvlc_entry & 0x7);
            uvlc_entry >>= 3;
            let len = uvlc_entry & 0xF;
            let tmp = vlc_val & ((1 << len) - 1);
            vlc.advance(len);
            uvlc_entry >>= 4;
            let len = uvlc_entry & 0x7;
            uvlc_entry >>= 3;
            scratch[sp + 1] = (1 + (uvlc_entry & 7) + (tmp & !(0xFFu32 << len))) as u16;
            scratch[sp + 3] = (1 + (uvlc_entry >> 3) + (tmp >> len)) as u16;
            sp += 4;
        }
        scratch[sp] = 0;
        scratch[sp + 1] = 0;

        // non-initial quad rows
        let mut y = 2usize;
        while y < height {
            c_q = 0;
            let row = (y >> 1) * sstr;
            let mut sp = row;
            let mut x = 0usize;
            while x < width {
                c_q |= (scratch[sp - sstr] as u32 & 0xA0) << 2;
                c_q |= (scratch[sp + 2 - sstr] as u32 & 0x20) << 4;

                let mut vlc_val = vlc.fetch();
                let mut t0 = VLC_TBL1[(c_q + (vlc_val & 0x7F)) as usize];
                if c_q == 0 {
                    run -= 2;
                    t0 = if run == -1 { t0 } else { 0 };
                    if run < 0 {
                        run = mel.get_run();
                    }
                }
                scratch[sp] = t0;
                x += 2;
                c_q = ((t0 as u32 & 0x40) << 2) | ((t0 as u32 & 0x80) << 1);
                c_q |= scratch[sp - sstr] as u32 & 0x80;
                c_q |= (scratch[sp + 2 - sstr] as u32 & 0xA0) << 2;
                c_q |= (scratch[sp + 4 - sstr] as u32 & 0x20) << 4;
                vlc_val = vlc.advance(t0 as u32 & 0x7);

                let mut t1 = VLC_TBL1[(c_q + (vlc_val & 0x7F)) as usize];
                if c_q == 0 && x < width {
                    run -= 2;
                    t1 = if run == -1 { t1 } else { 0 };
                    if run < 0 {
                        run = mel.get_run();
                    }
                }
                t1 = if x < width { t1 } else { 0 };
                scratch[sp + 2] = t1;
                x += 2;
                c_q = ((t1 as u32 & 0x40) << 2) | ((t1 as u32 & 0x80) << 1);
                c_q |= scratch[sp + 2 - sstr] as u32 & 0x80;
                vlc_val = vlc.advance(t1 as u32 & 0x7);

                let uvlc_mode = ((t0 as u32 & 0x8) << 3) | ((t1 as u32 & 0x8) << 4);
                let mut uvlc_entry = UVLC_TBL1[(uvlc_mode + (vlc_val & 0x3F)) as usize] as u32;
                vlc_val = vlc.advance(uvlc_entry & 0x7);
                uvlc_entry >>= 3;
                let len = uvlc_entry & 0xF;
                let tmp = vlc_val & ((1 << len) - 1);
                vlc.advance(len);
                uvlc_entry >>= 4;
                let len = uvlc_entry & 0x7;
                uvlc_entry >>= 3;
                scratch[sp + 1] = ((uvlc_entry & 7) + (tmp & !(0xFFu32 << len))) as u16;
                scratch[sp + 3] = ((uvlc_entry >> 3) + (tmp >> len)) as u16;
                sp += 4;
            }
            scratch[sp] = 0;
            scratch[sp + 1] = 0;
            y += 2;
        }
    }

    // ---- Step 2: MagSgn -> sign-magnitude coefficients ----
    {
        let mut magsgn = Frwd::init(&buf, base, lcup - scup);
        let v_n_size = width / 2 + 4;
        let mut v_n = vec![0u32; v_n_size];

        // initial row
        let mut sp = 0usize;
        let mut vp = 0usize;
        let mut dp = 0usize;
        let mut prev_v_n = 0u32;
        let mut x = 0usize;
        while x < width {
            let inf = scratch[sp] as u32;
            let u_q = scratch[sp + 1] as u32;
            if u_q > mmsbp2 {
                return None;
            }
            let mut vn;
            let mut val = 0u32;
            if inf & (1 << 4) != 0 {
                let ms = magsgn.fetch();
                let m_n = u_q - ((inf >> 12) & 1);
                magsgn.advance(m_n);
                val = ms << 31;
                let mut t = ms & ((1 << m_n) - 1);
                t |= ((inf >> 8) & 1) << m_n;
                t |= 1;
                vn = t;
                val |= (vn + 2) << (p - 1);
            } else {
                vn = 0;
            }
            decoded[dp] = val;

            val = 0;
            if inf & (1 << 5) != 0 {
                let ms = magsgn.fetch();
                let m_n = u_q - ((inf >> 13) & 1);
                magsgn.advance(m_n);
                val = ms << 31;
                let mut t = ms & ((1 << m_n) - 1);
                t |= ((inf >> 9) & 1) << m_n;
                t |= 1;
                vn = t;
                val |= (vn + 2) << (p - 1);
            } else {
                vn = 0;
            }
            decoded[dp + stride] = val;
            v_n[vp] = prev_v_n | vn;
            prev_v_n = 0;
            dp += 1;
            x += 1;
            if x >= width {
                vp += 1;
                break;
            }

            val = 0;
            if inf & (1 << 6) != 0 {
                let ms = magsgn.fetch();
                let m_n = u_q - ((inf >> 14) & 1);
                magsgn.advance(m_n);
                val = ms << 31;
                let mut t = ms & ((1 << m_n) - 1);
                t |= ((inf >> 10) & 1) << m_n;
                t |= 1;
                val |= (t + 2) << (p - 1);
            }
            decoded[dp] = val;

            val = 0;
            if inf & (1 << 7) != 0 {
                let ms = magsgn.fetch();
                let m_n = u_q - ((inf >> 15) & 1);
                magsgn.advance(m_n);
                val = ms << 31;
                let mut t = ms & ((1 << m_n) - 1);
                t |= ((inf >> 11) & 1) << m_n;
                t |= 1;
                vn = t;
                val |= (vn + 2) << (p - 1);
            } else {
                vn = 0;
            }
            decoded[dp + stride] = val;
            prev_v_n = vn;
            dp += 1;
            x += 1;
            sp += 2;
            vp += 1;
        }
        v_n[vp] = prev_v_n;

        // non-initial rows
        let mut y = 2usize;
        while y < height {
            let mut sp = (y >> 1) * sstr;
            let mut vp = 0usize;
            let mut dp = y * stride;
            let mut prev_v_n = 0u32;
            let mut x = 0usize;
            while x < width {
                let inf = scratch[sp] as u32;
                let u = scratch[sp + 1] as u32;

                let mut gamma = inf & 0xF0;
                gamma &= gamma.wrapping_sub(0x10);
                let e = v_n[vp] | v_n[vp + 1];
                let emax = 31 - (e | 2).leading_zeros();
                let kappa = if gamma != 0 { emax } else { 1 };
                let u_q = u + kappa;
                if u_q > mmsbp2 {
                    return None;
                }

                let mut vn;
                let mut val = 0u32;
                if inf & (1 << 4) != 0 {
                    let ms = magsgn.fetch();
                    let m_n = u_q - ((inf >> 12) & 1);
                    magsgn.advance(m_n);
                    val = ms << 31;
                    let mut t = ms & ((1 << m_n) - 1);
                    t |= ((inf >> 8) & 1) << m_n;
                    t |= 1;
                    val |= (t + 2) << (p - 1);
                }
                decoded[dp] = val;

                val = 0;
                if inf & (1 << 5) != 0 {
                    let ms = magsgn.fetch();
                    let m_n = u_q - ((inf >> 13) & 1);
                    magsgn.advance(m_n);
                    val = ms << 31;
                    let mut t = ms & ((1 << m_n) - 1);
                    t |= ((inf >> 9) & 1) << m_n;
                    t |= 1;
                    vn = t;
                    val |= (vn + 2) << (p - 1);
                } else {
                    vn = 0;
                }
                decoded[dp + stride] = val;
                v_n[vp] = prev_v_n | vn;
                prev_v_n = 0;
                dp += 1;
                x += 1;
                if x >= width {
                    vp += 1;
                    break;
                }

                val = 0;
                if inf & (1 << 6) != 0 {
                    let ms = magsgn.fetch();
                    let m_n = u_q - ((inf >> 14) & 1);
                    magsgn.advance(m_n);
                    val = ms << 31;
                    let mut t = ms & ((1 << m_n) - 1);
                    t |= ((inf >> 10) & 1) << m_n;
                    t |= 1;
                    val |= (t + 2) << (p - 1);
                }
                decoded[dp] = val;

                val = 0;
                if inf & (1 << 7) != 0 {
                    let ms = magsgn.fetch();
                    let m_n = u_q - ((inf >> 15) & 1);
                    magsgn.advance(m_n);
                    val = ms << 31;
                    let mut t = ms & ((1 << m_n) - 1);
                    t |= ((inf >> 11) & 1) << m_n;
                    t |= 1;
                    vn = t;
                    val |= (vn + 2) << (p - 1);
                } else {
                    vn = 0;
                }
                decoded[dp + stride] = val;
                prev_v_n = vn;
                dp += 1;
                x += 1;
                sp += 2;
                vp += 1;
            }
            v_n[vp] = prev_v_n;
            y += 2;
        }
    }

    decoded.truncate(stride * height); // drop the even-alignment slack row
    Some(decoded)
}

/// Convert sign-magnitude coefficients to signed integers for a reversible
/// (lossless) subband: `coeff = ±(magnitude >> (31 - k_max))`.
pub fn reversible_to_i32(decoded: &[u32], k_max: u32) -> Vec<i32> {
    let shift = 31 - k_max;
    decoded
        .iter()
        .map(|&v| {
            let val = ((v & 0x7FFF_FFFF) >> shift) as i32;
            if v & 0x8000_0000 != 0 {
                -val
            } else {
                val
            }
        })
        .collect()
}
