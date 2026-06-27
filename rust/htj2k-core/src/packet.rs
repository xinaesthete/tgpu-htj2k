//! Packet-header parsing for the simple case we target in v1: single tile,
//! single quality layer, single precinct per resolution, no SOP/EPH markers.
//!
//! Ported faithfully from OpenJPH `precinct::parse` + `bitbuffer_read.h`
//! (BSD-2-Clause). Produces, per code-block, whether it is included, its
//! missing-MSB count, number of coding passes, and the byte range of its
//! compressed (HT) data within the codestream.

use crate::geometry::{ComponentLayout, Orientation};

/// MSB-first bit reader with the JPEG 2000 packet-header bit-unstuffing rule:
/// the byte following a 0xFF byte contributes only its low 7 bits.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    tmp: u32,
    avail_bits: i32,
    unstuff: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], start: usize, end: usize) -> Self {
        BitReader { data, pos: start, end, tmp: 0, avail_bits: 0, unstuff: false }
    }

    fn fill(&mut self) {
        if self.pos < self.end {
            let t = self.data[self.pos];
            self.pos += 1;
            self.tmp = t as u32;
            self.avail_bits = 8 - self.unstuff as i32;
            self.unstuff = t == 0xFF;
        } else {
            // Past end: yield zero bits without consuming (matches OpenJPH bb_read).
            self.tmp = 0;
            self.avail_bits = 8 - self.unstuff as i32;
            self.unstuff = false;
        }
    }

    fn read_bit(&mut self) -> u32 {
        if self.avail_bits == 0 {
            self.fill();
        }
        self.avail_bits -= 1;
        (self.tmp >> self.avail_bits) & 1
    }

    fn read_bits(&mut self, mut num_bits: i32) -> u32 {
        let mut bits = 0u32;
        while num_bits > 0 {
            if self.avail_bits == 0 {
                self.fill();
            }
            let tx = self.avail_bits.min(num_bits);
            bits <<= tx;
            self.avail_bits -= tx;
            num_bits -= tx;
            bits |= (self.tmp >> self.avail_bits) & ((1u32 << tx) - 1);
        }
        bits
    }

    /// Byte-align after the packet header (consume the trailing stuff byte if
    /// the last header byte was 0xFF).
    fn terminate(&mut self) {
        if self.unstuff {
            self.fill();
        }
        self.tmp = 0;
        self.avail_bits = 0;
        self.unstuff = false;
    }

    /// Byte offset of the next unread byte (the packet body / next packet).
    fn byte_pos(&self) -> usize {
        self.pos
    }
}

/// A simple quad tag-tree as used by OpenJPH's packet parser (value + received
/// flag per node, levels stored densely).
struct TagTree {
    num_levels: u32,
    dim_w: Vec<usize>,
    values: Vec<Vec<u32>>,
    flags: Vec<Vec<u8>>,
}

fn log2_ceil(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

impl TagTree {
    fn new(w: u32, h: u32) -> Self {
        let num_levels = 1 + log2_ceil(w).max(log2_ceil(h));
        let mut dim_w = Vec::new();
        let mut values = Vec::new();
        let mut flags = Vec::new();
        for lev in 0..=num_levels {
            let lw = ((w + (1 << lev) - 1) >> lev).max(1) as usize;
            let lh = ((h + (1 << lev) - 1) >> lev).max(1) as usize;
            dim_w.push(lw);
            values.push(vec![0u32; lw * lh]);
            flags.push(vec![0u8; lw * lh]);
        }
        TagTree { num_levels, dim_w, values, flags }
    }

    fn idx(&self, x: u32, y: u32, lev: u32) -> usize {
        let w = self.dim_w[lev as usize];
        x as usize + y as usize * w
    }
    fn value(&self, x: u32, y: u32, lev: u32) -> u32 {
        self.values[lev as usize][self.idx(x, y, lev)]
    }
    fn set_value(&mut self, x: u32, y: u32, lev: u32, v: u32) {
        let i = self.idx(x, y, lev);
        self.values[lev as usize][i] = v;
    }
    fn flag(&self, x: u32, y: u32, lev: u32) -> u8 {
        self.flags[lev as usize][self.idx(x, y, lev)]
    }
    fn set_flag(&mut self, x: u32, y: u32, lev: u32) {
        let i = self.idx(x, y, lev);
        self.flags[lev as usize][i] = 1;
    }
}

#[derive(Clone, Debug)]
pub struct CodeBlock {
    pub component: u32,
    pub resolution: u32,
    pub subband: Orientation,
    pub x: u32,
    pub y: u32,
    pub included: bool,
    pub missing_msbs: u32,
    pub num_passes: u32,
    /// Byte offset of this code-block's HT data within the codestream.
    pub offset: u32,
    /// Byte length of this code-block's HT data.
    pub length: u32,
}

#[derive(Clone, Debug)]
pub struct PacketParse {
    pub code_blocks: Vec<CodeBlock>,
    pub bytes_consumed: u32,
}

/// Parse all packets of a single-tile, single-layer, single-component
/// codestream into per-code-block byte ranges.
pub fn parse_packets(
    data: &[u8],
    tile_offset: usize,
    tile_len: usize,
    layout: &ComponentLayout,
    num_components: u32,
) -> Result<PacketParse, String> {
    let tile_end = tile_offset + tile_len;
    let mut offset = tile_offset;
    let mut code_blocks = Vec::new();

    // RPCL progression, single precinct and single layer: for each resolution,
    // one packet per component. (v1 assumes all components share geometry.)
    for res in &layout.resolutions {
      for component in 0..num_components {
        let mut br = BitReader::new(data, offset, tile_end);
        let mut empty_packet = true;
        // Per-subband working state captured for the body pass.
        struct Cb {
            res: u32,
            orient: Orientation,
            x: u32,
            y: u32,
            len: u32,
            missing_msbs: u32,
            num_passes: u32,
        }
        let mut packet_cbs: Vec<Cb> = Vec::new();

        for sb in &res.subbands {
            let ncx = sb.num_cblks_x;
            let ncy = sb.num_cblks_y;
            if ncx == 0 || ncy == 0 {
                continue;
            }
            if empty_packet {
                // One bit signals whether the packet has any content.
                if br.read_bit() == 0 {
                    br.terminate();
                    offset = br.byte_pos();
                    // Empty packet: nothing in any subband. Move on.
                    // (subbands after the first are also empty for this packet)
                    break;
                }
                empty_packet = false;
            }

            let mut inc = TagTree::new(ncx, ncy);
            let mut mmsb = TagTree::new(ncx, ncy);

            for y in 0..ncy {
                for x in 0..ncx {
                    // Inclusion (first-layer) tag-tree walk.
                    let mut empty_cb = false;
                    let mut cl = inc.num_levels;
                    while cl > 0 {
                        let cur = cl - 1;
                        let xv = x >> cur;
                        let yv = y >> cur;
                        empty_cb = inc.value(xv, yv, cur) == 1;
                        if empty_cb {
                            break;
                        }
                        if inc.flag(xv, yv, cur) == 0 {
                            let bit = br.read_bit();
                            empty_cb = bit == 0;
                            inc.set_value(xv, yv, cur, 1 - bit);
                            inc.set_flag(xv, yv, cur);
                        }
                        if empty_cb {
                            break;
                        }
                        cl -= 1;
                    }
                    if empty_cb {
                        continue;
                    }

                    // Missing-MSBs (zero-bit-plane) tag-tree walk.
                    let mut mmsbs = 0u32;
                    let mut levp1 = mmsb.num_levels;
                    while levp1 > 0 {
                        let cur = levp1 - 1;
                        mmsbs = mmsb.value(x >> levp1, y >> levp1, levp1);
                        if mmsb.flag(x >> cur, y >> cur, cur) == 0 {
                            loop {
                                let bit = br.read_bit();
                                mmsbs += 1 - bit;
                                if bit != 0 {
                                    break;
                                }
                            }
                            mmsb.set_value(x >> cur, y >> cur, cur, mmsbs);
                            mmsb.set_flag(x >> cur, y >> cur, cur);
                        } else {
                            mmsbs = mmsb.value(x >> cur, y >> cur, cur);
                        }
                        levp1 -= 1;
                    }

                    // Number of coding passes.
                    let mut num_passes = 1u32;
                    if br.read_bit() != 0 {
                        num_passes = 2;
                        if br.read_bit() != 0 {
                            let b = br.read_bits(2);
                            num_passes = 3 + b;
                            if b == 3 {
                                let b = br.read_bits(5);
                                num_passes = 6 + b;
                                if b == 31 {
                                    let b = br.read_bits(7);
                                    num_passes = 37 + b;
                                }
                            }
                        }
                    }

                    // Placeholder passes (HT): fold multiples of 3 back.
                    let num_phld_passes = (num_passes - 1) / 3;
                    let mut missing_msbs = mmsbs + num_phld_passes;
                    let effective_passes = num_passes - num_phld_passes * 3;

                    // Lblock (comma code) then segment lengths.
                    let mut lblock = 3i32;
                    while br.read_bit() != 0 {
                        lblock += 1;
                    }
                    let clz = (num_phld_passes + 1).leading_zeros() as i32;
                    let bits0 = lblock + 31 - clz;
                    let len0 = br.read_bits(bits0);
                    let mut length = len0;
                    if effective_passes > 1 {
                        let bits1 = lblock + if effective_passes > 2 { 1 } else { 0 };
                        let len1 = br.read_bits(bits1);
                        length += len1;
                    }

                    let _ = &mut missing_msbs;
                    packet_cbs.push(Cb {
                        res: res.index,
                        orient: sb.orientation,
                        x,
                        y,
                        len: length,
                        missing_msbs,
                        num_passes: effective_passes,
                    });
                }
            }
        }

        if empty_packet {
            // Already handled (offset advanced) — go to next resolution.
            continue;
        }

        br.terminate();
        let mut body = br.byte_pos();
        for cb in &packet_cbs {
            let off = body;
            body += cb.len as usize;
            code_blocks.push(CodeBlock {
                component,
                resolution: cb.res,
                subband: cb.orient,
                x: cb.x,
                y: cb.y,
                included: true,
                missing_msbs: cb.missing_msbs,
                num_passes: cb.num_passes,
                offset: off as u32,
                length: cb.len,
            });
        }
        if body > tile_end {
            return Err(format!(
                "packet body ({} bytes) overruns tile data (end {})",
                body, tile_end
            ));
        }
        offset = body;
      }
    }

    Ok(PacketParse {
        code_blocks,
        bytes_consumed: (offset - tile_offset) as u32,
    })
}
