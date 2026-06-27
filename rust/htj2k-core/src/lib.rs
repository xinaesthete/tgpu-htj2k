//! htj2k-core: CPU core for an HTJ2K (JPEG 2000 Part 15) codec.
//!
//! This first increment parses the main-header marker segments of a J2K
//! codestream (SOC, SIZ, COD, QCD, CAP) into a `CodestreamInfo`. The HT block
//! decoder is added in a later phase. We only care about the codestream, not
//! the JP2/JPH box wrapper.

use wasm_bindgen::prelude::*;

pub mod geometry;

mod markers {
    pub const SOC: u16 = 0xFF4F;
    pub const SIZ: u16 = 0xFF51;
    pub const COD: u16 = 0xFF52;
    pub const QCD: u16 = 0xFF5C;
    pub const CAP: u16 = 0xFF50;
    pub const SOT: u16 = 0xFF90;
    pub const SOD: u16 = 0xFF93;
}

/// Wavelet kernel used by the (inverse) DWT.
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaveletKernel {
    /// 5/3 reversible integer lifting (lossless).
    Reversible53 = 0,
    /// 9/7 irreversible (lossy).
    Irreversible97 = 1,
}

#[derive(Clone, Copy, Debug)]
struct Component {
    bit_depth: u8,
    signed: bool,
    dx: u8,
    dy: u8,
}

/// Parsed main-header metadata that drives the rest of the decode.
#[wasm_bindgen]
pub struct CodestreamInfo {
    width: u32,
    height: u32,
    x_offset: u32,
    y_offset: u32,
    tile_width: u32,
    tile_height: u32,
    x_tile_offset: u32,
    y_tile_offset: u32,
    components: Vec<Component>,
    num_decompositions: u8,
    code_block_width: u32,
    code_block_height: u32,
    kernel: WaveletKernel,
    progression_order: u8,
    num_layers: u16,
    mct: bool,
    is_htj2k: bool,
    // QCD (quantization defaults)
    quant_style: u8,
    guard_bits: u8,
    subband_exponents: Vec<u8>,
    subband_mantissas: Vec<u16>,
}

/// Cursor over a big-endian byte slice with bounds-checked reads.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn u8(&mut self) -> Result<u8, JsError> {
        let v = *self
            .data
            .get(self.pos)
            .ok_or_else(|| JsError::new("unexpected end of codestream"))?;
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16, JsError> {
        Ok(((self.u8()? as u16) << 8) | self.u8()? as u16)
    }

    fn u32(&mut self) -> Result<u32, JsError> {
        Ok(((self.u16()? as u32) << 16) | self.u16()? as u32)
    }

    fn skip(&mut self, n: usize) -> Result<(), JsError> {
        if n > self.remaining() {
            return Err(JsError::new("marker segment runs past end of codestream"));
        }
        self.pos += n;
        Ok(())
    }
}

#[wasm_bindgen]
impl CodestreamInfo {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width.saturating_sub(self.x_offset)
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height.saturating_sub(self.y_offset)
    }
    #[wasm_bindgen(getter)]
    pub fn tile_width(&self) -> u32 {
        self.tile_width
    }
    #[wasm_bindgen(getter)]
    pub fn tile_height(&self) -> u32 {
        self.tile_height
    }
    #[wasm_bindgen(getter)]
    pub fn num_components(&self) -> u32 {
        self.components.len() as u32
    }
    #[wasm_bindgen(getter)]
    pub fn num_decompositions(&self) -> u8 {
        self.num_decompositions
    }
    #[wasm_bindgen(getter)]
    pub fn code_block_width(&self) -> u32 {
        self.code_block_width
    }
    #[wasm_bindgen(getter)]
    pub fn code_block_height(&self) -> u32 {
        self.code_block_height
    }
    #[wasm_bindgen(getter)]
    pub fn kernel(&self) -> WaveletKernel {
        self.kernel
    }
    #[wasm_bindgen(getter)]
    pub fn reversible(&self) -> bool {
        self.kernel == WaveletKernel::Reversible53
    }
    #[wasm_bindgen(getter)]
    pub fn progression_order(&self) -> u8 {
        self.progression_order
    }
    #[wasm_bindgen(getter)]
    pub fn num_layers(&self) -> u16 {
        self.num_layers
    }
    #[wasm_bindgen(getter)]
    pub fn mct(&self) -> bool {
        self.mct
    }
    #[wasm_bindgen(getter)]
    pub fn is_htj2k(&self) -> bool {
        self.is_htj2k
    }
    /// Quantization style: 0 = none (reversible), 1 = scalar derived, 2 = scalar expounded.
    #[wasm_bindgen(getter)]
    pub fn quant_style(&self) -> u8 {
        self.quant_style
    }
    #[wasm_bindgen(getter)]
    pub fn guard_bits(&self) -> u8 {
        self.guard_bits
    }
    /// Number of subbands described by QCD (1 + 3·decompositions for dyadic).
    #[wasm_bindgen(getter)]
    pub fn num_quant_subbands(&self) -> u32 {
        self.subband_exponents.len() as u32
    }

    /// Quantization exponent (ε) of subband `idx` (0 = LL of the coarsest level).
    pub fn subband_exponent(&self, idx: u32) -> Result<u8, JsError> {
        self.subband_exponents
            .get(idx as usize)
            .copied()
            .ok_or_else(|| JsError::new("subband index out of range"))
    }

    /// Quantization mantissa (μ) of subband `idx` (0 for reversible).
    pub fn subband_mantissa(&self, idx: u32) -> u16 {
        self.subband_mantissas.get(idx as usize).copied().unwrap_or(0)
    }

    /// Number of resolution levels for a component (decompositions + 1).
    #[wasm_bindgen(getter)]
    pub fn num_resolutions(&self) -> u32 {
        self.num_decompositions as u32 + 1
    }

    /// Total number of code-blocks for component `idx` (single-tile geometry).
    pub fn total_code_blocks(&self, idx: u32) -> Result<u32, JsError> {
        let comp = self
            .components
            .get(idx as usize)
            .ok_or_else(|| JsError::new("component index out of range"))?;
        let dx = comp.dx.max(1) as u32;
        let dy = comp.dy.max(1) as u32;
        let layout = geometry::compute_component_layout(
            (self.x_offset / dx) as i64,
            (self.y_offset / dy) as i64,
            (self.width / dx) as i64,
            (self.height / dy) as i64,
            self.num_decompositions as u32,
            self.code_block_width,
            self.code_block_height,
        );
        Ok(layout.total_code_blocks())
    }

    /// Bit depth of component `idx` (0-based).
    pub fn component_bit_depth(&self, idx: u32) -> Result<u8, JsError> {
        self.components
            .get(idx as usize)
            .map(|c| c.bit_depth)
            .ok_or_else(|| JsError::new("component index out of range"))
    }

    /// Whether component `idx` carries signed samples.
    pub fn component_is_signed(&self, idx: u32) -> Result<bool, JsError> {
        self.components
            .get(idx as usize)
            .map(|c| c.signed)
            .ok_or_else(|| JsError::new("component index out of range"))
    }
}

/// Parse the main header of a J2K/HTJ2K codestream.
#[wasm_bindgen]
pub fn parse_codestream(data: &[u8]) -> Result<CodestreamInfo, JsError> {
    let mut r = Reader::new(data);

    if r.u16()? != markers::SOC {
        return Err(JsError::new("codestream does not start with SOC (0xFF4F)"));
    }

    let mut info = CodestreamInfo {
        width: 0,
        height: 0,
        x_offset: 0,
        y_offset: 0,
        tile_width: 0,
        tile_height: 0,
        x_tile_offset: 0,
        y_tile_offset: 0,
        components: Vec::new(),
        num_decompositions: 0,
        code_block_width: 0,
        code_block_height: 0,
        kernel: WaveletKernel::Reversible53,
        progression_order: 0,
        num_layers: 0,
        mct: false,
        is_htj2k: false,
        quant_style: 0,
        guard_bits: 0,
        subband_exponents: Vec::new(),
        subband_mantissas: Vec::new(),
    };
    let mut seen_siz = false;
    let mut seen_cod = false;

    loop {
        let marker = r.u16()?;
        match marker {
            markers::SOT | markers::SOD => break, // reached tile data
            markers::SIZ => {
                let lsiz = r.u16()? as usize;
                let seg_end = r.pos + lsiz - 2;
                let _rsiz = r.u16()?;
                info.width = r.u32()?;
                info.height = r.u32()?;
                info.x_offset = r.u32()?;
                info.y_offset = r.u32()?;
                info.tile_width = r.u32()?;
                info.tile_height = r.u32()?;
                info.x_tile_offset = r.u32()?;
                info.y_tile_offset = r.u32()?;
                let csiz = r.u16()? as usize;
                for _ in 0..csiz {
                    let ssiz = r.u8()?;
                    let dx = r.u8()?;
                    let dy = r.u8()?;
                    info.components.push(Component {
                        bit_depth: (ssiz & 0x7F) + 1,
                        signed: (ssiz & 0x80) != 0,
                        dx,
                        dy,
                    });
                }
                r.pos = seg_end; // tolerate any trailing bytes
                seen_siz = true;
            }
            markers::CAP => {
                let lcap = r.u16()? as usize;
                info.is_htj2k = true; // CAP present => Part 15 capabilities
                r.skip(lcap - 2)?;
            }
            markers::COD => {
                let lcod = r.u16()? as usize;
                let seg_end = r.pos + lcod - 2;
                let scod = r.u8()?;
                // SGcod
                info.progression_order = r.u8()?;
                info.num_layers = r.u16()?;
                info.mct = r.u8()? != 0;
                // SPcod
                info.num_decompositions = r.u8()?;
                let xcb = r.u8()?; // code-block width exponent, value = log2(w) - 2
                let ycb = r.u8()?;
                info.code_block_width = 1u32 << ((xcb & 0x0F) + 2);
                info.code_block_height = 1u32 << ((ycb & 0x0F) + 2);
                let _cb_style = r.u8()?;
                let transform = r.u8()?;
                info.kernel = if transform == 1 {
                    WaveletKernel::Reversible53
                } else {
                    WaveletKernel::Irreversible97
                };
                let _ = scod; // precinct sizes (scod & 1) ignored for now
                r.pos = seg_end;
                seen_cod = true;
            }
            markers::QCD => {
                let lqcd = r.u16()? as usize;
                let seg_end = r.pos + lqcd - 2;
                let sqcd = r.u8()?;
                info.quant_style = sqcd & 0x1F;
                info.guard_bits = sqcd >> 5;
                if info.quant_style == 0 {
                    // No quantization (reversible): one byte per subband, exponent in top 5 bits.
                    while r.pos < seg_end {
                        let b = r.u8()?;
                        info.subband_exponents.push(b >> 3);
                    }
                } else {
                    // Scalar derived (1) or expounded (2): 16 bits per entry (5-bit exp, 11-bit mantissa).
                    while r.pos + 2 <= seg_end {
                        let v = r.u16()?;
                        info.subband_exponents.push((v >> 11) as u8);
                        info.subband_mantissas.push(v & 0x07FF);
                    }
                }
                r.pos = seg_end;
            }
            _ => {
                // Any other marker segment carries a 2-byte length we can skip.
                let len = r.u16()? as usize;
                if len < 2 {
                    return Err(JsError::new("invalid marker segment length"));
                }
                r.skip(len - 2)?;
            }
        }
    }

    if !seen_siz {
        return Err(JsError::new("codestream missing SIZ marker"));
    }
    if !seen_cod {
        return Err(JsError::new("codestream missing COD marker"));
    }
    Ok(info)
}
