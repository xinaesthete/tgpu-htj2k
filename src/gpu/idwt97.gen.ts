// Bootstrapped by tgpu-gen from the inverse-9/7 WGSL, then curated to the typed
// resource interface (uniform struct + f32 bind group layout). The compute
// logic lives in `idwt97.ts` as a WGSL template resolved against this.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

/* structs */
export const Lvl = d.struct({
  rw0: d.u32,
  rh0: d.u32,
  rw1: d.u32,
  rh1: d.u32,
  out_w: d.u32,
  out_h: d.u32,
  even_x: d.u32,
  even_y: d.u32,
  hl_off: d.u32,
  lh_off: d.u32,
  hh_off: d.u32,
  _pad: d.u32,
});

/* bindGroupLayouts */
export const layout0 = tgpu.bindGroupLayout({
  L: {
    uniform: Lvl,
  },
  inbuf: {
    storage: (arrayLength: number) => d.arrayOf(d.f32, arrayLength),
    access: 'readonly',
  },
  coeffs: {
    storage: (arrayLength: number) => d.arrayOf(d.f32, arrayLength),
    access: 'readonly',
  },
  hbuf: {
    storage: (arrayLength: number) => d.arrayOf(d.f32, arrayLength),
    access: 'mutable',
  },
  outbuf: {
    storage: (arrayLength: number) => d.arrayOf(d.f32, arrayLength),
    access: 'mutable',
  },
  // (global `scratch` dropped — the optimised kernel lifts in workgroup shared
  // memory; see idwt97.ts)
});
