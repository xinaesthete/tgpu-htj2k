// GPU inverse 9/7 (irreversible) DWT — float port of the CPU `idwt_level_f32`
// in `rust/htj2k-core/src/decode.rs`. Same thread-per-line strategy and
// descriptor layout as `idwt53.ts`, but f32, with the 9/7 K pre-scale and four
// lifting steps `aug -= a*(l+r)`. The four steps alternate which deinterleaved
// region is `aug` (low/high) by step parity, so no in-WGSL buffer swap is
// needed. The caller converts the normalized output to pixels (`irvToPixels`).
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "./device";

const WGSL = /* wgsl */ `
struct Lvl {
  rw0: u32, rh0: u32, rw1: u32, rh1: u32,
  out_w: u32, out_h: u32, even_x: u32, even_y: u32,
  hl_off: u32, lh_off: u32, hh_off: u32, _pad: u32,
};

const K: f32 = 1.230174104914001;
const STEP0: f32 = 0.443506852043971;
const STEP1: f32 = 0.882911075530934;
const STEP2: f32 = -0.052980118572961;
const STEP3: f32 = -1.586134342059924;

@group(0) @binding(0) var<uniform> L: Lvl;
@group(0) @binding(1) var<storage, read>        inbuf:  array<f32>;
@group(0) @binding(2) var<storage, read>        coeffs: array<f32>;
@group(0) @binding(3) var<storage, read_write>  hbuf:   array<f32>;
@group(0) @binding(4) var<storage, read_write>  outbuf: array<f32>;
@group(0) @binding(5) var<storage, read_write>  scratch: array<f32>;

fn clampr(j: i32, n: u32) -> u32 {
  if (j < 0) { return 0u; }
  if (u32(j) >= n) { return n - 1u; }
  return u32(j);
}

fn outw(out_base: u32, out_stride: u32, p: u32, v: f32) {
  let sel = out_stride >> 31u;
  let stride = out_stride & 0x7fffffffu;
  let idx = out_base + p * stride;
  if (sel == 0u) { hbuf[idx] = v; } else { outbuf[idx] = v; }
}

fn step_coeff(j: u32) -> f32 {
  if (j == 0u) { return STEP0; }
  if (j == 1u) { return STEP1; }
  if (j == 2u) { return STEP2; }
  return STEP3;
}

// In-place inverse 9/7 lifting on scratch[base..base+nl+nh]:
// scratch[base..base+nl)        = low, scratch[base+nl..)= high.
fn lift(base: u32, nl: u32, nh: u32, even: u32) {
  let lo = base;
  let hi = base + nl;
  // K pre-scale: low *= K, high *= 1/K.
  for (var i: u32 = 0u; i < nl; i = i + 1u) { scratch[lo + i] = scratch[lo + i] * K; }
  for (var i: u32 = 0u; i < nh; i = i + 1u) { scratch[hi + i] = scratch[hi + i] * (1.0 / K); }
  // Four lifting steps; even-index steps act on low (aug), odd on high.
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    let a = step_coeff(j);
    let aug_low = (j & 1u) == 0u;
    let aug_base = select(hi, lo, aug_low);
    let oth_base = select(lo, hi, aug_low);
    let aug_n = select(nh, nl, aug_low);
    let oth_n = select(nl, nh, aug_low);
    var ev: u32 = even; if (!aug_low) { ev = 1u - even; }
    let off: i32 = select(1i, 0i, ev == 1u);
    for (var i: u32 = 0u; i < aug_n; i = i + 1u) {
      let l = scratch[oth_base + clampr(i32(i) + off - 1, oth_n)];
      let r = scratch[oth_base + clampr(i32(i) + off, oth_n)];
      scratch[aug_base + i] = scratch[aug_base + i] - a * (l + r);
    }
  }
}

fn interleave(base: u32, nl: u32, nh: u32, even: u32, out_base: u32, out_stride: u32) {
  let lo = base; let hi = base + nl;
  let width = nl + nh;
  var dp: u32 = 0u; var li: u32 = 0u; var hii: u32 = 0u;
  if (even == 0u) {
    outw(out_base, out_stride, 0u, scratch[hi]); dp = 1u; hii = 1u;
  }
  loop {
    if (dp + 1u >= width) { break; }
    outw(out_base, out_stride, dp,      scratch[lo + li]);
    outw(out_base, out_stride, dp + 1u, scratch[hi + hii]);
    dp = dp + 2u; li = li + 1u; hii = hii + 1u;
  }
  if (dp < width) { outw(out_base, out_stride, dp, scratch[lo + li]); }
}

@compute @workgroup_size(64)
fn idwt_h(@builtin(global_invocation_id) gid: vec3u) {
  let ry = gid.x;
  if (ry >= L.rh0 + L.rh1) { return; }
  let nl = L.rw0; let nh = L.rw1; let width = L.out_w;
  let base = ry * L.out_w;
  let a_size = L.rh0 * L.out_w;
  var low_is_inbuf = false;
  var low_off: u32 = 0u; var high_off: u32 = 0u; var srow: u32 = 0u;
  if (ry < L.rh0) {
    low_is_inbuf = true; srow = ry;
    low_off = ry * L.rw0; high_off = L.hl_off + ry * L.rw1;
  } else {
    srow = ry - L.rh0;
    low_off = L.lh_off + srow * L.rw0; high_off = L.hh_off + srow * L.rw1;
  }
  for (var i: u32 = 0u; i < nl; i = i + 1u) {
    scratch[base + i] = select(coeffs[low_off + i], inbuf[low_off + i], low_is_inbuf);
  }
  for (var i: u32 = 0u; i < nh; i = i + 1u) {
    scratch[base + nl + i] = coeffs[high_off + i];
  }
  let out_base = select(a_size + srow * L.out_w, ry * L.out_w, ry < L.rh0);
  if (width == 1u) {
    hbuf[out_base] = select(scratch[base + nl] * 0.5, scratch[base], L.even_x == 1u);
    return;
  }
  lift(base, nl, nh, L.even_x);
  interleave(base, nl, nh, L.even_x, out_base, 1u);
}

@compute @workgroup_size(64)
fn idwt_v(@builtin(global_invocation_id) gid: vec3u) {
  let cx = gid.x;
  if (cx >= L.out_w) { return; }
  let nl = L.rh0; let nh = L.rh1; let height = L.out_h;
  let base = cx * L.out_h;
  let a_size = L.rh0 * L.out_w;
  for (var y: u32 = 0u; y < nl; y = y + 1u) {
    scratch[base + y] = hbuf[y * L.out_w + cx];
  }
  for (var y: u32 = 0u; y < nh; y = y + 1u) {
    scratch[base + nl + y] = hbuf[a_size + y * L.out_w + cx];
  }
  if (height == 1u) {
    outbuf[cx] = select(scratch[base + nl] * 0.5, scratch[base], L.even_y == 1u);
    return;
  }
  lift(base, nl, nh, L.even_y);
  interleave(base, nl, nh, L.even_y, cx, L.out_w | 0x80000000u);
}
`;

export interface DwtInput97 {
  descriptor: Uint32Array;
  coeffs: Float32Array;
  width: number;
  height: number;
}

interface Pipe {
  device: GPUDevice;
  bgl: GPUBindGroupLayout;
  pipeH: GPUComputePipeline;
  pipeV: GPUComputePipeline;
  paramsBuf: GPUBuffer;
}
let pipeCache: Promise<Pipe> | undefined;

async function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const module = device.createShaderModule({ code: WGSL });
    const storage = (i: number): GPUBindGroupLayoutEntry => ({
      binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const },
    });
    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" as const } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
        storage(3), storage(4), storage(5),
      ],
    });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const pipeH = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "idwt_h" } });
    const pipeV = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "idwt_v" } });
    const paramsBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    return { device, bgl, pipeH, pipeV, paramsBuf };
  })();
  return pipeCache;
}

/** Run the inverse 9/7 DWT on the GPU. Returns normalized float samples
 *  (row-major `width * height`); apply `irvToPixels` for the level shift. */
export async function idwt97Gpu(input: DwtInput97): Promise<Float32Array> {
  const { device, bgl, pipeH, pipeV, paramsBuf } = await getPipe();
  const root = tgpu.initFromDevice({ device });
  const { descriptor: desc, coeffs, width, height } = input;
  const at = (i: number): number => desc[i]!;
  const nLevels = at(1);
  const imgN = width * height;

  const ll0w = at(2), ll0h = at(3), ll0off = at(4);
  const coeffBuf = root.createBuffer(d.arrayOf(d.f32, Math.max(1, coeffs.length)), Array.from(coeffs)).$usage("storage");
  let inbuf = root.createBuffer(d.arrayOf(d.f32, Math.max(1, imgN))).$usage("storage");
  let outbuf = root.createBuffer(d.arrayOf(d.f32, Math.max(1, imgN))).$usage("storage");
  const hbuf = root.createBuffer(d.arrayOf(d.f32, Math.max(1, imgN))).$usage("storage");
  const scratch = root.createBuffer(d.arrayOf(d.f32, Math.max(1, imgN))).$usage("storage");
  const owned = [coeffBuf, inbuf, outbuf, hbuf, scratch];

  inbuf.write(Array.from(coeffs.subarray(ll0off, ll0off + ll0w * ll0h)).concat(
    new Array(Math.max(0, imgN - ll0w * ll0h)).fill(0)));

  for (let lvl = 0; lvl < nLevels; lvl++) {
    const o = 5 + lvl * 12;
    const rec = desc.subarray(o, o + 12);
    const outW = at(o + 4);
    const rowCount = at(o + 1) + at(o + 3);
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array(rec));

    const gpuIn = root.unwrap(inbuf), gpuOut = root.unwrap(outbuf);
    const bind = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: gpuIn } },
        { binding: 2, resource: { buffer: root.unwrap(coeffBuf) } },
        { binding: 3, resource: { buffer: root.unwrap(hbuf) } },
        { binding: 4, resource: { buffer: gpuOut } },
        { binding: 5, resource: { buffer: root.unwrap(scratch) } },
      ],
    });

    const enc = device.createCommandEncoder();
    const ph = enc.beginComputePass();
    ph.setPipeline(pipeH); ph.setBindGroup(0, bind);
    ph.dispatchWorkgroups(Math.ceil(rowCount / 64));
    ph.end();
    const pv = enc.beginComputePass();
    pv.setPipeline(pipeV); pv.setBindGroup(0, bind);
    pv.dispatchWorkgroups(Math.ceil(outW / 64));
    pv.end();
    device.queue.submit([enc.finish()]);

    [inbuf, outbuf] = [outbuf, inbuf];
  }

  const result = (await inbuf.read()) as ArrayLike<number>;
  const out = Float32Array.from(result).slice(0, imgN);
  for (const b of owned) b.destroy();
  return out;
}

/** Convert normalized 9/7 float samples to integer pixels — JS port of
 *  `irv_to_pixels` (`lib.rs`): scale by 2^bit_depth, round half-away-from-zero,
 *  clamp, level shift. `Math.fround` mirrors the Rust f32 multiply. */
export function irvToPixels(coeffs: Float32Array, bitDepth: number, signed: boolean): Int32Array {
  const mul = 2 ** bitDepth;
  const up = (0x7fffffff >> (32 - bitDepth));
  const low = -(up + 1);
  const flUp = -low, flLow = low;
  const half = 1 << (bitDepth - 1);
  const out = new Int32Array(coeffs.length);
  for (let i = 0; i < coeffs.length; i++) {
    const t = Math.fround(coeffs[i]! * mul);
    let v = Math.trunc(t + (t >= 0 ? 0.5 : -0.5));
    if (!(t >= flLow)) v = low;
    if (!(t < flUp)) v = up;
    out[i] = signed ? v : v + half;
  }
  return out;
}
