// GPU forward 9/7 (irreversible) DWT — float analysis, the float half of the
// transform pair (with idwt97.ts). Mirror of fdwt53.ts but float, with the four
// 9/7 lifting steps run in reverse with `aug += a*(l+r)` and the K post-scale
// (low /= K, high *= K) — the exact inverse of the synthesis. One workgroup per
// line, deinterleave on load, lift in workgroup shared memory, vertical-then-
// horizontal split. Per-level submits (cross-level ping-pong needs the barrier).
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "./device";
import { Lvl } from "./idwt97.gen";

const WG = 64;
const MAXLINE = 2048;

const fwdLayout = tgpu.bindGroupLayout({
  L: { uniform: Lvl },
  inbuf: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  hbuf: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  outbuf: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  coeffs: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const TEMPLATE = /* wgsl */ `
const K: f32 = 1.230174104914001;
const STEP0: f32 = 0.443506852043971;
const STEP1: f32 = 0.882911075530934;
const STEP2: f32 = -0.052980118572961;
const STEP3: f32 = -1.586134342059924;

fn clampr(j: i32, n: u32) -> u32 {
  if (j < 0) { return 0u; }
  if (u32(j) >= n) { return n - 1u; }
  return u32(j);
}
fn step_coeff(j: u32) -> f32 {
  if (j == 0u) { return STEP0; }
  if (j == 1u) { return STEP1; }
  if (j == 2u) { return STEP2; }
  return STEP3;
}

var<workgroup> sh: array<f32, ${MAXLINE}u>;  // sh[0..nl) low, sh[nl..nl+nh) high

// Forward 9/7 analysis on the deinterleaved shared line: four lifting steps in
// reverse with aug += a*(l+r), then undo the K pre-scale.
fn fwd_lift(t: u32, nl: u32, nh: u32, even: u32) {
  for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
    let j = 3u - jj;
    let a = step_coeff(j);
    let aug_low = (j & 1u) == 0u;
    let aug_off = select(nl, 0u, aug_low);
    let oth_off = select(0u, nl, aug_low);
    let aug_n = select(nh, nl, aug_low);
    let oth_n = select(nl, nh, aug_low);
    var ev: u32 = even; if (!aug_low) { ev = 1u - even; }
    let off: i32 = select(1i, 0i, ev == 1u);
    for (var i: u32 = t; i < aug_n; i = i + ${WG}u) {
      let l = sh[oth_off + clampr(i32(i) + off - 1, oth_n)];
      let r = sh[oth_off + clampr(i32(i) + off, oth_n)];
      sh[aug_off + i] = sh[aug_off + i] + a * (l + r);
    }
    workgroupBarrier();
  }
  let k_inv = 1.0 / K;
  for (var i: u32 = t; i < nl; i = i + ${WG}u) { sh[i] = sh[i] * k_inv; }
  for (var i: u32 = t; i < nh; i = i + ${WG}u) { sh[nl + i] = sh[nl + i] * K; }
  workgroupBarrier();
}

@compute @workgroup_size(${WG})
fn fwd_v(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let cx = wid.x;
  if (cx >= L.out_w) { return; }
  let t = lid.x;
  let nl = L.rh0; let nh = L.rh1; let height = L.out_h;
  let a_size = L.rh0 * L.out_w;
  if (height == 1u) {
    if (t == 0u) { let v = inbuf[cx]; hbuf[cx] = select(v * 2.0, v, L.even_y == 1u); }
    return;
  }
  let low_phase = select(1u, 0u, L.even_y == 1u);
  for (var p: u32 = t; p < height; p = p + ${WG}u) {
    let v = inbuf[p * L.out_w + cx];
    let idx = p >> 1u;
    if ((p & 1u) == low_phase) { sh[idx] = v; } else { sh[nl + idx] = v; }
  }
  workgroupBarrier();
  fwd_lift(t, nl, nh, L.even_y);
  for (var y: u32 = t; y < nl; y = y + ${WG}u) { hbuf[y * L.out_w + cx] = sh[y]; }
  for (var y: u32 = t; y < nh; y = y + ${WG}u) { hbuf[a_size + y * L.out_w + cx] = sh[nl + y]; }
}

@compute @workgroup_size(${WG})
fn fwd_h(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let ry = wid.x;
  if (ry >= L.rh0 + L.rh1) { return; }
  let t = lid.x;
  let nl = L.rw0; let nh = L.rw1; let width = L.out_w;
  let is_a = ry < L.rh0;
  let srow = select(ry - L.rh0, ry, is_a);
  let row_base = ry * L.out_w;
  if (width == 1u) {
    if (t == 0u) {
      let v = hbuf[row_base];
      if (L.even_x == 1u) {
        if (is_a) { outbuf[ry * L.rw0] = v; } else { coeffs[L.lh_off + srow * L.rw0] = v; }
      } else {
        let hgh = v * 2.0;
        if (is_a) { coeffs[L.hl_off + ry * L.rw1] = hgh; } else { coeffs[L.hh_off + srow * L.rw1] = hgh; }
      }
    }
    return;
  }
  let low_phase = select(1u, 0u, L.even_x == 1u);
  for (var p: u32 = t; p < width; p = p + ${WG}u) {
    let v = hbuf[row_base + p];
    let idx = p >> 1u;
    if ((p & 1u) == low_phase) { sh[idx] = v; } else { sh[nl + idx] = v; }
  }
  workgroupBarrier();
  fwd_lift(t, nl, nh, L.even_x);
  for (var x: u32 = t; x < nl; x = x + ${WG}u) {
    let lv = sh[x];
    if (is_a) { outbuf[ry * L.rw0 + x] = lv; } else { coeffs[L.lh_off + srow * L.rw0 + x] = lv; }
  }
  for (var x: u32 = t; x < nh; x = x + ${WG}u) {
    let hv = sh[nl + x];
    if (is_a) { coeffs[L.hl_off + ry * L.rw1 + x] = hv; } else { coeffs[L.hh_off + srow * L.rw1 + x] = hv; }
  }
}
`;

export interface FwdInput97 {
  descriptor: Uint32Array;
  image: Float32Array;
  width: number;
  height: number;
  coeffsLen: number;
}

interface Pipe {
  root: ReturnType<typeof tgpu.initFromDevice>;
  device: GPUDevice;
  pipeV: GPUComputePipeline;
  pipeH: GPUComputePipeline;
}
let pipeCache: Promise<Pipe> | undefined;

async function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({
      template: TEMPLATE, externals: { ...fwdLayout.bound }, names: "strict",
    });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeV = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "fwd_v" } });
    const pipeH = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "fwd_h" } });
    return { root, device, pipeV, pipeH };
  })();
  return pipeCache;
}

type Root = Awaited<ReturnType<typeof getPipe>>["root"];
function makePool(root: Root, cap: number, ccap: number) {
  const A = (n: number) => d.arrayOf(d.f32, Math.max(1, n));
  return {
    cap, ccap,
    bufA: root.createBuffer(A(cap)).$usage("storage"),
    bufB: root.createBuffer(A(cap)).$usage("storage"),
    hbuf: root.createBuffer(A(cap)).$usage("storage"),
    coeffBuf: root.createBuffer(A(ccap)).$usage("storage"),
    lvlBuf: root.createBuffer(Lvl).$usage("uniform"),
  };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, imgN: number, coeffLen: number) {
  if (pool && pool.cap >= imgN && pool.ccap >= coeffLen) return pool;
  const cap = Math.max(imgN, pool?.cap ?? 0);
  const ccap = Math.max(coeffLen, pool?.ccap ?? 0);
  if (pool) for (const b of [pool.bufA, pool.bufB, pool.hbuf, pool.coeffBuf, pool.lvlBuf]) b.destroy();
  pool = makePool(root, cap, ccap);
  return pool;
}

export interface Fdwt97Opts {
  readback?: boolean;
}

/** Run the forward 9/7 DWT on the GPU. Returns the packed float coefficients,
 *  or null when `opts.readback === false`. */
export async function fdwt97Gpu(input: FwdInput97, opts: Fdwt97Opts = {}): Promise<Float32Array | null> {
  const { root, device, pipeV, pipeH } = await getPipe();
  const { descriptor: desc, image, width, height, coeffsLen } = input;
  const at = (i: number): number => desc[i]!;
  const nLevels = at(1);
  const imgN = width * height;

  const p = ensurePool(root, imgN, coeffsLen);
  device.queue.writeBuffer(root.unwrap(p.bufA), 0, image as BufferSource);

  let inbuf = p.bufA, outbuf = p.bufB;
  for (let lvl = nLevels - 1; lvl >= 0; lvl--) {
    const o = 5 + lvl * 12;
    const outW = at(o + 4);
    const rowCount = at(o + 1) + at(o + 3);
    p.lvlBuf.write({
      rw0: at(o), rh0: at(o + 1), rw1: at(o + 2), rh1: at(o + 3),
      out_w: at(o + 4), out_h: at(o + 5), even_x: at(o + 6), even_y: at(o + 7),
      hl_off: at(o + 8), lh_off: at(o + 9), hh_off: at(o + 10), _pad: 0,
    });
    const bind = root.unwrap(root.createBindGroup(fwdLayout, {
      L: p.lvlBuf, inbuf, hbuf: p.hbuf, outbuf, coeffs: p.coeffBuf,
    }));
    const enc = device.createCommandEncoder();
    const pv = enc.beginComputePass();
    pv.setPipeline(pipeV); pv.setBindGroup(0, bind);
    pv.dispatchWorkgroups(outW);
    pv.end();
    const ph = enc.beginComputePass();
    ph.setPipeline(pipeH); ph.setBindGroup(0, bind);
    ph.dispatchWorkgroups(rowCount);
    ph.end();
    device.queue.submit([enc.finish()]);
    [inbuf, outbuf] = [outbuf, inbuf];
  }
  const ll0N = at(2) * at(3);
  const cenc = device.createCommandEncoder();
  cenc.copyBufferToBuffer(root.unwrap(inbuf), 0, root.unwrap(p.coeffBuf), 0, ll0N * 4);
  device.queue.submit([cenc.finish()]);

  if (opts.readback === false) {
    await device.queue.onSubmittedWorkDone();
    return null;
  }
  const result = (await p.coeffBuf.read()) as ArrayLike<number>;
  return Float32Array.from(result).slice(0, coeffsLen);
}
