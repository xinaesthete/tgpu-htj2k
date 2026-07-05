// GPU forward 5/3 (reversible) DWT — the analysis/encode direction, and a
// reusable transform primitive (pairs with the inverse in idwt53.ts). It
// decomposes a level-shifted image into LL + per-level (HL, LH, HH) subbands in
// the same packed layout the inverse consumes, so forward → inverse is identity.
//
// Mirror of the inverse, reversed: per level, **vertical analysis then
// horizontal** (the inverse does horizontal-combine then vertical-combine).
// Each 1D step deinterleaves the line on load, then runs predict and update in
// workgroup shared memory with intra-line parallelism (barriers between steps).
// One workgroup per line; i32 throughout, so it is bit-exact vs the CPU forward.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "./device";
import { Lvl } from "./idwt53.gen";

const WG = 64;
const MAXLINE = 2048;

// Forward bind group: LL in (inbuf), V→H intermediate (hbuf), LL out (outbuf),
// detail out (coeffs). inbuf/outbuf ping-pong across levels.
const fwdLayout = tgpu.bindGroupLayout({
  L: { uniform: Lvl },
  inbuf: { storage: (n: number) => d.arrayOf(d.i32, n), access: "readonly" },
  hbuf: { storage: (n: number) => d.arrayOf(d.i32, n), access: "mutable" },
  outbuf: { storage: (n: number) => d.arrayOf(d.i32, n), access: "mutable" },
  coeffs: { storage: (n: number) => d.arrayOf(d.i32, n), access: "mutable" },
});

const TEMPLATE = /* wgsl */ `
fn clampr(j: i32, n: u32) -> u32 {
  if (j < 0) { return 0u; }
  if (u32(j) >= n) { return n - 1u; }
  return u32(j);
}

var<workgroup> sh: array<i32, ${MAXLINE}u>;  // sh[0..nl) low, sh[nl..nl+nh) high

// Forward 5/3 analysis lifting on the deinterleaved shared line: predict (on
// high, reading low) then update (on low, reading high) — reverse of synthesis.
fn fwd_lift(t: u32, nl: u32, nh: u32, even: u32) {
  let offp: i32 = select(0i, 1i, even == 1u);   // even ? 1 : 0
  for (var i: u32 = t; i < nh; i = i + ${WG}u) {
    let a = sh[clampr(i32(i) + offp - 1, nl)];
    let b = sh[clampr(i32(i) + offp, nl)];
    sh[nl + i] = sh[nl + i] - ((a + b) >> 1u);
  }
  workgroupBarrier();
  let offu: i32 = select(1i, 0i, even == 1u);   // even ? 0 : 1
  for (var i: u32 = t; i < nl; i = i + ${WG}u) {
    let a = sh[nl + clampr(i32(i) + offu - 1, nh)];
    let b = sh[nl + clampr(i32(i) + offu, nh)];
    sh[i] = sh[i] + ((2 + a + b) >> 2u);
  }
  workgroupBarrier();
}

// ---- Vertical analysis: one workgroup per column → low rows (A) + high (B) ----
@compute @workgroup_size(${WG})
fn fwd_v(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let cx = wid.x;
  if (cx >= L.out_w) { return; }
  let t = lid.x;
  let nl = L.rh0; let nh = L.rh1; let height = L.out_h;
  let a_size = L.rh0 * L.out_w;
  if (height == 1u) {
    if (t == 0u) { let v = inbuf[cx]; hbuf[cx] = select(v << 1u, v, L.even_y == 1u); }
    return;
  }
  // Deinterleave the column into sh: low at parity-even positions.
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

// ---- Horizontal analysis: one workgroup per hbuf row → LL/HL (A) or LH/HH (B) ----
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
      if (L.even_x == 1u) {            // single low sample → LL (A) or LH (B)
        if (is_a) { outbuf[ry * L.rw0] = v; } else { coeffs[L.lh_off + srow * L.rw0] = v; }
      } else {                          // single high sample (×2) → HL (A) or HH (B)
        let hgh = v << 1u;
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
  // low → LL (A rows) or LH (B rows); high → HL (A) or HH (B)
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

export interface FwdInput53 {
  /** Geometry descriptor (from dwt_forward_53 / dwt_input_53). */
  descriptor: Uint32Array;
  /** Level-shifted image, row-major width*height. */
  image: Int32Array;
  width: number;
  height: number;
  /** Total length of the packed coefficient output. */
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
      template: TEMPLATE,
      externals: { ...fwdLayout.bound },
      names: "strict",
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
  const A = (n: number) => d.arrayOf(d.i32, Math.max(1, n));
  return {
    cap,
    ccap,
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

export interface Fdwt53Opts {
  /** When false, skip the readback and just sync (keep-on-GPU). Returns null. */
  readback?: boolean;
}

/** Run the forward 5/3 DWT on the GPU. Returns the packed coefficients
 *  (row-major per band, at the descriptor offsets), or null when
 *  `opts.readback === false`. */
export async function fdwt53Gpu(input: FwdInput53, opts: Fdwt53Opts = {}): Promise<Int32Array | null> {
  const { root, device, pipeV, pipeH } = await getPipe();
  const { descriptor: desc, image, width, height, coeffsLen } = input;
  const at = (i: number): number => desc[i]!;
  const nLevels = at(1);
  const imgN = width * height;

  const p = ensurePool(root, imgN, coeffsLen);
  device.queue.writeBuffer(root.unwrap(p.bufA), 0, image as BufferSource);

  let inbuf = p.bufA,
    outbuf = p.bufB;
  // Finest level first (descriptor records are ordered coarse→fine, r=1..N).
  for (let lvl = nLevels - 1; lvl >= 0; lvl--) {
    const o = 5 + lvl * 12;
    const outW = at(o + 4);
    const rowCount = at(o + 1) + at(o + 3); // rh0 + rh1
    p.lvlBuf.write({
      rw0: at(o),
      rh0: at(o + 1),
      rw1: at(o + 2),
      rh1: at(o + 3),
      out_w: at(o + 4),
      out_h: at(o + 5),
      even_x: at(o + 6),
      even_y: at(o + 7),
      hl_off: at(o + 8),
      lh_off: at(o + 9),
      hh_off: at(o + 10),
      shift: 0,
    });
    const bind = root.unwrap(
      root.createBindGroup(fwdLayout, {
        L: p.lvlBuf,
        inbuf,
        hbuf: p.hbuf,
        outbuf,
        coeffs: p.coeffBuf,
      }),
    );
    const enc = device.createCommandEncoder();
    const pv = enc.beginComputePass();
    pv.setPipeline(pipeV);
    pv.setBindGroup(0, bind);
    pv.dispatchWorkgroups(outW);
    pv.end();
    const ph = enc.beginComputePass();
    ph.setPipeline(pipeH);
    ph.setBindGroup(0, bind);
    ph.dispatchWorkgroups(rowCount);
    ph.end();
    device.queue.submit([enc.finish()]);
    [inbuf, outbuf] = [outbuf, inbuf]; // the LL we just wrote feeds the next level
  }
  // After the coarsest level, `inbuf` holds ll0 → copy to coeffs[0..ll0N).
  const ll0N = at(2) * at(3);
  const cenc = device.createCommandEncoder();
  cenc.copyBufferToBuffer(root.unwrap(inbuf), 0, root.unwrap(p.coeffBuf), 0, ll0N * 4);
  device.queue.submit([cenc.finish()]);

  if (opts.readback === false) {
    await device.queue.onSubmittedWorkDone();
    return null;
  }
  const result = (await p.coeffBuf.read()) as ArrayLike<number>;
  return Int32Array.from(result).slice(0, coeffsLen);
}
