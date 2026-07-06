// GPU inverse 5/3 (reversible) DWT — the TypeGPU/WebGPU port of the CPU
// `idwt_level` in `rust/htj2k-core/src/decode.rs`, used as the golden reference.
//
// Strategy: one workgroup per DWT line, lifting in workgroup shared memory with
// intra-line parallelism (see the kernel comment below). The 5/3 lifting runs in
// i32 (WGSL `>>` on i32 is an arithmetic shift), so the result is bit-exact vs
// the CPU. The vertical pass folds in the DC level shift, so the GPU emits
// display-ready pixels (no CPU post-pass).
//
// Per level: horizontal pass (LL+HL → A low-vertical rows, LH+HH → B
// high-vertical rows, packed in `hbuf`), then vertical pass (A+B per column →
// output). Levels run coarse→fine in a JS loop, ping-ponging `inbuf`/`outbuf`.
//
// The uniform struct (`Lvl`) and bind group layout (`layout0`) are typed
// TypeGPU resources generated from the WGSL by tgpu-gen (see `idwt53.gen.ts`);
// the compute logic below references them as externals and is stitched together
// by `tgpu.resolveWithContext`, so the struct/binding declarations live only in
// TypeScript (single source of truth, type-checked buffers and bind groups).
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "./device";
import { Lvl, layout0 } from "./idwt53.gen";

// Compute logic only — `L`, `inbuf`, `coeffs`, `hbuf`, `outbuf` and the `Lvl`
// struct are injected by tgpu.resolveWithContext from `layout0`.
//
// Optimised kernel: **one workgroup per DWT line**, lifting in **workgroup
// shared memory** (`sh`) rather than a global scratch buffer. The threads of a
// workgroup cooperatively load the line, then run each lifting step in parallel
// — every update within a step is independent; the dependency is only between
// steps, handled by `workgroupBarrier()`. This removes the per-element global
// memory traffic (and the global scratch buffer) of the previous
// one-thread-per-line version. Lines must fit in `MAXLINE` (shared memory).
const WG = 64;
const MAXLINE = 2048; // max samples/line; 2048*4 = 8 KiB shared mem per workgroup
const TEMPLATE = /* wgsl */ `
fn clampr(j: i32, n: u32) -> u32 {
  if (j < 0) { return 0u; }
  if (u32(j) >= n) { return n - 1u; }
  return u32(j);
}

var<workgroup> sh: array<i32, ${MAXLINE}u>;  // sh[0..nl) low, sh[nl..nl+nh) high

// Steps 0 (update low) and 1 (predict high) of the inverse 5/3, run in parallel
// across the line by the t-strided threads, with a barrier between them.
fn lift(t: u32, nl: u32, nh: u32, even: u32) {
  let off0: i32 = select(1i, 0i, even == 1u);   // ev = even at step 0
  let off1: i32 = select(0i, 1i, even == 1u);   // ev = !even at step 1
  for (var i: u32 = t; i < nl; i = i + ${WG}u) {
    let a = sh[nl + clampr(i32(i) + off0 - 1, nh)];
    let b = sh[nl + clampr(i32(i) + off0, nh)];
    sh[i] = sh[i] - ((2 + a + b) >> 2u);
  }
  workgroupBarrier();
  for (var i: u32 = t; i < nh; i = i + ${WG}u) {
    let a = sh[clampr(i32(i) + off1 - 1, nl)];
    let b = sh[clampr(i32(i) + off1, nl)];
    sh[nl + i] = sh[nl + i] + ((a + b) >> 1u);
  }
  workgroupBarrier();
}

// Interleave shared low/high into out[out_base + p*out_stride], parallel over p.
// sel: 0 -> hbuf, 1 -> outbuf. shift (DC level shift) is added on the final
// output so the GPU emits display-ready pixels; 0 for intermediate writes.
fn scatter(t: u32, nl: u32, nh: u32, even: u32, out_base: u32, out_stride: u32, sel: u32, shift: i32) {
  let width = nl + nh;
  let low_phase = select(1u, 0u, even == 1u); // low samples sit at this output parity
  for (var p: u32 = t; p < width; p = p + ${WG}u) {
    let idx = p >> 1u;
    var v: i32;
    if ((p & 1u) == low_phase) { v = sh[idx]; } else { v = sh[nl + idx]; }
    let o = out_base + p * out_stride;
    if (sel == 0u) { hbuf[o] = v; } else { outbuf[o] = v + shift; }
  }
}

// ---- Horizontal pass: one workgroup per row (A rows then B rows) ----
@compute @workgroup_size(${WG})
fn idwt_h(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let ry = wid.x;
  if (ry >= L.rh0 + L.rh1) { return; } // uniform per workgroup
  let t = lid.x;
  let nl = L.rw0; let nh = L.rw1; let width = L.out_w;
  let a_size = L.rh0 * L.out_w;        // hbuf offset where B rows begin
  var low_is_inbuf = false;
  var low_off: u32 = 0u; var high_off: u32 = 0u; var srow: u32 = 0u;
  if (ry < L.rh0) {
    low_is_inbuf = true; srow = ry;
    low_off = ry * L.rw0; high_off = L.hl_off + ry * L.rw1;
  } else {
    srow = ry - L.rh0;
    low_off = L.lh_off + srow * L.rw0; high_off = L.hh_off + srow * L.rw1;
  }
  for (var i: u32 = t; i < nl; i = i + ${WG}u) {
    sh[i] = select(coeffs[low_off + i], inbuf[low_off + i], low_is_inbuf);
  }
  for (var i: u32 = t; i < nh; i = i + ${WG}u) {
    sh[nl + i] = coeffs[high_off + i];
  }
  workgroupBarrier();
  let out_base = select(a_size + srow * L.out_w, ry * L.out_w, ry < L.rh0);
  if (width == 1u) {
    if (t == 0u) { hbuf[out_base] = select(sh[nl], sh[0], L.even_x == 1u); }
    return;
  }
  lift(t, nl, nh, L.even_x);
  scatter(t, nl, nh, L.even_x, out_base, 1u, 0u, 0i); // hbuf is intermediate: no shift
}

// ---- Vertical pass: one workgroup per output column ----
@compute @workgroup_size(${WG})
fn idwt_v(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let cx = wid.x;
  if (cx >= L.out_w) { return; } // uniform per workgroup
  let t = lid.x;
  let nl = L.rh0; let nh = L.rh1; let height = L.out_h;
  let a_size = L.rh0 * L.out_w;
  for (var y: u32 = t; y < nl; y = y + ${WG}u) {
    sh[y] = hbuf[y * L.out_w + cx];
  }
  for (var y: u32 = t; y < nh; y = y + ${WG}u) {
    sh[nl + y] = hbuf[a_size + y * L.out_w + cx];
  }
  workgroupBarrier();
  if (height == 1u) {
    if (t == 0u) { outbuf[cx] = select(sh[nl], sh[0], L.even_y == 1u) + i32(L.shift); }
    return;
  }
  lift(t, nl, nh, L.even_y);
  scatter(t, nl, nh, L.even_y, cx, L.out_w, 1u, i32(L.shift)); // final output: + level shift
}
`;

export interface DwtInput53 {
  descriptor: Uint32Array;
  coeffs: Int32Array;
  width: number;
  height: number;
  /** DC level shift to add on the final level (the GPU emits display-ready
   *  pixels). 0 for signed components. Defaults to 0. */
  shift?: number;
}

// Pipeline state is expensive to build and immutable across calls, so cache it
// per device. (Re-creating modules/pipelines every call also churns the Dawn
// native addon, which destabilises its process-exit teardown under Node.)
interface Pipe {
  root: ReturnType<typeof tgpu.initFromDevice>;
  device: GPUDevice;
  pipeH: GPUComputePipeline;
  pipeV: GPUComputePipeline;
}
let pipeCache: Promise<Pipe> | undefined;

async function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    // Stitch the compute logic together with the generated struct + bind group
    // layout; `code` carries the injected declarations, and `usedBindGroupLayouts`
    // gives the layouts in @group order for the pipeline layout.
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({
      template: TEMPLATE,
      // `Lvl` is pulled in automatically as a dependency of the `L` binding.
      externals: { ...layout0.bound },
      names: "strict",
    });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({
      bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)),
    });
    const pipeH = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "idwt_h" } });
    const pipeV = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "idwt_v" } });
    return { root, device, pipeH, pipeV };
  })();
  return pipeCache;
}

// Pooled GPU buffers, reused (and grown) across calls. Per-call allocate +
// destroy churns the Dawn addon and destabilises it; reuse also matches a real
// streaming decoder (allocate once per stream, not per frame). `bufA`/`bufB`
// ping-pong as the running LL; `scratch` is the per-line lifting workspace.
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

export interface Idwt53Opts {
  /** When false, skip the (Dawn-on-Node-fragile) full readback and just wait
   *  for GPU completion — for timing the compute, or a keep-on-GPU pipeline.
   *  Returns null in that case. Default true. */
  readback?: boolean;
}

/** Run the inverse 5/3 DWT on the GPU. Returns the reconstructed coefficients
 *  (before level shift), row-major `width * height`, or null when
 *  `opts.readback === false`. */
export async function idwt53Gpu(input: DwtInput53, opts: Idwt53Opts = {}): Promise<Int32Array | null> {
  const { root, device, pipeH, pipeV } = await getPipe();
  const { descriptor: desc, coeffs, width, height } = input;
  const shift = input.shift ?? 0;
  const at = (i: number): number => desc[i]!;
  const nLevels = at(1);
  const imgN = width * height;
  const ll0w = at(2),
    ll0h = at(3),
    ll0off = at(4);

  const p = ensurePool(root, imgN, coeffs.length);
  // Upload coefficients and seed the running LL (only the ll0 region is read at
  // level 1; deeper levels read the full previous-level output we write).
  device.queue.writeBuffer(root.unwrap(p.coeffBuf), 0, coeffs as BufferSource);
  device.queue.writeBuffer(root.unwrap(p.bufA), 0, coeffs.subarray(ll0off, ll0off + ll0w * ll0h) as BufferSource);

  let inbuf = p.bufA,
    outbuf = p.bufB;
  for (let lvl = 0; lvl < nLevels; lvl++) {
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
      shift: lvl === nLevels - 1 ? shift : 0,
    });
    const rawBind = root.unwrap(
      root.createBindGroup(layout0, {
        L: p.lvlBuf,
        inbuf,
        coeffs: p.coeffBuf,
        hbuf: p.hbuf,
        outbuf,
      }),
    );

    // One workgroup per line (row for the horizontal pass, column for vertical).
    const enc = device.createCommandEncoder();
    const ph = enc.beginComputePass();
    ph.setPipeline(pipeH);
    ph.setBindGroup(0, rawBind);
    ph.dispatchWorkgroups(rowCount);
    ph.end();
    const pv = enc.beginComputePass();
    pv.setPipeline(pipeV);
    pv.setBindGroup(0, rawBind);
    pv.dispatchWorkgroups(outW);
    pv.end();
    device.queue.submit([enc.finish()]);

    [inbuf, outbuf] = [outbuf, inbuf];
  }

  if (opts.readback === false) {
    await device.queue.onSubmittedWorkDone();
    return null;
  }
  const result = (await inbuf.read()) as ArrayLike<number>;
  return Int32Array.from(result).slice(0, imgN);
}
