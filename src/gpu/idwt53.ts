// GPU inverse 5/3 (reversible) DWT — the TypeGPU/WebGPU port of the CPU
// `idwt_level` in `rust/htj2k-core/src/decode.rs`, used as the golden reference.
//
// Strategy: one GPU thread per DWT line. The 5/3 lifting is sequential *along*
// a line (each step reads its neighbours), but lines are independent, so we
// parallelise over rows (horizontal pass) and columns (vertical pass). Each
// thread runs the exact deinterleaved/clamp/parity/swap synthesis the CPU does,
// in i32 (WGSL `>>` on i32 is an arithmetic shift), so the result is bit-exact.
//
// Per level: horizontal pass (LL+HL → A low-vertical rows, LH+HH → B
// high-vertical rows, packed in `hbuf`), then vertical pass (A+B per column →
// output). Levels run coarse→fine in a JS loop, ping-ponging `inbuf`/`outbuf`.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "./device";

const WGSL = /* wgsl */ `
struct Lvl {
  rw0: u32, rh0: u32, rw1: u32, rh1: u32,
  out_w: u32, out_h: u32, even_x: u32, even_y: u32,
  hl_off: u32, lh_off: u32, hh_off: u32, _pad: u32,
};

@group(0) @binding(0) var<uniform> L: Lvl;
@group(0) @binding(1) var<storage, read>        inbuf:  array<i32>; // LL input (rw0 x rh0)
@group(0) @binding(2) var<storage, read>        coeffs: array<i32>; // HL/LH/HH
@group(0) @binding(3) var<storage, read_write>  hbuf:   array<i32>; // horiz out: A then B
@group(0) @binding(4) var<storage, read_write>  outbuf: array<i32>; // vert out
@group(0) @binding(5) var<storage, read_write>  scratch: array<i32>;

fn clampr(j: i32, n: u32) -> u32 {
  if (j < 0) { return 0u; }
  if (u32(j) >= n) { return n - 1u; }
  return u32(j);
}

// out_stride high bit selects the target buffer (0=hbuf, 1=outbuf).
fn outw(out_base: u32, out_stride: u32, p: u32, v: i32) {
  let sel = out_stride >> 31u;
  let stride = out_stride & 0x7fffffffu;
  let idx = out_base + p * stride;
  if (sel == 0u) { hbuf[idx] = v; } else { outbuf[idx] = v; }
}

// In-place 5/3 inverse lifting on scratch[base..base+nl+nh]:
// scratch[base..base+nl)        = low samples
// scratch[base+nl..base+nl+nh)  = high samples
fn lift(base: u32, nl: u32, nh: u32, even: u32) {
  let lo = base;
  let hi = base + nl;
  let off0: i32 = select(1i, 0i, even == 1u);   // ev = even at step 0
  let off1: i32 = select(0i, 1i, even == 1u);   // ev = !even at step 1
  // step 0 (update): low[i] -= (2 + high[i+off0-1] + high[i+off0]) >> 2
  for (var i: u32 = 0u; i < nl; i = i + 1u) {
    let a = scratch[hi + clampr(i32(i) + off0 - 1, nh)];
    let b = scratch[hi + clampr(i32(i) + off0, nh)];
    scratch[lo + i] = scratch[lo + i] - ((2 + a + b) >> 2u);
  }
  // step 1 (predict): high[i] += (low[i+off1-1] + low[i+off1]) >> 1
  for (var i: u32 = 0u; i < nh; i = i + 1u) {
    let a = scratch[lo + clampr(i32(i) + off1 - 1, nl)];
    let b = scratch[lo + clampr(i32(i) + off1, nl)];
    scratch[hi + i] = scratch[hi + i] + ((a + b) >> 1u);
  }
}

// Interleave scratch low/high into out[out_base + p*out_stride], p in 0..width.
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

// ---- Horizontal pass: one thread per row (A rows then B rows) ----
@compute @workgroup_size(64)
fn idwt_h(@builtin(global_invocation_id) gid: vec3u) {
  let ry = gid.x;
  let total_rows = L.rh0 + L.rh1;
  if (ry >= total_rows) { return; }

  let nl = L.rw0; let nh = L.rw1; let width = L.out_w;
  let base = ry * L.out_w;        // scratch region for this row
  // Choose source rows: A (ry<rh0) from LL(inbuf)+HL; B from LH+HH.
  var low_is_inbuf = false;
  var low_off: u32 = 0u; var high_off: u32 = 0u; var srow: u32 = 0u;
  var a_size = L.rh0 * L.out_w;   // hbuf offset where B rows begin
  if (ry < L.rh0) {
    low_is_inbuf = true; srow = ry;
    low_off = ry * L.rw0; high_off = L.hl_off + ry * L.rw1;
  } else {
    srow = ry - L.rh0;
    low_off = L.lh_off + srow * L.rw0; high_off = L.hh_off + srow * L.rw1;
  }
  // copy low/high into scratch
  for (var i: u32 = 0u; i < nl; i = i + 1u) {
    scratch[base + i] = select(coeffs[low_off + i], inbuf[low_off + i], low_is_inbuf);
  }
  for (var i: u32 = 0u; i < nh; i = i + 1u) {
    scratch[base + nl + i] = coeffs[high_off + i];
  }
  if (width == 1u) {
    let v = select(scratch[base + nl], scratch[base], L.even_x == 1u);
    let dst = select(a_size + srow * L.out_w, ry * L.out_w, ry < L.rh0) + 0u;
    hbuf[dst] = v;
    return;
  }
  lift(base, nl, nh, L.even_x);
  // write row into hbuf (stride 1, buffer sel 0)
  var out_base = select(a_size + srow * L.out_w, ry * L.out_w, ry < L.rh0);
  interleave(base, nl, nh, L.even_x, out_base, 1u);
}

// ---- Vertical pass: one thread per output column ----
@compute @workgroup_size(64)
fn idwt_v(@builtin(global_invocation_id) gid: vec3u) {
  let cx = gid.x;
  if (cx >= L.out_w) { return; }
  let nl = L.rh0; let nh = L.rh1; let height = L.out_h;
  let base = cx * L.out_h;       // scratch region for this column
  let a_size = L.rh0 * L.out_w;
  for (var y: u32 = 0u; y < nl; y = y + 1u) {
    scratch[base + y] = hbuf[y * L.out_w + cx];
  }
  for (var y: u32 = 0u; y < nh; y = y + 1u) {
    scratch[base + nl + y] = hbuf[a_size + y * L.out_w + cx];
  }
  if (height == 1u) {
    outbuf[cx] = select(scratch[base + nl], scratch[base], L.even_y == 1u);
    return;
  }
  lift(base, nl, nh, L.even_y);
  // write column into outbuf with stride out_w and buffer sel 1 (high bit set).
  interleave(base, nl, nh, L.even_y, cx, L.out_w | 0x80000000u);
}
`;

export interface DwtInput53 {
  descriptor: Uint32Array;
  coeffs: Int32Array;
  width: number;
  height: number;
}

// Pipeline state is expensive to build and immutable across calls, so cache it
// per device. (Re-creating modules/pipelines every call also churns the Dawn
// native addon, which destabilises its process-exit teardown under Node.)
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
    // Explicit layout so every binding is present for both entry points (auto
    // layout prunes bindings a given entry point doesn't statically touch).
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

/** Run the inverse 5/3 DWT on the GPU. Returns the reconstructed coefficients
 *  (before level shift), row-major `width * height`. */
export async function idwt53Gpu(input: DwtInput53): Promise<Int32Array> {
  const { device, bgl, pipeH, pipeV, paramsBuf } = await getPipe();
  const root = tgpu.initFromDevice({ device });
  const { descriptor: desc, coeffs, width, height } = input;
  // Descriptor is a well-formed flat header (see decode::dwt_input_53); reads
  // are in-bounds by construction, so assert non-null past noUncheckedIndexedAccess.
  const at = (i: number): number => desc[i]!;
  const nLevels = at(1);
  const imgN = width * height;

  // Per-call buffers (image-sized i32 storage). ll0 dims:
  const ll0w = at(2), ll0h = at(3), ll0off = at(4);
  const coeffBuf = root.createBuffer(d.arrayOf(d.i32, Math.max(1, coeffs.length)), Array.from(coeffs)).$usage("storage");
  let inbuf = root.createBuffer(d.arrayOf(d.i32, Math.max(1, imgN))).$usage("storage");
  let outbuf = root.createBuffer(d.arrayOf(d.i32, Math.max(1, imgN))).$usage("storage");
  const hbuf = root.createBuffer(d.arrayOf(d.i32, Math.max(1, imgN))).$usage("storage");
  const scratch = root.createBuffer(d.arrayOf(d.i32, Math.max(1, imgN))).$usage("storage");
  const owned = [coeffBuf, inbuf, outbuf, hbuf, scratch];

  // Seed inbuf with ll0.
  inbuf.write(Array.from(coeffs.subarray(ll0off, ll0off + ll0w * ll0h)).concat(
    new Array(Math.max(0, imgN - ll0w * ll0h)).fill(0)));

  for (let lvl = 0; lvl < nLevels; lvl++) {
    const o = 5 + lvl * 12;
    const rec = desc.subarray(o, o + 12);
    const outW = at(o + 4);
    const rowCount = at(o + 1) + at(o + 3); // rh0 + rh1
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array(rec));

    const gpuIn = root.unwrap(inbuf), gpuOut = root.unwrap(outbuf);
    const entries = [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: gpuIn } },
      { binding: 2, resource: { buffer: root.unwrap(coeffBuf) } },
      { binding: 3, resource: { buffer: root.unwrap(hbuf) } },
      { binding: 4, resource: { buffer: gpuOut } },
      { binding: 5, resource: { buffer: root.unwrap(scratch) } },
    ];
    const bind = device.createBindGroup({ layout: bgl, entries });

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

    // outbuf now holds this level's reconstruction; it becomes the next LL.
    [inbuf, outbuf] = [outbuf, inbuf];
  }

  const result = (await inbuf.read()) as ArrayLike<number>;
  const out = Int32Array.from(result).slice(0, imgN);
  for (const b of owned) b.destroy();
  return out;
}
