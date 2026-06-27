import { test } from "vitest";
import { encode } from "openjph-wasm";
import init, { decode_dwt_input_53, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";
import { idwt53Gpu } from "../src/gpu/idwt53";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    await init({ module_or_path: await readFile(new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url)) });
  })();
  await ready;
}
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
async function timeA(reps: number, warm: number, fn: () => Promise<unknown>) {
  for (let i = 0; i < warm; i++) await fn();
  const ts: number[] = [];
  for (let i = 0; i < reps; i++) { const t = performance.now(); await fn(); ts.push(performance.now() - t); }
  return median(ts);
}
function timeS(reps: number, warm: number, fn: () => unknown) {
  for (let i = 0; i < warm; i++) fn();
  const ts: number[] = [];
  for (let i = 0; i < reps; i++) { const t = performance.now(); fn(); ts.push(performance.now() - t); }
  return median(ts);
}

function makePx(n: number) {
  const px = new Uint16Array(n * n);
  let s = 1;
  for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
  return px;
}

// CPU vs GPU inverse 5/3 DWT. Buffer pooling (reuse across calls) keeps Dawn
// stable; large-size GPU timing uses the *no-readback* path — the realistic
// keep-on-GPU viz case, and the only one that survives Dawn-on-Node at large
// sizes (the mapAsync readback, not the compute, is what crashes there). CPU is
// the full `idwt53_cpu`.
//
// Opt-in (BENCH=1): benchmarks are timing-noisy and the heavy GPU work
// accumulates in the reused fork process alongside the other GPU tests, which
// destabilises Dawn-on-Node. Run on demand: `pnpm bench:gpu`.
test.runIf(!!process.env.BENCH)("benchmark: inverse 5/3 DWT, CPU vs GPU (pooled, keep-on-GPU)", async () => {
  await ensure();

  // Pure timing (no readback). Correctness of the GPU DWT is covered by
  // gpu_idwt53.gpu.test.ts; mixing a full-buffer readback in here and then
  // growing the pool destabilises Dawn-on-Node. Largest size first so the
  // buffer pool is sized once and reused (repeated grow/destroy is the churn).
  process.stdout.write(`\n  size   | CPU DWT | GPU compute | speedup\n  -------+---------+-------------+--------\n`);
  // Capped at 256² for reliability: the full per-size loop (openjph encode +
  // CPU-DWT reps + GPU reps) is marginal at 512² and crashes at 1024² under
  // Dawn-on-Node, though those compute fine in isolation. The CPU↔GPU crossover
  // is ~512² (GPU ~1.1× with this naive kernel) — see docs.
  const rows: string[] = [];
  for (const n of [256, 128]) {
    const cs = await encode({ data: makePx(n), width: n, height: n, components: 1, reversible: true, decompositions: 5 });
    const inp = decode_dwt_input_53(cs);
    const desc = inp.descriptor, coeffs = inp.coeffs;
    const gpuInput = { descriptor: desc, coeffs, width: inp.width, height: inp.height };
    const reps = n >= 512 ? 5 : 10;
    const tCpu = timeS(reps, 2, () => idwt53_cpu(desc, coeffs));
    const tGpu = await timeA(reps, 3, () => idwt53Gpu(gpuInput, { readback: false }));
    rows.push(`  ${String(n).padStart(4)}²  | ${tCpu.toFixed(2).padStart(7)} | ${tGpu.toFixed(2).padStart(7)} ms  | ${(tCpu / tGpu).toFixed(2)}x`);
  }
  for (const r of rows.reverse()) process.stdout.write(r + "\n");
  process.stdout.write(
    `\n  Medians, ms. GPU = compute only (upload + dispatch + sync, result stays on\n` +
    `  GPU); CPU = full inverse DWT into CPU memory. Buffers are pooled/reused.\n`,
  );
});
