import { test, expect } from "vitest";
import { encode } from "openjph-wasm";
import init, { decode_image, decode_dwt_input_53, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";
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

// CPU vs GPU for the inverse 5/3 DWT, at small sizes. Kept lean and in its own
// fork process (minimal surrounding wasm), because the Dawn (`webgpu`) addon
// under Node destabilises after enough cumulative GPU+wasm work in one process
// — which currently blocks rigorous large-size GPU timing here (a follow-up:
// raise that ceiling or run in a browser). The CPU-side size sweep (entropy vs
// DWT vs OpenJPH) lives in `bench_idwt.test.ts`, which has no such limit.
test("benchmark: inverse 5/3 DWT, CPU vs GPU (small sizes)", async () => {
  await ensure();
  process.stdout.write(`\n  size  | CPU DWT | GPU DWT(+io) | ratio\n  ------+---------+--------------+------\n`);
  // Single size: ~8 cumulative GPU calls in one Node process hits the Dawn
  // teardown-stability ceiling, so we measure one size with few iterations.
  for (const n of [64]) {
    const px = new Uint16Array(n * n);
    let s = 1;
    for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
    const cs = await encode({ data: px, width: n, height: n, components: 1, reversible: true, decompositions: 5 });
    const inp = decode_dwt_input_53(cs);
    const desc = inp.descriptor, coeffs = inp.coeffs, shift = inp.level_shift;
    const golden = decode_image(cs) as Int32Array;
    const gpuInput = { descriptor: desc, coeffs, width: inp.width, height: inp.height };

    const g = await idwt53Gpu(gpuInput);
    let m = 0;
    for (let i = 0; i < golden.length; i++) if (g[i]! + shift !== golden[i]) m++;
    expect(m, `GPU DWT @${n}`).toBe(0);

    const tCpu = timeS(8, 2, () => idwt53_cpu(desc, coeffs));
    const tGpu = await timeA(3, 1, () => idwt53Gpu(gpuInput));
    process.stdout.write(
      `  ${String(n).padStart(4)}² | ${tCpu.toFixed(2).padStart(7)} | ${tGpu.toFixed(2).padStart(7)} ms   | ${(tCpu / tGpu).toFixed(2)}x\n`,
    );
  }
  process.stdout.write(`\n  Medians, ms. GPU includes upload + dispatch + full readback.\n`);
});
