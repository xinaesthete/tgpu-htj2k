import { readFile } from "node:fs/promises";
import { decode, encode } from "openjph-wasm";
import { expect, test } from "vitest";
import init, { decode_dwt_input_53, decode_image, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";

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
  for (let i = 0; i < reps; i++) {
    const t = performance.now();
    await fn();
    ts.push(performance.now() - t);
  }
  return median(ts);
}
function timeS(reps: number, warm: number, fn: () => unknown) {
  for (let i = 0; i < warm; i++) fn();
  const ts: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t = performance.now();
    fn();
    ts.push(performance.now() - t);
  }
  return median(ts);
}

// CPU decode breakdown vs OpenJPH across sizes. Shows where time goes: the HT
// entropy decode (bit-serial, CPU-only) vs the inverse 5/3 DWT (the parallel,
// GPU-offloadable stage), against OpenJPH's fully-optimised C reference. No GPU
// here, so this runs at all sizes without the Dawn-on-Node stability ceiling.
test("benchmark: CPU decode breakdown vs OpenJPH", async () => {
  await ensure();
  process.stdout.write(`\n  size  | openjph full | ours entropy | ours CPU DWT | entropy/DWT\n`);
  process.stdout.write(`  ------+--------------+--------------+--------------+------------\n`);
  for (const n of [64, 128, 256, 512]) {
    const px = new Uint16Array(n * n);
    let s = 1;
    for (let i = 0; i < px.length; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      px[i] = s & 0x0fff;
    }
    const cs = await encode({ data: px, width: n, height: n, components: 1, reversible: true, decompositions: 5 });
    const inp = decode_dwt_input_53(cs);
    const desc = inp.descriptor,
      coeffs = inp.coeffs,
      shift = inp.level_shift;

    // Correctness (also covers sizes beyond the unit tests).
    const golden = decode_image(cs) as Int32Array;
    const cpuDwt = idwt53_cpu(desc, coeffs) as Int32Array;
    let m = 0;
    for (let i = 0; i < golden.length; i++) if (cpuDwt[i]! + shift !== golden[i]) m++;
    expect(m, `CPU DWT @${n}`).toBe(0);

    const tOjph = await timeA(5, 1, () => decode(cs));
    const tEntropy = timeS(8, 2, () => decode_dwt_input_53(cs));
    const tCpu = timeS(8, 2, () => idwt53_cpu(desc, coeffs));
    process.stdout.write(
      `  ${String(n).padStart(4)}² | ${tOjph.toFixed(2).padStart(8)} ms  | ${tEntropy.toFixed(2).padStart(8)} ms  | ${tCpu.toFixed(2).padStart(8)} ms  | ${(tEntropy / tCpu).toFixed(2)}x\n`,
    );
  }
  process.stdout.write(`\n  Medians, ms. "entropy" = HT decode + reassembly (decode_dwt_input_53).\n`);
});
