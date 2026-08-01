// How much does getting the inverse-DWT result back to CPU memory cost?
//
// Deliberately NOT a vitest benchmark. Inside the vitest fork, `mapAsync`
// completion is only observed on a coarse tick: every size from 128² to 1024²
// reports a flat ~125 ms, which is a fixed wait rather than bandwidth, and the
// ordering against size inverts. Run as a plain process the same measurement
// scales cleanly with the data (0.25 → 123 ms across 128² → 2048²). So the
// timing lives here and `test/bench_idwt.gpu.test.ts` keeps only the compute
// timings plus a readback *correctness* check.
//
// The column this produces is what decides whether a GPU DWT helps a
// decode-to-CPU consumer, as opposed to the keep-on-GPU viz path. See
// docs/dwt-gpu-and-high-bit-depth.md §1-2.
//
//   pnpm bench:readback
import { readFile } from "node:fs/promises";
import { encode } from "openjph-wasm";
import init, { decode_dwt_input_53, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { releaseDevice } from "../src/gpu/device";
import { idwt53Gpu } from "../src/gpu/idwt53";

const SIZES = [128, 256, 512, 1024, 2048];

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? Number.NaN;
}

function makePx(n: number) {
  const px = new Uint16Array(n * n);
  let s = 1;
  for (let i = 0; i < px.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    px[i] = s & 0x0fff;
  }
  return px;
}

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

async function main() {
  await init({
    module_or_path: await readFile(new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url)),
  });

  process.stdout.write(
    `\n  size   | CPU DWT | GPU compute | GPU+readback | readback | keep-on-GPU | decode-to-CPU\n` +
      `  -------+---------+-------------+--------------+----------+-------------+--------------\n`,
  );
  for (const n of SIZES) {
    const cs = await encode({
      data: makePx(n),
      width: n,
      height: n,
      components: 1,
      reversible: true,
      decompositions: 5,
    });
    const inp = decode_dwt_input_53(cs);
    const desc = inp.descriptor;
    const coeffs = inp.coeffs;
    const gpuInput = { descriptor: desc, coeffs, width: inp.width, height: inp.height };
    const reps = n >= 1024 ? 5 : 10;

    // A fast readback number must not be an empty buffer.
    const ref = idwt53_cpu(desc, coeffs);
    const got = await idwt53Gpu(gpuInput, { readback: true });
    if (!got || got.length !== ref.length) {
      throw new Error(`${n}²: readback returned ${got?.length ?? "null"}, want ${ref.length}`);
    }
    for (let i = 0; i < ref.length; i++) {
      if (got[i] !== ref[i]) throw new Error(`${n}²: readback mismatch at ${i}: ${got[i]} !== ${ref[i]}`);
    }

    const tCpu = timeS(reps, 2, () => idwt53_cpu(desc, coeffs));
    const tGpu = await timeA(reps, 3, () => idwt53Gpu(gpuInput, { readback: false }));
    const tRb = await timeA(reps, 3, () => idwt53Gpu(gpuInput, { readback: true }));
    process.stdout.write(
      `  ${String(n).padStart(4)}²  | ${tCpu.toFixed(2).padStart(7)} | ${tGpu.toFixed(2).padStart(7)} ms  |` +
        ` ${tRb.toFixed(2).padStart(8)} ms  | ${(tRb - tGpu).toFixed(2).padStart(6)} ms |` +
        ` ${(tCpu / tGpu).toFixed(2).padStart(8)}x   | ${(tCpu / tRb).toFixed(2).padStart(9)}x\n`,
    );
  }
  process.stdout.write(
    `\n  Medians, ms, over a verified-correct readback. A decode-to-CPU figure below\n` +
      `  1.00x means the GPU round-trip loses to just doing the DWT on the CPU.\n`,
  );

  // Exactly once, as the very last thing: a retained Dawn Instance holds a libuv
  // handle open, and releasing it mid-run crashes on live buffers' finalisers.
  await releaseDevice();
}

void main();
