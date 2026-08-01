import { readFile } from "node:fs/promises";
import { encode } from "openjph-wasm";
import { test } from "vitest";
import init, { decode_dwt_input_53, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";
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

function makePx(n: number) {
  const px = new Uint16Array(n * n);
  let s = 1;
  for (let i = 0; i < px.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    px[i] = s & 0x0fff;
  }
  return px;
}

// CPU vs GPU inverse 5/3 DWT, keep-on-GPU (no readback) — the viz case.
//
// This file used to say the no-readback path was "the only one that survives
// Dawn-on-Node at large sizes (the mapAsync readback, not the compute, is what
// crashes there)". That was **our own bug**: src/gpu/device.ts let Dawn's
// Instance be GC'd out from under a live device, and allocation is what triggers
// collection, so a large staging buffer was an ideal trigger. Fixed 2026-07-29,
// and readback now runs clean to at least 4096²/67 MB — so each size below
// verifies its readback against the CPU result, which also stops a fast timing
// from being a silently empty buffer.
//
// Readback is *timed* in `pnpm bench:readback`, not here: inside the vitest fork
// mapAsync completion is only observed on a coarse tick, so every size from 128²
// to 1024² reports a flat ~125 ms — a fixed wait, not bandwidth. Same code as a
// plain process scales properly with the data.
//
// Opt-in (BENCH=1): benchmarks are timing-noisy and the heavy GPU work
// accumulates in the reused fork process alongside the other GPU tests. Run on
// demand: `pnpm bench:gpu`.
test.runIf(!!process.env.BENCH)("benchmark: inverse 5/3 DWT, CPU vs GPU (pooled, keep-on-GPU)", async () => {
  await ensure();

  process.stdout.write(`\n  size   | CPU DWT | GPU compute | speedup\n  -------+---------+-------------+--------\n`);
  // Largest size first so the buffer pool is sized once and reused (repeated
  // grow/destroy is the churn).
  const rows: string[] = [];
  for (const n of [2048, 1024, 512, 256, 128]) {
    const cs = await encode({ data: makePx(n), width: n, height: n, components: 1, reversible: true, decompositions: 5 });
    const inp = decode_dwt_input_53(cs);
    const desc = inp.descriptor,
      coeffs = inp.coeffs;
    const gpuInput = { descriptor: desc, coeffs, width: inp.width, height: inp.height };
    const reps = n >= 512 ? 5 : 10;

    // Check the readback path against the CPU once per size, so a fast number
    // here can never turn out to be a silently empty buffer.
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
    rows.push(
      `  ${String(n).padStart(4)}²  | ${tCpu.toFixed(2).padStart(7)} | ${tGpu.toFixed(2).padStart(7)} ms  | ${(tCpu / tGpu).toFixed(2)}x`,
    );
  }
  for (const r of rows.reverse()) process.stdout.write(`${r}\n`);
  process.stdout.write(
    `\n  Medians, ms. GPU = compute only (upload + dispatch + sync, result stays on\n` +
      `  GPU); CPU = full inverse DWT into CPU memory. Buffers are pooled/reused.\n` +
      `  Readback is verified here but timed in \`pnpm bench:readback\`.\n`,
  );
});
