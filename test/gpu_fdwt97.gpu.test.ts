import { test, expect } from "vitest";
import init, { dwt_forward_97, idwt97_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";
import { fdwt97Gpu } from "../src/gpu/fdwt97";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    await init({ module_or_path: await readFile(new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url)) });
  })();
  await ready;
}

// GPU forward 9/7 ≈ CPU forward 9/7 (float, ≤ small tolerance), and round-trips
// through the inverse to recover the input.
test("GPU forward 9/7 DWT matches CPU forward and round-trips", async () => {
  await ensure();
  const sizes: [number, number][] = [[8, 8], [9, 9], [16, 16], [17, 23], [33, 48]];
  for (const [w, h] of sizes) {
    for (let lv = 1; lv <= Math.min(4, Math.floor(Math.log2(Math.min(w, h)))); lv++) {
      const img = new Float32Array(w * h);
      let s = 1;
      for (let i = 0; i < img.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; img[i] = ((s & 0x0fff) - 2048) / 4096; }

      const cpu = dwt_forward_97(img, w, h, lv, 64, 64); // golden
      const gpu = (await fdwt97Gpu({
        descriptor: cpu.descriptor, image: img, width: w, height: h, coeffsLen: cpu.coeffs.length,
      }))!;

      const ref = cpu.coeffs as Float32Array;
      let maxd = 0;
      for (let i = 0; i < ref.length; i++) maxd = Math.max(maxd, Math.abs(gpu[i]! - ref[i]!));
      expect(maxd, `coeffs w=${w} h=${h} lv=${lv}`).toBeLessThan(1e-4);

      // GPU forward → CPU inverse recovers the input.
      const back = idwt97_cpu(cpu.descriptor, gpu) as Float32Array;
      let rt = 0;
      for (let i = 0; i < back.length; i++) rt = Math.max(rt, Math.abs(back[i]! - img[i]!));
      expect(rt, `round-trip w=${w} h=${h} lv=${lv}`).toBeLessThan(1e-3);
    }
  }
});
