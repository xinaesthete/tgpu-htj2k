import { test, expect } from "vitest";
import { encode } from "openjph-wasm";
import init, { decode_dwt_input_53, dwt_forward_53, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";
import { fdwt53Gpu } from "../src/gpu/fdwt53";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    await init({ module_or_path: await readFile(new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url)) });
  })();
  await ready;
}

// The GPU forward 5/3 must be bit-exact vs the CPU forward (dwt_forward_53,
// itself bit-exact vs OpenJPH's analysis — see forward.test.ts), and round-trip
// through the inverse to recover the level-shifted image.
test("GPU forward 5/3 DWT is bit-exact vs CPU forward (and round-trips)", async () => {
  await ensure();
  const sizes: [number, number][] = [[8, 8], [9, 9], [16, 16], [17, 23], [32, 32], [33, 48]];
  for (const [w, h] of sizes) {
    for (let lv = 1; lv <= Math.min(4, Math.floor(Math.log2(Math.min(w, h)))); lv++) {
      const px = new Uint16Array(w * h);
      let s = 1;
      for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
      const cs = await encode({ data: px, width: w, height: h, components: 1, reversible: true, decompositions: lv });
      const shift = decode_dwt_input_53(cs).level_shift;
      const shifted = Int32Array.from(px, (p) => p - shift);

      const cpu = dwt_forward_53(shifted, w, h, lv, 64, 64); // golden (== OpenJPH analysis)
      const gpu = (await fdwt53Gpu({
        descriptor: cpu.descriptor, image: shifted, width: w, height: h, coeffsLen: cpu.coeffs.length,
      }))!;

      const ref = cpu.coeffs as Int32Array;
      let mism = 0, first = "";
      for (let i = 0; i < ref.length; i++) if (gpu[i] !== ref[i]) { if (!mism) first = ` @${i}: gpu=${gpu[i]} cpu=${ref[i]}`; mism++; }
      expect(mism, `coeffs w=${w} h=${h} lv=${lv}${first}`).toBe(0);

      // Forward (GPU) → inverse (CPU) recovers the level-shifted image.
      const back = idwt53_cpu(cpu.descriptor, gpu) as Int32Array;
      let rt = 0;
      for (let i = 0; i < back.length; i++) if (back[i] !== shifted[i]) rt++;
      expect(rt, `round-trip w=${w} h=${h} lv=${lv}`).toBe(0);
    }
  }
});
