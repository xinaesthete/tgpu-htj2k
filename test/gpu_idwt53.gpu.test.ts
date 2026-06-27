import { test, expect } from "vitest";
import { encode } from "openjph-wasm";
import init, { decode_image, decode_dwt_input_53 } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";
import { idwt53Gpu } from "../src/gpu/idwt53";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    const wasmUrl = new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url);
    await init({ module_or_path: await readFile(wasmUrl) });
  })();
  await ready;
}

test("GPU inverse 5/3 DWT is bit-exact vs CPU decode_image", async () => {
  await ensure();
  const sizes: [number, number][] = [[8, 8], [9, 9], [17, 23], [33, 48]];
  for (const [w, h] of sizes) {
    for (let lv = 1; lv <= Math.min(4, Math.floor(Math.log2(Math.min(w, h)))); lv++) {
      const px = new Uint16Array(w * h);
      let s = 1;
      for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
      const cs = await encode({ data: px, width: w, height: h, components: 1, reversible: true, decompositions: lv });

      const cpu = decode_image(cs) as Int32Array;

      const inp = decode_dwt_input_53(cs);
      const shift = inp.level_shift;
      const gpuCoeffs = (await idwt53Gpu({
        descriptor: inp.descriptor,
        coeffs: inp.coeffs,
        width: inp.width,
        height: inp.height,
      }))!;
      const gpu = Int32Array.from(gpuCoeffs, (c) => c + shift);

      let mismatches = 0, first = "";
      for (let i = 0; i < cpu.length; i++) {
        if (cpu[i] !== gpu[i]) {
          if (!mismatches) first = ` first @${i}: cpu=${cpu[i]} gpu=${gpu[i]}`;
          mismatches++;
        }
      }
      expect(mismatches, `w=${w} h=${h} lv=${lv}${first}`).toBe(0);
    }
  }
});
