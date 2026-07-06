import { readFile } from "node:fs/promises";
import { encode } from "openjph-wasm";
import { expect, test } from "vitest";
import init, { decode_dwt_input_97, decode_image } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { idwt97Gpu } from "../src/gpu/idwt97";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    const wasmUrl = new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url);
    await init({ module_or_path: await readFile(wasmUrl) });
  })();
  await ready;
}

test("GPU inverse 9/7 DWT matches CPU decode_image within tolerance", async () => {
  await ensure();
  const sizes: [number, number][] = [
    [16, 16],
    [9, 9],
    [33, 48],
  ];
  for (const quality of [undefined, 0.02] as const) {
    for (const [w, h] of sizes) {
      for (let lv = 1; lv <= Math.min(3, Math.floor(Math.log2(Math.min(w, h)))); lv++) {
        const px = new Uint16Array(w * h);
        let s = 1;
        for (let i = 0; i < px.length; i++) {
          s = (s * 1103515245 + 12345) & 0x7fffffff;
          px[i] = s & 0x0fff;
        }
        const opts: any = { data: px, width: w, height: h, components: 1, reversible: false, decompositions: lv };
        if (quality !== undefined) opts.quality = quality;
        const cs = await encode(opts);

        const cpu = decode_image(cs) as Int32Array;

        const inp = decode_dwt_input_97(cs);
        // GPU does the float→pixel conversion too, so it equals decode_image
        // directly (within float tolerance).
        const gpu = (await idwt97Gpu(
          {
            descriptor: inp.descriptor,
            coeffs: inp.coeffs,
            width: inp.width,
            height: inp.height,
          },
          { pixels: { bitDepth: inp.bit_depth, signed: inp.signed } },
        ))! as Int32Array;

        let maxd = 0,
          sumAbs = 0;
        for (let i = 0; i < cpu.length; i++) {
          const dd = Math.abs(cpu[i]! - gpu[i]!);
          maxd = Math.max(maxd, dd);
          sumAbs += dd;
        }
        const mae = sumAbs / cpu.length;
        // The GPU 9/7 matches the CPU within 1 LSB: maxd<=1 proves the DWT +
        // dequant are correct (a real bug diverges by far more). Some pixels
        // differ by 1 between Dawn's f32 and Rust's f32 at round-to-int
        // boundaries (more under aggressive quantization), so we bound the mean
        // absolute error rather than requiring exactness.
        const label = `q=${quality} w=${w} h=${h} lv=${lv} maxd=${maxd} mae=${mae.toFixed(3)}`;
        expect(maxd, label).toBeLessThanOrEqual(1);
        expect(mae, label).toBeLessThan(0.5);
      }
    }
  }
});
