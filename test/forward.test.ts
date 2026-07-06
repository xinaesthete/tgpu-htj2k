import { readFile } from "node:fs/promises";
import { encode } from "openjph-wasm";
import { expect, test } from "vitest";
import init, { decode_dwt_input_53, dwt_forward_53, idwt53_cpu } from "../rust/htj2k-core/pkg/htj2k_core.js";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    await init({ module_or_path: await readFile(new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url)) });
  })();
  await ready;
}

// The forward (analysis) 5/3 DWT must be bit-exact vs OpenJPH's analysis: encode
// an image with OpenJPH, extract its subband coefficients (decode_dwt_input_53 =
// OpenJPH's forward output), and check our forward on the same level-shifted
// pixels produces identical coefficients. Also round-trips with our inverse.
test("dwt_forward_53 is bit-exact vs OpenJPH analysis (and round-trips)", async () => {
  await ensure();
  const sizes: [number, number][] = [
    [8, 8],
    [9, 9],
    [16, 16],
    [17, 23],
    [32, 32],
    [33, 48],
  ];
  for (const [w, h] of sizes) {
    for (let lv = 1; lv <= Math.min(4, Math.floor(Math.log2(Math.min(w, h)))); lv++) {
      const px = new Uint16Array(w * h);
      let s = 1;
      for (let i = 0; i < px.length; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        px[i] = s & 0x0fff;
      }
      const cs = await encode({ data: px, width: w, height: h, components: 1, reversible: true, decompositions: lv });

      const ref = decode_dwt_input_53(cs); // OpenJPH's forward output
      const shift = ref.level_shift;
      const shifted = Int32Array.from(px, (p) => p - shift);

      const fwd = dwt_forward_53(shifted, w, h, lv, 64, 64);

      // Same geometry and identical coefficients as OpenJPH's analysis.
      expect(Array.from(fwd.descriptor), `desc w=${w} h=${h} lv=${lv}`).toEqual(Array.from(ref.descriptor));
      const a = fwd.coeffs as Int32Array,
        b = ref.coeffs as Int32Array;
      let mism = 0,
        first = "";
      for (let i = 0; i < a.length; i++)
        if (a[i] !== b[i]) {
          if (!mism) first = ` @${i}: ours=${a[i]} ref=${b[i]}`;
          mism++;
        }
      expect(mism, `coeffs w=${w} h=${h} lv=${lv}${first}`).toBe(0);

      // Forward → inverse is the identity (recovers the level-shifted image).
      const back = idwt53_cpu(fwd.descriptor, fwd.coeffs) as Int32Array;
      let rt = 0;
      for (let i = 0; i < back.length; i++) if (back[i] !== shifted[i]) rt++;
      expect(rt, `round-trip w=${w} h=${h} lv=${lv}`).toBe(0);
    }
  }
});
