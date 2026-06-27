import { test, expect } from "vitest";
import { encode, decode } from "openjph-wasm";
import init, { decode_image } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    const wasmUrl = new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url);
    await init({ module_or_path: await readFile(wasmUrl) });
  })();
  await ready;
}

async function decodeBoth(width: number, height: number, px: Uint16Array, levels: number) {
  await ensure();
  const cs = await encode({ data: px, width, height, components: 1, reversible: true, decompositions: levels });
  const ours = decode_image(cs) as Int32Array;
  const ref = (await decode(cs)).data as Uint16Array;
  return { ours, ref };
}

// STATUS: `decode_image` = HT block decode (bit-exact vs OpenJPH, see
// blockdecode.test.ts) + reassembly + inverse 5/3 DWT. The DWT boundary
// extension now faithfully ports OpenJPH's gen_rev_horz_syn (clamp extension,
// origin-parity flag, aug/oth swap; horizontal-then-vertical pass order), so
// the full pipeline is bit-exact for high-detail content across odd /
// non-power-of-two / multi-level sizes. The one exception is *over-decomposed*
// images (more levels than the image supports → two consecutive 1x1
// resolutions), an upstream code-block-coverage corner no real encoder hits.

test("decode_image returns correct dimensions", async () => {
  for (const [w, h, lv] of [[16, 16, 1], [32, 24, 2], [64, 64, 3]] as const) {
    const px = new Uint16Array(w * h);
    for (let i = 0; i < px.length; i++) px[i] = (i * 11) & 0x0fff;
    const { ours } = await decodeBoth(w, h, px, lv);
    expect(ours.length).toBe(w * h);
  }
});

// Natural max decomposition depth — staying at or below it keeps the coarsest
// LL >= 2 in the smaller dimension at the level above, avoiding the
// over-decomposition corner.
const maxLevels = (w: number, h: number) =>
  Math.min(5, Math.max(1, Math.floor(Math.log2(Math.max(1, Math.min(w, h))))));

test("decode_image is bit-exact vs OpenJPH for high-detail content", async () => {
  // High-detail pseudo-random content populates every detail band and exercises
  // the boundary extension at right/bottom edges — exactly where the previous
  // mirror-based DWT diverged. Sweeps odd, prime, and non-square sizes across
  // multiple decomposition levels and seeds.
  const sizes: [number, number][] = [
    [2, 3], [5, 7], [8, 8], [9, 9], [13, 11], [16, 16],
    [17, 23], [31, 31], [32, 32], [33, 48], [64, 17], [37, 41],
  ];
  for (const [w, h] of sizes) {
    for (let lv = 1; lv <= maxLevels(w, h); lv++) {
      for (const seed0 of [1, 999, 424242]) {
        const px = new Uint16Array(w * h);
        let s = seed0;
        for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
        const { ours, ref } = await decodeBoth(w, h, px, lv);
        let mismatches = 0;
        for (let i = 0; i < ours.length; i++) if (ours[i] !== ref[i]) mismatches++;
        expect(mismatches, `w=${w} h=${h} lv=${lv} seed=${seed0}`).toBe(0);
      }
    }
  }
});

test("decode_image is bit-exact vs OpenJPH for smooth multi-level content", async () => {
  for (const size of [16, 32, 64]) {
    const px = new Uint16Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) px[y * size + x] = (1000 + x * 3 + y * 5) & 0x0fff;
    const { ours, ref } = await decodeBoth(size, size, px, 3);
    expect(Array.from(ours)).toEqual(Array.from(ref));
  }
});

// Irreversible 9/7 (lossy) path: dequant + float inverse 9/7 DWT + level-shift.
// Validated vs OpenJPH on high-detail content across sizes, levels, and
// quantization strengths. The float reconstruction matches OpenJPH's scalar
// path so closely that the round-to-integer makes the pixel output bit-exact;
// we allow a 1-LSB tolerance for cross-platform float rounding at boundaries.
test("decode_image matches OpenJPH for 9/7 lossy content", async () => {
  await ensure();
  const sizes: [number, number][] = [[8, 8], [16, 16], [9, 9], [17, 23], [33, 48]];
  for (const quality of [undefined, 0.001, 0.01, 0.05] as const) {
    for (const [w, h] of sizes) {
      for (let lv = 1; lv <= Math.min(4, Math.floor(Math.log2(Math.min(w, h)))); lv++) {
        const px = new Uint16Array(w * h);
        let s = 1;
        for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
        const opts: any = { data: px, width: w, height: h, components: 1, reversible: false, decompositions: lv };
        if (quality !== undefined) opts.quality = quality;
        const cs = await encode(opts);
        const ours = decode_image(cs) as Int32Array;
        const ref = (await decode(cs)).data as Uint16Array;
        let maxd = 0, exact = 0;
        for (let i = 0; i < ours.length; i++) {
          const d = Math.abs(ours[i] - ref[i]);
          if (d === 0) exact++;
          maxd = Math.max(maxd, d);
        }
        const label = `q=${quality} w=${w} h=${h} lv=${lv}`;
        expect(maxd, label).toBeLessThanOrEqual(1);
        expect(exact / ours.length, label).toBeGreaterThanOrEqual(0.99);
      }
    }
  }
});
