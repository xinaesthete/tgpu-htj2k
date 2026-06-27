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

// STATUS (WIP): `decode_image` = HT block decode (bit-exact vs OpenJPH, see
// blockdecode.test.ts) + a reassembly/inverse-5/3-DWT pipeline. The DWT is
// structurally complete and reconstructs the image, but its boundary-extension
// parity does not yet match OpenJPH exactly, so it is NOT bit-exact for
// high-detail content (errors appear near, and propagate from, the right/bottom
// edges). These tests pin what currently holds; the strict bit-exact guarantee
// is tracked as the next DWT task.

test("decode_image returns correct dimensions", async () => {
  for (const [w, h, lv] of [[16, 16, 1], [32, 24, 2], [64, 64, 3]] as const) {
    const px = new Uint16Array(w * h);
    for (let i = 0; i < px.length; i++) px[i] = (i * 11) & 0x0fff;
    const { ours } = await decodeBoth(w, h, px, lv);
    expect(ours.length).toBe(w * h);
  }
});

test("decode_image is bit-exact vs OpenJPH for smooth single-level content", async () => {
  // Smooth content (near-zero detail bands) avoids the boundary-extension gap,
  // exercising the full entropy + DWT + level-shift pipeline end to end.
  for (const size of [16, 32, 64]) {
    const px = new Uint16Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) px[y * size + x] = (1000 + x * 3 + y * 5) & 0x0fff;
    const { ours, ref } = await decodeBoth(size, size, px, 1);
    expect(Array.from(ours)).toEqual(Array.from(ref));
  }
});
