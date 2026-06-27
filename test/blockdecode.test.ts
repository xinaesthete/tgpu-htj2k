import { test, expect } from "vitest";
import { encode, decode } from "openjph-wasm";
import init, { decode_first_codeblock } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    const wasmUrl = new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url);
    await init({ module_or_path: await readFile(wasmUrl) });
  })();
  await ready;
}

// A 0-decomposition codestream has no DWT, so the single code-block's decoded
// coefficients are the (level-shifted) pixels. We validate our HT cleanup block
// decoder bit-exact against OpenJPH's OWN decoder on the same codestream — the
// reference of record (this also matches truth except for inherent edge cases
// like the value that maps to coefficient -2^15).
async function compareToOpenJph(width: number, height: number, pixels: Uint16Array) {
  await ensure();
  const cs = await encode({ data: pixels, width, height, components: 1, reversible: true, decompositions: 0 });
  const ours = decode_first_codeblock(cs); // Int32Array
  const ref = await decode(cs); // OpenJPH reference
  expect(ours.length).toBe(width * height);
  expect(Array.from(ours)).toEqual(Array.from(ref.data as Uint16Array));
}

test("HT block decode == OpenJPH: constant image", async () => {
  const w = 16, h = 16;
  await compareToOpenJph(w, h, new Uint16Array(w * h).fill(1234));
});

test("HT block decode == OpenJPH: gradient (32x32)", async () => {
  const w = 32, h = 32;
  const px = new Uint16Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) px[y * w + x] = (x * 7 + y * 13) & 0x0fff;
  await compareToOpenJph(w, h, px);
});

test("HT block decode == OpenJPH: pseudo-random (64x64)", async () => {
  const w = 64, h = 64;
  const px = new Uint16Array(w * h);
  let s = 0x12345;
  for (let i = 0; i < px.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; px[i] = s & 0x0fff; }
  await compareToOpenJph(w, h, px);
});

test("HT block decode == OpenJPH: 8-bit image (48x40)", async () => {
  const w = 48, h = 40;
  const px = new Uint16Array(w * h);
  for (let i = 0; i < px.length; i++) px[i] = (i * 37) & 0xff;
  // encode as 8-bit by passing a Uint8Array
  await ensure();
  const u8 = Uint8Array.from(px);
  const cs = await encode({ data: u8, width: w, height: h, components: 1, reversible: true, decompositions: 0 });
  const ours = decode_first_codeblock(cs);
  const ref = await decode(cs);
  expect(Array.from(ours)).toEqual(Array.from(ref.data as Uint8Array));
});
