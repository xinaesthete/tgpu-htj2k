import { readFile } from "node:fs/promises";
import { encode } from "openjph-wasm";
import { expect, test } from "vitest";
import init, { parse_packets_summary } from "../rust/htj2k-core/pkg/htj2k_core.js";

let ready: Promise<unknown> | undefined;
async function ensure() {
  ready ??= (async () => {
    const wasmUrl = new URL("../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url);
    await init({ module_or_path: await readFile(wasmUrl) });
  })();
  await ready;
}

function ramp(width: number, height: number, components: number): Uint16Array {
  const data = new Uint16Array(components * width * height);
  for (let c = 0; c < components; c++) for (let i = 0; i < width * height; i++) data[c * width * height + i] = (c * 131 + i * 7) & 0x0fff;
  return data;
}

test("packet parse fully consumes the tile data (single component)", async () => {
  await ensure();
  const width = 64,
    height = 64;
  const cs = await encode({ data: ramp(width, height, 1), width, height, components: 1, reversible: true });
  const s = parse_packets_summary(cs);
  expect(s.num_code_blocks).toBeGreaterThan(0);
  expect(s.total_code_block_bytes).toBeGreaterThan(0);
  expect(s.bytes_consumed).toBe(s.tile_data_length);
  expect(s.fully_consumed).toBe(true);
  s.free();
});

test("0-decomposition lossless is cleanup-pass-only (decoder v1 scope)", async () => {
  await ensure();
  // No DWT: the single code-block's coefficients ARE the (level-shifted) pixels,
  // which is how the HT block decoder will be validated against OpenJPH.
  const width = 32,
    height = 32;
  const cs = await encode({ data: ramp(width, height, 1), width, height, components: 1, reversible: true, decompositions: 0 });
  const s = parse_packets_summary(cs);
  expect(s.num_code_blocks).toBe(1);
  expect(s.max_num_passes).toBe(1); // cleanup only — no SigProp/MagRef needed for v1
  expect(s.fully_consumed).toBe(true);
  s.free();
});

test("packet parse fully consumes the tile data (multi-component)", async () => {
  await ensure();
  const width = 48,
    height = 40,
    components = 3;
  const cs = await encode({ data: ramp(width, height, components), width, height, components, reversible: true });
  const s = parse_packets_summary(cs);
  expect(s.num_code_blocks).toBeGreaterThan(0);
  expect(s.fully_consumed).toBe(true);
  s.free();
});
