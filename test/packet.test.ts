import { test, expect } from "vitest";
import { encode } from "openjph-wasm";
import init, { parse_packets_summary } from "../rust/htj2k-core/pkg/htj2k_core.js";
import { readFile } from "node:fs/promises";

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
  for (let c = 0; c < components; c++)
    for (let i = 0; i < width * height; i++)
      data[c * width * height + i] = (c * 131 + i * 7) & 0x0fff;
  return data;
}

test("packet parse fully consumes the tile data (single component)", async () => {
  await ensure();
  const width = 64, height = 64;
  const cs = await encode({ data: ramp(width, height, 1), width, height, components: 1, reversible: true });
  const s = parse_packets_summary(cs);
  expect(s.num_code_blocks).toBeGreaterThan(0);
  expect(s.total_code_block_bytes).toBeGreaterThan(0);
  expect(s.bytes_consumed).toBe(s.tile_data_length);
  expect(s.fully_consumed).toBe(true);
  s.free();
});

test("packet parse fully consumes the tile data (multi-component)", async () => {
  await ensure();
  const width = 48, height = 40, components = 3;
  const cs = await encode({ data: ramp(width, height, components), width, height, components, reversible: true });
  const s = parse_packets_summary(cs);
  expect(s.num_code_blocks).toBeGreaterThan(0);
  expect(s.fully_consumed).toBe(true);
  s.free();
});
