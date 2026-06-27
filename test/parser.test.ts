import { test, expect } from "vitest";
import { encode } from "openjph-wasm";
import { parseCodestream, WaveletKernel } from "../src/wasm/htj2kCore";

function ramp(width: number, height: number, components: number): Uint16Array {
  const data = new Uint16Array(components * width * height);
  for (let c = 0; c < components; c++)
    for (let i = 0; i < width * height; i++)
      data[c * width * height + i] = (c * 997 + i) & 0xffff;
  return data;
}

test("parses geometry of a single-component HTJ2K codestream from OpenJPH", async () => {
  const width = 64, height = 48;
  const cs = await encode({
    data: ramp(width, height, 1),
    width,
    height,
    components: 1,
    reversible: true,
    decompositions: 5,
    blockSize: [64, 64],
  });

  const info = await parseCodestream(cs);
  expect(info.width).toBe(width);
  expect(info.height).toBe(height);
  expect(info.num_components).toBe(1);
  expect(info.reversible).toBe(true);
  expect(info.kernel).toBe(WaveletKernel.Reversible53);
  expect(info.num_decompositions).toBe(5);
  expect(info.code_block_width).toBe(64);
  expect(info.code_block_height).toBe(64);
  expect(info.component_bit_depth(0)).toBe(16);
  expect(info.component_is_signed(0)).toBe(false);
  expect(info.is_htj2k).toBe(true);

  // QCD: reversible => no quantization; dyadic subband count = 1 + 3·decompositions.
  expect(info.quant_style).toBe(0);
  expect(info.guard_bits).toBeGreaterThanOrEqual(1);
  expect(info.num_quant_subbands).toBe(1 + 3 * info.num_decompositions);

  // Layout geometry derived from the parsed params.
  expect(info.num_resolutions).toBe(info.num_decompositions + 1);
  expect(info.total_code_blocks(0)).toBeGreaterThan(0);

  // Tile-part: a single-tile codestream has one tile-part whose packet data
  // lies past SOD and within the codestream bounds.
  expect(info.num_tile_parts).toBe(1);
  expect(info.tile_part_tile_index(0)).toBe(0);
  const off = info.tile_part_data_offset(0);
  const len = info.tile_part_data_length(0);
  expect(off).toBeGreaterThan(0);
  expect(len).toBeGreaterThan(0);
  expect(off + len).toBeLessThanOrEqual(cs.length);
  info.free();
});

test("parses a multi-component (volumetric) codestream", async () => {
  const width = 32, height = 32, components = 4;
  const cs = await encode({
    data: ramp(width, height, components),
    width,
    height,
    components,
    reversible: true,
  });

  const info = await parseCodestream(cs);
  expect(info.num_components).toBe(components);
  expect(info.width).toBe(width);
  expect(info.height).toBe(height);
  for (let c = 0; c < components; c++) {
    expect(info.component_bit_depth(c)).toBe(16);
    expect(info.component_is_signed(c)).toBe(false);
  }
  info.free();
});
