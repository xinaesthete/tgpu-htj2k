// Loader for the Rust htj2k-core wasm module.
//
// wasm-pack `--target web` output fetches its .wasm by URL in the browser; under
// Node we read the bytes and hand them to the init function instead.
import init, { CodestreamInfo, parse_codestream, WaveletKernel } from "../../rust/htj2k-core/pkg/htj2k_core.js";

export { CodestreamInfo, WaveletKernel };

let ready: Promise<void> | undefined;

async function ensureReady(): Promise<void> {
  ready ??= (async () => {
    if (typeof window === "undefined") {
      // Node: load the wasm bytes from disk.
      const { readFile } = await import("node:fs/promises");
      const wasmUrl = new URL("../../rust/htj2k-core/pkg/htj2k_core_bg.wasm", import.meta.url);
      const bytes = await readFile(wasmUrl);
      await init({ module_or_path: bytes });
    } else {
      await init();
    }
  })();
  await ready;
}

/** Parse the main header of a J2K/HTJ2K codestream. */
export async function parseCodestream(data: Uint8Array): Promise<CodestreamInfo> {
  await ensureReady();
  return parse_codestream(data);
}
