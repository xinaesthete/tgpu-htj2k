# Packaging and consumer seams

Status: **design note** (2026-07-23) — promote to an ADR when the first package is actually cut.

The goal is to be able to use this work from other codebases: psychogeo/TerraCognita (terrain, GIS
raster), MDV + spatialdata.js (interactive spatial analysis), assorted experimental graphics and
architectural work. Today none of that is possible — the repo publishes nothing, and the most
reusable code is inside a private demo app.

## What the audit found (the good news first)

- **`src/` is entirely three.js-free.** Verified: zero `from "three"` outside `playground/`. The
  renderer-agnostic boundary ADR-0008 asserted actually holds.
- **`src/datasource/index.ts` already says** *"Self-contained: no dependency on the dancer/evo/sim
  code, so it can graduate."* The intent predates this note.
- **`src/gpu/spatial` has ten GPU-tested ops** and `src/geometry` a working procedural-geometry
  catalogue. There is real substance to publish, not just scaffolding.

The blockers are small and specific:

1. **`webgpu` (the Node-only Dawn addon) is a direct dependency of the root package.** A browser
   consumer would pull it. It must become `optionalDependencies` + peer, resolved through the
   existing `src/gpu/device.ts` seam (which already prefers `navigator.gpu`).
2. **No `exports` field, no build step, version `0.0.0`.**
3. **The reusable *viewer* layer is 6 381 lines trapped in `playground/`** — `TileRenderer`,
   `ChannelComposite`, the sd.js loaders, the volume renderers, camera work. This is precisely what
   MDV and psychogeo would consume, and it is private only because it was written inside a demo.

## The proposal: fewest packages that respect the hard boundaries

A package boundary should exist where a **dependency would otherwise be forced on consumers who do
not want it**. By that test there are three, so:

| Package | Contains | Heavy deps |
|---|---|---|
| **`@intraspatial/core`** | all of `src/`, subpath exports: `/datasource`, `/spatial`, `/geometry`, `/graph`, `/color`, `/evo` | typegpu (peer); **no** three, deck, or Dawn |
| **`@intraspatial/viewer-three`** | the three.js layer promoted out of `playground/`: tile renderer, channel composite, camera controls, `Viewport` shell | three (peer) |
| **`@intraspatial/viewer-deck`** | deck.gl interop — layers over the same datasource/ops | deck.gl, luma.gl (peer) |

Not six packages, one per `src/` subdirectory: every extra package is release overhead, and
subpath exports give the same import ergonomics without it. Split `core` further only when a real
consumer's dependency footprint demands it.

`@intraspatial` and the bare `intraspatial` are both free on npm (checked 2026-07-23).

### A likely fourth: the codec

`rust/htj2k-core` + the DWT kernels are the repo's origin and have no relationship to spatial
analysis. `@intraspatial/htj2k` — or keeping it under its own name entirely, as `openjph-wasm`
already is — is a reasonable split, but it is not on the critical path and can wait until someone
wants the codec without the toolbox.

## The deck.gl seam

deck.gl is expected to be central to the data-vis stack, and it has deliberately not been used here
yet: it is a fiddly dependency tree and less pleasant to prototype against than three.js. Both of
those are arguments for the seam rather than against the dependency — consumers who want deck opt in,
and nobody else pays.

**The interop question is the thing to spike, and it is not obviously easy.** Two shapes:

- **Two canvases, synchronised cameras.** deck renders its layers, three renders the scene, and a
  shared camera state drives both. Simple, robust, no shared context — but compositing is by DOM
  stacking, so there is no true depth interleaving between deck layers and three geometry.
- **One context, interleaved.** deck supports rendering into an externally-owned context; this is
  how deck-on-Mapbox works. It gives real interleaving, and it requires both sides to agree on a
  graphics API.

**The constraint that probably decides it, and which must be verified first:** this repo is
committed to **WebGPU** (three's `WebGPURenderer`, TSL node materials, TypeGPU compute), while deck.gl
is built on luma.gl and has historically been WebGL2-first, with WebGPU support experimental. If that
is still true, then one-context interleaving is simply unavailable and the answer is two canvases
with synchronised cameras — which in turn makes `viewer-deck` a thin package (camera-state adapter +
layer factories over `@intraspatial/core` data) rather than a rendering integration.

So the spike, in order:

1. What is deck.gl/luma.gl's actual WebGPU status now? This single fact decides the architecture.
2. Can a deck `Deck` instance and a three `WebGPURenderer` share a camera convincingly, including
   during interaction? The camera state model in ADR-0019 (`{pivot, orientation, distance}`) is the
   natural shared representation, and it is API-agnostic.
3. What is the smallest useful deck layer over our data — probably points/shapes from a SpatialData
   element, coloured by a spatial-stat op output.

## Sequencing

1. **Make `core` publishable.** Move Dawn to optional/peer, add `exports` + a build, version it.
   Cheap, because `src/` is already clean.
2. **Promote the viewer layer out of `playground/`** into `viewer-three`. This is the single
   highest-value structural move: it is what every downstream target needs, and the playground
   becomes a consumer of the package rather than its owner — which also proves the surface is real.
3. **Bridge SpatialData elements → `src/gpu/spatial` ops**, on a real 2-D store. This is the thing
   that makes the whole enterprise believable, and the audit says it is a bridge, not a build.
4. **Then** the deck spike, informed by (3) — because the first genuinely useful deck layer is a
   spatial-stat result, and it is easier to design the seam once there is something to put through it.

## Open questions

- **Release mechanics.** Changesets vs manual versioning; whether the packages version in lockstep.
  Unresolved and deliberately deferred until there are two packages to co-ordinate.
- **Build tooling.** The repo currently ships no built artefacts at all. TS 7 native emit vs a
  bundler (tsdown/unbuild) is an open call; the deciding factor is likely the `"use gpu"` TGSL
  transform, which today runs through `unplugin-typegpu` in the *consumer's* Vite config — so either
  the package ships pre-transformed kernels or it documents the plugin as a requirement. **This is
  the least understood part of publishing and should be resolved early**, because it constrains
  everything else.
- **Name.** `IntraSpatial`. One honest caveat: "intra-" conventionally reads as *within a single*
  discipline, which is close to the opposite of "usable across histology, GIS, architecture, and
  graphics". The intended reading — one substrate living *inside* each of many disciplines rather
  than bridging two — is coherent, but it will need explaining more than once.
