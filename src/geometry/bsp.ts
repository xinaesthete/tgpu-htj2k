// Plane-native boundary evaluation — the `implicit → clean mesh` bridge for the POLYHEDRAL subset
// (ADR-0010, CSGLib/quickhouse lineage). Where `tessellateSdf` samples the field on a grid (and so
// staircases every non-axis-aligned plane), this evaluates the boundary *exactly*: a solid's surface
// is a subset of its primitives' faces, so we generate each primitive's polygonal faces and keep the
// fragments that lie on the final boundary. Vertices are exact plane∩plane∩plane intersections and
// faces are as few and as large as the geometry allows — the "clean lines" the grid can't give.
//
// This is intentionally CPU, not GPU. The mesher is recursive convex-polygon clipping with a
// canonical-plane test and per-fragment classification — irregular, low-cardinality, precision-
// sensitive work (tens–hundreds of faces), the opposite of the massively-parallel per-pixel job the
// raymarch already does. The *image-space* form of this boolean IS on the GPU (that's `sdScene`
// sphere-tracing); this produces the orthogonal thing the raymarch can't — an actual mesh with
// topology and per-face provenance (ADR-0012). The small triangle list uploads to a resident buffer
// like any other. See the two-bridge picture in the module header of `implicit.ts`.
//
// The classifier is the exact CPU SDF (`evalSdf`) itself: for polyhedral primitives min/max of exact
// half-space/box fields gives the exact inside/outside *sign*, so the mesh boundary is validated
// against the same golden the raymarch renders — genuine red/green (see `bsp.test.ts`).
//
// Scope (first slice): generators `box`/`plane`; booleans `union`/`intersect`/`subtract`; domain
// transforms `translate`/`scale` (uniform, positive). Curved/smooth ops (`sphere`/`smoothUnion`) are
// rejected with a clear error — they route to grid-DC or raymarch. NOT yet handled: exactly coincident
// faces from two primitives sharing a plane (shared walls, the L/T junctions). That needs the
// coincident-face rule over a canonical-plane registry and is the next slice; `evaluateBrep` throws
// rather than emit a doubled or dropped face silently.

import type { Expr } from "./expr";
import { evalExpr } from "./expr";
import { evalSdf, type IsoMesh, type Sdf } from "./implicit";
import type { Vec3 } from "./superellipsoid";

// ── vector helpers ────────────────────────────────────────────────────────────────────
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: Vec3): Vec3 => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

const val = (e: Expr): number => evalExpr(e, 0, 0);

// ── planes and primitives ───────────────────────────────────────────────────────────────

/** An oriented half-space `dot(p, n) ≤ d`, `n` unit. The solid is on the `≤` side; `n` is the
 *  *outward* normal (points from inside to outside), matching the SDF sign (`dot(p,n) − d`). */
interface Plane {
  n: Vec3;
  d: number;
}

/** A polyhedral leaf: the intersection of its half-spaces, tagged with the primitive's provenance
 *  address (`primId` = pre-order index of the leaf in the tree). */
interface Prim {
  primId: number;
  planes: Plane[];
}

/** A cumulative uniform similarity from leaf coords to world: `world = S·leaf + O`. Descending the
 *  tree, `translate(t)` ⇒ `O += S·t`; `scale(f)` ⇒ `S *= f` (both keep plane normals unit). */
interface Xform {
  s: number;
  o: Vec3;
}
const IDENTITY: Xform = { s: 1, o: [0, 0, 0] };

/** Carry a leaf-space half-space `dot(leaf,n) ≤ d` into world space under `x`. With
 *  `leaf = (world − O)/S`: `dot(world, n) ≤ d·S + dot(O, n)`. `n` (unit) is unchanged. */
function planeToWorld(n: Vec3, d: number, x: Xform): Plane {
  return { n, d: d * x.s + dot(x.o, n) };
}

/** Flatten the tree into world-space polyhedral leaves, assigning each a pre-order `primId`. Throws on
 *  a non-polyhedral op (sphere/smoothUnion) — those don't have planar faces and must route elsewhere. */
function lower(node: Sdf): Prim[] {
  const prims: Prim[] = [];
  let nextId = 0;
  const walk = (n: Sdf, x: Xform): void => {
    switch (n.kind) {
      case "box": {
        const h: Vec3 = [val(n.half[0]), val(n.half[1]), val(n.half[2])];
        const axes: Vec3[] = [
          [1, 0, 0],
          [-1, 0, 0],
          [0, 1, 0],
          [0, -1, 0],
          [0, 0, 1],
          [0, 0, -1],
        ];
        const planes = axes.map((ax) => planeToWorld(ax, ax[0] !== 0 ? h[0] : ax[1] !== 0 ? h[1] : h[2], x));
        prims.push({ primId: nextId++, planes });
        break;
      }
      case "plane": {
        const raw: Vec3 = [val(n.n[0]), val(n.n[1]), val(n.n[2])];
        const nn = norm(raw); // the primitive normalises on evaluation; match it here
        prims.push({ primId: nextId++, planes: [planeToWorld(nn, val(n.d), x)] });
        break;
      }
      case "union":
      case "intersect":
      case "subtract":
        walk(n.a, x);
        walk(n.b, x);
        break;
      case "translate":
        walk(n.child, { s: x.s, o: add(x.o, mul([val(n.t[0]), val(n.t[1]), val(n.t[2])], x.s)) });
        break;
      case "scale":
        walk(n.child, { s: x.s * val(n.factor), o: x.o });
        break;
      case "sphere":
      case "smoothUnion":
        throw new Error(`bsp: non-polyhedral op '${n.kind}' — route curved/smooth geometry to grid-DC (toMesh) or raymarch`);
    }
  };
  walk(node, IDENTITY);
  return prims;
}

// ── convex-polygon clipping ─────────────────────────────────────────────────────────────

/** Two coplanar-up-to-sign planes are the *same geometric plane* — a face must not be split by its
 *  own plane (degenerate), and (next slice) coincident faces get merged by this test. */
function samePlane(a: Plane, b: Plane, eps: number): boolean {
  const parN = Math.abs(dot(a.n, b.n));
  if (parN < 1 - 1e-6) return false;
  const s = dot(a.n, b.n) >= 0 ? 1 : -1; // align orientation before comparing offsets
  return Math.abs(a.d - s * b.d) <= eps;
}

/** Clip a convex polygon to the half-space `dot(p,n) ≤ d` (Sutherland–Hodgman); returns the kept
 *  (inside) part, possibly empty. */
function clipHalf(poly: Vec3[], n: Vec3, d: number, eps: number): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i] as Vec3;
    const b = poly[(i + 1) % poly.length] as Vec3;
    const sa = dot(a, n) - d;
    const sb = dot(b, n) - d;
    if (sa <= eps) out.push(a);
    if ((sa < -eps && sb > eps) || (sa > eps && sb < -eps)) {
      const t = sa / (sa - sb);
      out.push(add(a, mul(sub(b, a), t)));
    }
  }
  return out;
}

/** Split a convex polygon by a plane into its inside (`≤ d`) and outside (`> d`) parts. Either may be
 *  empty when the polygon lies wholly on one side. */
function split(poly: Vec3[], n: Vec3, d: number, eps: number): { inside: Vec3[]; outside: Vec3[] } {
  return { inside: clipHalf(poly, n, d, eps), outside: clipHalf(poly, mul(n, -1), -d, eps) };
}

/** Drop consecutive near-duplicate vertices; a polygon with fewer than 3 distinct vertices has no
 *  area and is discarded upstream. */
function dedupe(poly: Vec3[], eps: number): Vec3[] {
  const out: Vec3[] = [];
  for (const v of poly) {
    const prev = out[out.length - 1];
    if (!prev || len(sub(v, prev)) > eps) out.push(v);
  }
  if (out.length >= 2 && len(sub(out[0] as Vec3, out[out.length - 1] as Vec3)) <= eps) out.pop();
  return out;
}

const centroid = (poly: Vec3[]): Vec3 =>
  mul(
    poly.reduce((acc, v) => add(acc, v), [0, 0, 0] as Vec3),
    1 / poly.length,
  );

/** A seed quad on `plane`, centred at the plane's foot-point and large enough (`half`) to enclose the
 *  scene; carved down by clipping/classification to the true face. */
function seedQuad(plane: Plane, half: number): Vec3[] {
  const { n, d } = plane;
  // Any axis not parallel to n gives an in-plane basis via two cross products.
  const ref: Vec3 = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = norm(cross(n, ref));
  const v = cross(n, u); // unit (n,u orthonormal)
  const c = mul(n, d); // foot of the plane from the origin
  return [
    add(c, add(mul(u, -half), mul(v, -half))),
    add(c, add(mul(u, half), mul(v, -half))),
    add(c, add(mul(u, half), mul(v, half))),
    add(c, add(mul(u, -half), mul(v, half))),
  ];
}

// ── boundary evaluation ───────────────────────────────────────────────────────────────

/** A face of the evaluated boundary: a convex polygon (CCW around `normal`), its outward `normal`,
 *  and provenance — which primitive (`primId`) and which of its planes (`planeId`) produced it. */
export interface BrepFace {
  poly: Vec3[];
  normal: Vec3;
  primId: number;
  planeId: number;
}

/** The boundary representation: a set of convex, provenance-tagged faces. */
export interface Brep {
  faces: BrepFace[];
}

export interface BrepOptions {
  /** Half-extent used to seed unbounded faces (bare half-spaces). Must enclose the solid. */
  bounds?: number;
}

/** Evaluate the exact polyhedral boundary of `node`. Each primitive face is seeded, clipped to its own
 *  primitive, split by every *other* primitive's planes so each fragment is uniformly in/out of the
 *  whole solid, then kept iff it separates inside from outside — the classic CSG boundary-by-
 *  classification, with the exact SDF (`evalSdf`) as the in/out oracle. */
export function evaluateBrep(node: Sdf, opts: BrepOptions = {}): Brep {
  const prims = lower(node);
  const half = (opts.bounds ?? 4) * 4; // seed quads well outside the solid; classification trims them
  const eps = 1e-7 * Math.max(half, 1);
  const nEps = 1e-4 * Math.max(opts.bounds ?? 4, 1); // classification offset off the face plane

  // Reject exactly-coincident faces up front (next-slice work) rather than doubling/dropping silently.
  const all: { owner: number; plane: Plane }[] = prims.flatMap((p) => p.planes.map((plane) => ({ owner: p.primId, plane })));
  for (let i = 0; i < all.length; i++)
    for (let j = i + 1; j < all.length; j++) {
      const a = all[i] as { owner: number; plane: Plane };
      const b = all[j] as { owner: number; plane: Plane };
      if (a.owner !== b.owner && samePlane(a.plane, b.plane, eps))
        throw new Error(
          "bsp: two primitives share a coincident plane — coincident-face handling is the next slice; use grid-DC (toMesh) for now",
        );
    }

  const faces: BrepFace[] = [];
  for (const prim of prims) {
    prim.planes.forEach((plane, planeId) => {
      // The primitive's own face on this plane: seed quad clipped by the primitive's other planes.
      let face = seedQuad(plane, half);
      for (const other of prim.planes) {
        if (other === plane) continue;
        face = clipHalf(face, other.n, other.d, eps);
      }
      if (face.length < 3) return;

      // Split by every other primitive's planes so each fragment is uniformly inside/outside the whole
      // solid. Skip planes coincident with this face (a face can't be split by its own geometric plane).
      let frags = [face];
      for (const { owner, plane: sp } of all) {
        if (owner === prim.primId) continue;
        if (samePlane(sp, plane, eps)) continue;
        const next: Vec3[][] = [];
        for (const f of frags) {
          const { inside, outside } = split(f, sp.n, sp.d, eps);
          if (inside.length >= 3) next.push(inside);
          if (outside.length >= 3) next.push(outside);
        }
        frags = next;
      }

      // Keep a fragment iff it separates inside from outside; orient its normal outward.
      for (const f0 of frags) {
        const f = dedupe(f0, eps);
        if (f.length < 3) continue;
        const c = centroid(f);
        const insideSolid = evalSdf(node, sub(c, mul(plane.n, nEps))) < 0; // just inside this half-space
        const outsideSolid = evalSdf(node, add(c, mul(plane.n, nEps))) < 0; // just outside it
        if (insideSolid === outsideSolid) continue; // buried or floating — not on ∂S
        if (insideSolid) faces.push({ poly: f, normal: plane.n, primId: prim.primId, planeId });
        else faces.push({ poly: [...f].reverse(), normal: mul(plane.n, -1), primId: prim.primId, planeId });
      }
    });
  }
  return { faces };
}

/** Triangulate a Brep into the interop `IsoMesh` form — one flat-shaded fan per convex face, so the
 *  clean facets read as clean facets. `facePrim[t]` gives the source `primId` of triangle `t` (the
 *  ADR-0012 provenance channel; `undefined`-free, one entry per triangle). */
export function brepToMesh(brep: Brep): IsoMesh & { facePrim: Uint32Array } {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const facePrim: number[] = [];
  for (const face of brep.faces) {
    const base = positions.length / 3;
    for (const v of face.poly) {
      positions.push(v[0], v[1], v[2]);
      normals.push(face.normal[0], face.normal[1], face.normal[2]);
    }
    for (let i = 1; i + 1 < face.poly.length; i++) {
      indices.push(base, base + i, base + i + 1);
      facePrim.push(face.primId);
    }
  }
  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    indices: Uint32Array.from(indices),
    vertexCount: positions.length / 3,
    facePrim: Uint32Array.from(facePrim),
  };
}
