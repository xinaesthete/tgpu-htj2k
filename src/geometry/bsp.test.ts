// Plane-native boundary evaluation (`bsp.ts`) validated against the exact SDF golden (`evalSdf`): the
// evaluated mesh must lie ON the field's zero-set and be oriented OUTWARD. Because both the mesh and
// the raymarch consume the same `evalSdf`, these are the same red/green the render is judged by.

import { describe, expect, it } from "vitest";
import { type Brep, brepToMesh, evaluateBrep } from "./bsp";
import { evalSdf, type Sdf } from "./implicit";
import { box, plane, sphere } from "./index";
import type { Vec3 } from "./superellipsoid";

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const centroid = (poly: Vec3[]): Vec3 =>
  mul(
    poly.reduce((acc, v) => add(acc, v), [0, 0, 0] as Vec3),
    1 / poly.length,
  );

/** A spread of interior sample points on a convex face — centroid, and each vertex nudged toward it. */
function facePoints(poly: Vec3[]): Vec3[] {
  const c = centroid(poly);
  return [c, ...poly.map((v) => add(v, mul(sub(c, v), 0.15)))];
}

/** Every face lies on the surface (|SDF| ≈ 0) and its normal points outward (SDF grows along +n). */
function expectValidBoundary(node: Sdf, brep: Brep): void {
  expect(brep.faces.length).toBeGreaterThan(0);
  for (const face of brep.faces) {
    for (const p of facePoints(face.poly)) {
      expect(Math.abs(evalSdf(node, p))).toBeLessThan(1e-5); // on the zero-set
    }
    const c = centroid(face.poly);
    expect(evalSdf(node, add(c, mul(face.normal, 1e-3)))).toBeGreaterThan(0); // just outside → outside
    expect(evalSdf(node, sub(c, mul(face.normal, 1e-3)))).toBeLessThan(0); // just inside → inside
  }
}

// A hip roof: a box (walls + floor) capped by four inclined eave planes — the archetypal polyhedral
// CSG with no axis-aligned roof (exactly what the grid staircases and this evaluates exactly).
function roofPlane(n: Vec3, through: Vec3) {
  const l = Math.hypot(n[0], n[1], n[2]);
  return plane(n, (n[0] * through[0] + n[1] * through[1] + n[2] * through[2]) / l);
}
const hipRoof = () =>
  box(1, 1.5, 0.6)
    .translate(0, 1.5, 0)
    .intersect(roofPlane([1, 1, 0], [1, 0.5, 0]))
    .intersect(roofPlane([-1, 1, 0], [-1, 0.5, 0]))
    .intersect(roofPlane([0, 1, 1], [0, 0.5, 0.6]))
    .intersect(roofPlane([0, 1, -1], [0, 0.5, -0.6]));

describe("evaluateBrep — plane-native boundary", () => {
  it("meshes a box as six outward quads", () => {
    const brep = evaluateBrep(box(1).node, { bounds: 1.5 });
    expect(brep.faces).toHaveLength(6);
    for (const f of brep.faces) expect(f.poly).toHaveLength(4);
    expectValidBoundary(box(1).node, brep);
  });

  it("triangulates to a flat-shaded fan mesh with per-triangle provenance", () => {
    const mesh = brepToMesh(evaluateBrep(box(1).node, { bounds: 1.5 }));
    expect(mesh.indices.length / 3).toBe(12); // six quads → two tris each
    expect(mesh.vertexCount).toBe(24); // fans don't share vertices (flat shading)
    expect(mesh.facePrim.length).toBe(12); // one provenance id per triangle
    expect(new Set(mesh.facePrim)).toEqual(new Set([0])); // all from the single box primitive
  });

  it("evaluates a hip roof exactly onto the SDF surface", () => {
    const g = hipRoof();
    const brep = evaluateBrep(g.node, { bounds: 2 });
    expectValidBoundary(g.node, brep);
    // Walls + floor from the box (primId 0) and one facet per roof plane (primIds 1–4) all survive.
    const prims = new Set(brep.faces.map((f) => f.primId));
    expect(prims).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("evaluates a subtractive notch (box ∖ box) onto the surface, oriented into the cavity", () => {
    const g = box(1).subtract(box(0.4).translate(1, 0, 0)); // a square bite out of the +x face
    const brep = evaluateBrep(g.node, { bounds: 2 });
    expectValidBoundary(g.node, brep);
    // The cavity walls come from the tool primitive (primId 1), the outer shell from the main (0).
    const prims = new Set(brep.faces.map((f) => f.primId));
    expect(prims.has(0)).toBe(true);
    expect(prims.has(1)).toBe(true);
  });

  it("carries transforms (translate/scale) into world-space planes", () => {
    const g = box(0.5).scale(2).translate(3, -1, 0.5); // → a box(1) centred at (3,-1,0.5)
    const brep = evaluateBrep(g.node, { bounds: 5 });
    expect(brep.faces).toHaveLength(6);
    expectValidBoundary(g.node, brep);
    // A face centre should sit a unit from the translated centre along an axis.
    const centres = brep.faces.map((f) => centroid(f.poly));
    expect(centres.some((c) => Math.abs(c[0] - 4) < 1e-6)).toBe(true); // +x face at x = 3 + 1
  });
});

describe("evaluateBrep — scope guards", () => {
  it("rejects a non-polyhedral op with a routing hint", () => {
    expect(() => evaluateBrep(box(1).subtract(sphere(0.5)).node)).toThrow(/non-polyhedral/);
  });

  it("rejects exactly-coincident faces (shared-wall case) — the next slice", () => {
    const shared = box(1).union(box(1).translate(2, 0, 0)); // both share the plane x = 1
    expect(() => evaluateBrep(shared.node, { bounds: 3 })).toThrow(/coincident/);
  });
});
