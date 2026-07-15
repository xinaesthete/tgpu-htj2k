// Implicit RAYMARCH view — the same `sdScene` (our WGSL) rendered exactly per-pixel: sharp CSG edges
// AND smooth curves, no meshing, no staircase. Built as a HYBRID pass in a three.js scene: it shares
// the depth buffer with real mesh geometry (a ground plane + a cube), so the raymarched surface and
// the meshes mutually occlude.
//
// Boundary (WebGPU-first / TSL-boundary principle): the SDF stays in OUR WGSL — embedded via `wgslFn`
// (constants baked, `Implicit.toWgsl({ bakeConstants: true })`). Only the ray loop / shading / depth
// plumbing is TSL, which is presentation, not geometry. Self-lit (shadertoy-ish); feeding three's PBR
// fragment stage is a later stretch.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  Break,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraViewMatrix,
  clamp,
  Discard,
  dot,
  Fn,
  float,
  If,
  Loop,
  max,
  normalize,
  positionWorld,
  struct,
  vec3,
  vec4,
  viewZToPerspectiveDepth,
  wgslFn,
} from "three/tsl";
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu";
import { SHAPES, type Shape } from "./geometryShapes";

const RayResult = struct({ color: "vec4", depth: "float" });

/** Build the raymarch material for a shape: an unlit node material whose fragment sphere-traces our
 *  `sdScene`, self-shades, and writes the hit's clip depth so it composites with mesh geometry. */
function makeRaymarchMaterial(shape: Shape): MeshBasicNodeMaterial {
  // Our SDF, constants baked → self-contained WGSL. `sdField` is the wgslFn entry (helpers below it).
  const sdField = wgslFn(`
fn sdField(p: vec3<f32>) -> f32 { return sdScene(p); }
${shape.make().toWgsl({ bakeConstants: true })}
`);

  const mat = new MeshBasicNodeMaterial();
  mat.side = THREE.DoubleSide; // proxy box faces generate fragments over the shape from any angle

  const march = Fn(() => {
    const ro = cameraPosition;
    const rd = normalize(positionWorld.sub(cameraPosition));

    // Sphere-trace. `relax` (<1) keeps steps safe where the CSG field over-estimates distance
    // (subtract / smooth regions), and a small minimum step stops grazing rays from crawling to a
    // halt before they reach the surface (the glancing-angle gaps). On a hit we interpolate the exact
    // zero-crossing between the last two samples rather than snapping to the step — that removes the
    // depth quantisation that bands along ridge-lines.
    const relax = float(0.9);
    const hitEps = float(0.0012);
    const t = float(0).toVar();
    const tPrev = float(0).toVar();
    const dPrev = float(1e9).toVar();
    const hit = float(0).toVar();
    const tHit = float(0).toVar();
    Loop(200, () => {
      const p = ro.add(rd.mul(t));
      const d = sdField({ p });
      If(d.lessThan(hitEps), () => {
        hit.assign(1);
        // Zero-crossing between (tPrev, dPrev) and (t, d) — handles under- and over-shoot.
        const f = dPrev.div(max(dPrev.sub(d), float(1e-5)));
        tHit.assign(tPrev.add(t.sub(tPrev).mul(f)));
        Break();
      });
      tPrev.assign(t);
      dPrev.assign(d);
      t.addAssign(max(d.mul(relax), float(0.0012)));
      If(t.greaterThan(26), () => {
        Break();
      });
    });
    Discard(hit.equal(0)); // no hit → let the mesh scene / background show
    const pHit = ro.add(rd.mul(tHit));

    // Normal by the tetrahedral gradient of the field.
    const e = float(0.0016);
    const k0 = vec3(1, -1, -1);
    const k1 = vec3(-1, -1, 1);
    const k2 = vec3(-1, 1, -1);
    const k3 = vec3(1, 1, 1);
    const n = normalize(
      k0
        .mul(sdField({ p: pHit.add(k0.mul(e)) }))
        .add(k1.mul(sdField({ p: pHit.add(k1.mul(e)) })))
        .add(k2.mul(sdField({ p: pHit.add(k2.mul(e)) })))
        .add(k3.mul(sdField({ p: pHit.add(k3.mul(e)) }))),
    );

    // Self-lit shade: a key light + hemisphere sky/ground + a little ambient.
    const lig = normalize(vec3(0.5, 0.85, 0.35));
    const dif = clamp(dot(n, lig), 0, 1);
    const hemi = clamp(n.y.mul(0.5).add(0.5), 0, 1);
    const base = vec3(0.95, 0.55, 0.22);
    const col = base.mul(dif.mul(vec3(1.0, 0.96, 0.86)).add(0.12)).add(base.mul(vec3(0.16, 0.2, 0.32)).mul(hemi));

    // Depth: project the world hit to clip depth so the hardware depth test composites with meshes.
    const hitViewZ = cameraViewMatrix.mul(vec4(pHit, 1)).z;
    const depth = viewZToPerspectiveDepth(hitViewZ, cameraNear, cameraFar);
    return RayResult(vec4(col, 1), depth);
  })();

  mat.colorNode = march.get("color");
  mat.depthNode = march.get("depth");
  return mat;
}

function fail(msg: string): void {
  const err = document.getElementById("err");
  if (err) {
    err.style.display = "grid";
    err.innerHTML = msg;
  }
}

async function main(): Promise<void> {
  const canvas = document.getElementById("stage") as HTMLCanvasElement | null;
  if (!canvas) return;
  if (!navigator.gpu) {
    fail("This view renders with <b>WebGPU</b>. Try a recent Chrome, Edge, or Safari.");
    return;
  }

  const renderer = new WebGPURenderer({ canvas, antialias: true });
  await renderer.init().catch((e) => {
    fail(`Couldn’t initialise WebGPU: ${e}`);
    throw e;
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 100);
  camera.position.set(4.4, 3.4, 5.0);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0.3, 0.2);

  // Real mesh geometry (lit by three), to prove the raymarch coexists + shares depth.
  scene.add(new THREE.HemisphereLight(0x9fb4ff, 0x20160c, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(3, 5, 2);
  scene.add(key);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshStandardMaterial({ color: 0x2a3350, roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(0.75, 0.75, 0.75),
    new THREE.MeshStandardMaterial({ color: 0x37c6c0, roughness: 0.4, metalness: 0.1 }),
  );
  cube.position.set(0.35, 0.42, 0.55); // straddles the front wall: back half buried, front half pokes out
  scene.add(cube);

  // The raymarch proxy: a box that just needs to cover the shape on screen; its fragments run the march.
  const proxy = new THREE.Mesh(new THREE.BoxGeometry(4, 4, 4), makeRaymarchMaterial(SHAPES[0] as Shape));
  proxy.position.set(0, 0, 0);
  scene.add(proxy);

  // ── UI ────────────────────────────────────────────────────────────────────────────────
  const shapeSel = document.getElementById("shape") as HTMLSelectElement;
  const cubeBox = document.getElementById("cube") as HTMLInputElement;
  SHAPES.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = s.name;
    shapeSel.appendChild(o);
  });
  shapeSel.selectedIndex = 0;
  function applyShape(): void {
    const shape = SHAPES[Number(shapeSel.value)] ?? SHAPES[0];
    if (!shape) return;
    proxy.material.dispose();
    proxy.material = makeRaymarchMaterial(shape);
  }
  shapeSel.addEventListener("change", applyShape);
  applyShape();
  cubeBox.addEventListener("change", () => {
    cube.visible = cubeBox.checked;
  });

  function resize(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h || 1;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

main().catch((e) => fail(`Error: ${e}`));
