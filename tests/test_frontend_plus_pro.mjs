import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { execFileSync } from 'node:child_process';

const ROOT = path.resolve(import.meta.dirname, '..');
const THREE_JS_PATH = path.join(ROOT, 'frontend_lab', 'three.min.js');
const PLUS_PRO_JS_PATH = path.join(ROOT, 'frontend_lab', 'alpha-plus-pro.js');

function loadThreeAndModule() {
  const threeCode = fs.readFileSync(THREE_JS_PATH, 'utf8');
  const modCode = fs.readFileSync(PLUS_PRO_JS_PATH, 'utf8');

  const sandbox = {
    console,
    TextDecoder,
    TextEncoder,
    URL,
    URLSearchParams,
    Blob,
    FormData,
    Uint8Array,
    Uint16Array,
    Uint32Array,
    Int8Array,
    Int16Array,
    Float32Array,
    DataView,
    ArrayBuffer,
    Math,
    setTimeout,
    clearTimeout,
    localStorage: { _m: {}, getItem(k) { return this._m[k] ?? null; }, setItem(k, v) { this._m[k] = String(v); }, removeItem(k) { delete this._m[k]; } },
    window: null,
    self: null,
    document: { createElement: () => ({ style: {}, addEventListener() {} }), getElementById: () => null },
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(threeCode, sandbox);
  assert.ok(sandbox.THREE, 'THREE r128 should load cleanly');
  vm.runInContext(modCode, sandbox);
  assert.ok(sandbox.AlphaPlusPro, 'window.AlphaPlusPro should register cleanly');
  return sandbox;
}

function generateMockGlbBytes(privacyLevel, heightCm = 172.0, reconstructionMode = 'calibrated_mhr') {
  const pyScript = [
    'import base64, json, os, sys, tempfile',
    'from PIL import Image',
    'from bodyscan_pipeline.scan_service import MockScanEngine, ScanRequest',
    `level = ${privacyLevel}`,
    `h_cm = ${heightCm}`,
    `rmode = ${JSON.stringify(reconstructionMode)}`,
    'with tempfile.TemporaryDirectory() as tmp:',
    '    imgs = []',
    '    for i in range(4):',
    '        p = os.path.join(tmp, f"a{i}.jpg")',
    '        Image.new("RGB", (32, 32), (180, 140, 120)).save(p, format="JPEG")',
    '        imgs.append(p)',
    '    out_dir = os.path.join(tmp, "out")',
    '    res = MockScanEngine(step_delay_sec=0.0).process(ScanRequest(image_paths=imgs, height_cm=h_cm, sex="male", weight_kg=70.0, privacy_level=level, consent_granted=True, reconstruction_mode=rmode, output_dir=out_dir))',
    '    with open(res.artifacts[res.viewer_artifact_key], "rb") as fp:',
    '        b64 = base64.b64encode(fp.read()).decode("ascii")',
    '    core_b64 = None',
    '    if "core_glb" in res.artifacts and os.path.isfile(res.artifacts["core_glb"]):',
    '        with open(res.artifacts["core_glb"], "rb") as fp:',
    '            core_b64 = base64.b64encode(fp.read()).decode("ascii")',
    '    print(json.dumps({"b64": b64, "core_b64": core_b64, "body_scan": res.body_json}))',
  ].join('\n');

  const out = execFileSync('python', ['-c', pyScript], { cwd: ROOT, encoding: 'utf8' });
  const parsed = JSON.parse(out);
  const buf = Buffer.from(parsed.b64, 'base64');
  let coreArrayBuffer = null;
  if (parsed.core_b64) {
    const coreBuf = Buffer.from(parsed.core_b64, 'base64');
    coreArrayBuffer = coreBuf.buffer.slice(coreBuf.byteOffset, coreBuf.byteOffset + coreBuf.byteLength);
  }
  return {
    arrayBuffer: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    coreArrayBuffer,
    bodyScan: parsed.body_scan,
  };
}

test('isLoopbackUrl accepts localhost/127.0.0.1/[::1] and rejects remote hosts', () => {
  const { AlphaPlusPro } = loadThreeAndModule();
  assert.equal(AlphaPlusPro.isLoopbackUrl('http://localhost:8000'), true);
  assert.equal(AlphaPlusPro.isLoopbackUrl('http://127.0.0.1:8000'), true);
  assert.equal(AlphaPlusPro.isLoopbackUrl('https://[::1]:8443/api'), true);
  assert.equal(AlphaPlusPro.isLoopbackUrl('https://demo-gpu.trycloudflare.com'), false);
  assert.equal(AlphaPlusPro.isLoopbackUrl('http://192.168.1.50:8000'), false);
  assert.equal(AlphaPlusPro.isLoopbackUrl('not a url'), false);
  assert.equal(AlphaPlusPro.isLoopbackUrl(''), false);
});

test('Decision 2a guard blocks remote GPU server when consentMode === local', () => {
  const { AlphaPlusPro } = loadThreeAndModule();
  const base = { privacyLevel: 2, consentChecked: true, photoCount: 4, heightCm: 170 };

  const blocked = AlphaPlusPro.validatePrivacyAndConnection({
    ...base,
    consentMode: 'local',
    serverUrl: 'https://colab-gpu.trycloudflare.com',
  });
  assert.equal(blocked.ok, false);
  assert.equal(blocked.reasonKey, 'pp_err_local_only');

  const allowedLocal = AlphaPlusPro.validatePrivacyAndConnection({
    ...base,
    consentMode: 'local',
    serverUrl: 'http://127.0.0.1:8000',
  });
  assert.equal(allowedLocal.ok, true);

  const allowedSaved = AlphaPlusPro.validatePrivacyAndConnection({
    ...base,
    consentMode: 'save',
    serverUrl: 'https://colab-gpu.trycloudflare.com',
  });
  assert.equal(allowedSaved.ok, true);

  const noConsent = AlphaPlusPro.validatePrivacyAndConnection({
    ...base,
    consentChecked: false,
    consentMode: 'save',
    serverUrl: 'http://127.0.0.1:8000',
  });
  assert.equal(noConsent.ok, false);
  assert.equal(noConsent.reasonKey, 'pp_err_consent');

  // Even before selecting photos/privacy, LOCAL_ONLY with remote URL surfaces pp_err_local_only immediately
  const earlyBlocked = AlphaPlusPro.validatePrivacyAndConnection({
    consentMode: 'local',
    serverUrl: 'https://remote.example.com',
    privacyLevel: null,
    consentChecked: false,
    photoCount: 0,
    heightCm: 162,
  });
  assert.equal(earlyBlocked.ok, false);
  assert.equal(earlyBlocked.reasonKey, 'pp_err_local_only');
});

test('parseGlbBinary parses Level 1 & Level 2 MockScanEngine .glb cleanly with flipY=false and sRGBEncoding', async () => {
  const { THREE, AlphaPlusPro } = loadThreeAndModule();
  const loadedImages = [];
  const fakeImageFactory = async (bytes, mimeType) => {
    loadedImages.push({ size: bytes.byteLength, mimeType });
    return { width: 512, height: 512 };
  };

  // Level 1: untextured mesh
  const lvl1 = generateMockGlbBytes(1, 172.0);
  const parsed1 = await AlphaPlusPro.parseGlbBinary(lvl1.arrayBuffer, THREE, fakeImageFactory);
  assert.ok(parsed1.group instanceof THREE.Group);
  assert.equal(parsed1.stats.meshes, 1);
  assert.equal(parsed1.stats.textures, 0);
  assert.ok(Math.abs(parsed1.metrics.hY - 1.72) < 0.08, `Expected ~1.72m, got ${parsed1.metrics.hY}`);
  assert.ok(Math.abs(parsed1.metrics.center[1]) < 0.02, 'Model should center around Y=0');

  // Level 2: blurred face JPEG embedded
  const lvl2 = generateMockGlbBytes(2, 165.0);
  const parsed2 = await AlphaPlusPro.parseGlbBinary(lvl2.arrayBuffer, THREE, fakeImageFactory);
  assert.equal(parsed2.stats.meshes, 1);
  assert.equal(parsed2.stats.textures, 1);
  assert.deepEqual(Array.from(parsed2.stats.flipYs), [false]);
  assert.equal(loadedImages.length, 1);
  assert.equal(loadedImages[0].mimeType, 'image/jpeg');
  assert.ok(loadedImages[0].size > 500);

  let foundTex = null;
  let disposedGeo = 0;
  let disposedMat = 0;
  let disposedTex = 0;
  parsed2.group.traverse((obj) => {
    if (obj.isMesh) {
      foundTex = obj.material.map;
      const origG = obj.geometry.dispose.bind(obj.geometry);
      obj.geometry.dispose = () => { disposedGeo++; origG(); };
      const origM = obj.material.dispose.bind(obj.material);
      obj.material.dispose = () => { disposedMat++; origM(); };
      if (obj.material.map) {
        const origT = obj.material.map.dispose.bind(obj.material.map);
        obj.material.map.dispose = () => { disposedTex++; origT(); };
      }
    }
  });
  assert.ok(foundTex);
  assert.equal(foundTex.flipY, false);
  assert.equal(foundTex.encoding, THREE.sRGBEncoding);
  // Textured glTF material with omitted baseColorFactor must default to pure white (1,1,1), never brown tint
  let texturedMat = null;
  parsed2.group.traverse((obj) => { if (obj.isMesh && obj.material && obj.material.map) texturedMat = obj.material; });
  assert.ok(texturedMat);
  assert.ok(Math.abs(texturedMat.color.r - 1.0) < 1e-4 && Math.abs(texturedMat.color.g - 1.0) < 1e-4 && Math.abs(texturedMat.color.b - 1.0) < 1e-4, `Expected white (1,1,1) multiplier on textured material, got (${texturedMat.color.r}, ${texturedMat.color.g}, ${texturedMat.color.b})`);

  // External glTF where baseColorTexture is set but baseColorFactor is omitted (glTF 2.0 default [1,1,1,1])
  const lvl2OmittedBcf = (() => {
    const dv = new DataView(lvl2.arrayBuffer);
    const jsonLen = dv.getUint32(12, true);
    const gltfStr = Buffer.from(lvl2.arrayBuffer, 20, jsonLen).toString('utf8');
    const g = JSON.parse(gltfStr);
    if (g.materials && g.materials[0] && g.materials[0].pbrMetallicRoughness) {
      delete g.materials[0].pbrMetallicRoughness.baseColorFactor;
    }
    const rawJson = Buffer.from(JSON.stringify(g), 'utf8');
    const padJsonLen = (rawJson.length + 3) & ~3;
    const jsonChunk = Buffer.alloc(padJsonLen, 0x20);
    rawJson.copy(jsonChunk);
    const binLen = dv.getUint32(20 + jsonLen, true);
    const binData = Buffer.from(lvl2.arrayBuffer, 28 + jsonLen, binLen);
    const out = Buffer.alloc(12 + 8 + padJsonLen + 8 + binLen);
    out.writeUInt32LE(0x46546C67, 0); out.writeUInt32LE(2, 4); out.writeUInt32LE(out.length, 8);
    out.writeUInt32LE(padJsonLen, 12); out.writeUInt32LE(0x4E4F534A, 16); jsonChunk.copy(out, 20);
    const bOff = 20 + padJsonLen;
    out.writeUInt32LE(binLen, bOff); out.writeUInt32LE(0x004E4942, bOff + 4); binData.copy(out, bOff + 8);
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
  })();
  const parsedOmitted = await AlphaPlusPro.parseGlbBinary(lvl2OmittedBcf, THREE, fakeImageFactory);
  let omittedMat = null;
  parsedOmitted.group.traverse((obj) => { if (obj.isMesh && obj.material) omittedMat = obj.material; });
  assert.ok(omittedMat && omittedMat.map);
  assert.ok(Math.abs(omittedMat.color.r - 1.0) < 1e-4 && Math.abs(omittedMat.color.g - 1.0) < 1e-4 && Math.abs(omittedMat.color.b - 1.0) < 1e-4, `Omitted baseColorFactor on textured GLB must default to white (1,1,1), got (${omittedMat.color.r}, ${omittedMat.color.g}, ${omittedMat.color.b})`);

  AlphaPlusPro.disposeObject3D(parsed2.group);
  assert.equal(disposedGeo, 1);
  assert.equal(disposedMat, 1);
  assert.equal(disposedTex, 1);
});

test('extract16GirthRows and applyScanResultToClient preserve ANATOMICAL_ORDER and CLIENT_INBODY', () => {
  const { AlphaPlusPro } = loadThreeAndModule();
  const lvl2 = generateMockGlbBytes(2, 170.0);

  const rowsVi = AlphaPlusPro.extract16GirthRows(lvl2.bodyScan, 'vi');
  const rowsEn = AlphaPlusPro.extract16GirthRows(lvl2.bodyScan, 'en');
  assert.equal(rowsVi.length, 16);
  assert.equal(rowsEn.length, 16);
  assert.equal(rowsVi[0].key, 'neck');
  assert.equal(rowsVi[1].key, 'chest');
  assert.equal(rowsVi[2].key, 'waist');
  assert.equal(rowsVi[3].key, 'hip');
  assert.equal(rowsVi[15].key, 'ankle_r');

  // Normal client: fills waist/neck/hip with source='plus_pro'
  const normalClient = { sex: 'male', height: 170, waist: null, neck: null, hip: null };
  const res1 = AlphaPlusPro.applyScanResultToClient(normalClient, lvl2.bodyScan, false);
  assert.equal(res1.keptInBody, false);
  assert.equal(normalClient.girth_src.waist, 'plus_pro');
  assert.equal(normalClient.girth_src.neck, 'plus_pro');
  assert.equal(normalClient.girth_src.hip, 'plus_pro');
  assert.ok(normalClient.waist > 50);

  // InBody client: preserves CLIENT_INBODY metrics untouched!
  const inbodyClient = { sex: 'female', height: 162, waist: 66.0, neck: 31.0, hip: 91.0, girth_src: { waist: 'inbody' } };
  const res2 = AlphaPlusPro.applyScanResultToClient(inbodyClient, lvl2.bodyScan, true);
  assert.equal(res2.keptInBody, true);
  assert.equal(inbodyClient.waist, 66.0);
  assert.equal(inbodyClient.girth_src.waist, 'inbody');
});

test('frontend_lab/index.html wires Plus/PRO hooks, top-left #plus16Box, quick 4-photo button, 360 spin, and I18N.en keys', () => {
  const { AlphaPlusPro } = loadThreeAndModule();
  const htmlPath = path.join(ROOT, 'frontend_lab', 'index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');
  const modSrc = fs.readFileSync(PLUS_PRO_JS_PATH, 'utf8');

  // Script inclusion
  assert.match(html, /<script src="\.\/alpha-plus-pro\.js"><\/script>/);

  // #plus16Box placed inside #left above #metricsBox (between #cmpBox and #akTiles)
  const cmpIdx = html.indexOf('<div id="cmpBox"></div>');
  const p16Idx = html.indexOf('<div id="plus16Box"></div>');
  const tilesIdx = html.indexOf('<div class="ak-tiles" id="akTiles"');
  assert.ok(cmpIdx > 0 && p16Idx > cmpIdx && tilesIdx > p16Idx, '#plus16Box must sit top-left in #left between #cmpBox and #akTiles');

  // Entry buttons (#be_m_plus inside body edit modal & #ppQuickBtn in center toolbar)
  assert.match(html, /id="be_m_plus"/);
  assert.match(html, /id="ppQuickBtn"/);

  // realPivot + 360 default spin wiring
  assert.match(html, /realPivot/);
  assert.match(html, /AlphaPlusPro\.isReal3DActive\(\)/);
  assert.match(html, /AlphaPlusPro\.resetSession\(\)/);
  assert.match(html, /AlphaPlusPro\.renderReport16Html/);
  assert.match(html, /src==='plus_pro'/);

  // Every AlphaPlusPro.<method> called in index.html must exist on AlphaPlusPro
  const calledMethods = new Set();
  for (const m of html.matchAll(/AlphaPlusPro\.([A-Za-z0-9_]+)\s*\(/g)) {
    calledMethods.add(m[1]);
  }
  assert.ok(calledMethods.size >= 6);
  for (const fnName of calledMethods) {
    assert.equal(typeof AlphaPlusPro[fnName], 'function', `Missing AlphaPlusPro.${fnName}`);
  }

  // Every pp_* i18n key referenced in alpha-plus-pro.js has an EN translation in I18N.en
  const neededKeys = new Set();
  for (const m of modSrc.matchAll(/_t\(\s*['"](pp_[a-z0-9_]+)['"]\s*,/g)) {
    neededKeys.add(m[1]);
  }
  [
    'pp_lvl_short_1', 'pp_lvl_short_2', 'pp_lvl_short_3',
    'pp_rmode_short_dual_layer', 'pp_rmode_short_hi_res_head', 'pp_rmode_short_calibrated_mhr',
    'pp_audit_status_matched', 'pp_audit_status_estimated_only', 'pp_audit_status_mismatch_warning',
    'pp_quick_btn', 'pp_scene_hint',
  ].forEach((k) => neededKeys.add(k));
  for (const key of neededKeys) {
    assert.ok(new RegExp(`\\b${key}\\s*:`).test(html), `Missing I18N.en key: ${key}`);
  }
});

function buildTwoPrimitiveMultiMaterialGlbBuffer() {
  // Valid glTF 2.0 binary with 2 primitives (body + head), 2 materials, 2 textures, 2 embedded JPEGs
  const fakeJpegBody = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0xff, 0xd9]);
  const fakeJpegHead = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x02, 0xff, 0xd9]);

  const posBody = new Float32Array([0, -0.8, 0, 0.2, -0.8, 0, 0, 0.4, 0]);
  const normBody = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvBody = new Float32Array([0, 1, 1, 1, 0, 0.2]);
  const idxBody = new Uint16Array([0, 1, 2]);

  const posHead = new Float32Array([-0.1, 0.4, 0, 0.1, 0.4, 0, 0, 0.85, 0]);
  const normHead = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvHead = new Float32Array([0, 1, 1, 1, 0.5, 0]);
  const idxHead = new Uint16Array([0, 1, 2]);

  const chunks = [];
  const bufferViews = [];
  let binLen = 0;
  function pushView(bytes, target) {
    const pad = (4 - (binLen % 4)) % 4;
    if (pad) {
      chunks.push(Buffer.alloc(pad, 0));
      binLen += pad;
    }
    const offset = binLen;
    const buf = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    chunks.push(buf);
    binLen += buf.length;
    const bv = { buffer: 0, byteOffset: offset, byteLength: buf.length };
    if (target !== undefined) bv.target = target;
    bufferViews.push(bv);
    return bufferViews.length - 1;
  }

  const bvPosB = pushView(posBody, 34962);
  const bvNormB = pushView(normBody, 34962);
  const bvUvB = pushView(uvBody, 34962);
  const bvIdxB = pushView(idxBody, 34963);

  const bvPosH = pushView(posHead, 34962);
  const bvNormH = pushView(normHead, 34962);
  const bvUvH = pushView(uvHead, 34962);
  const bvIdxH = pushView(idxHead, 34963);

  const bvImgB = pushView(fakeJpegBody);
  const bvImgH = pushView(fakeJpegHead);
  const tailPad = (4 - (binLen % 4)) % 4;
  if (tailPad) {
    chunks.push(Buffer.alloc(tailPad, 0));
    binLen += tailPad;
  }

  const gltf = {
    asset: { version: '2.0', generator: 'test_multi_material' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'DualPartNode' }],
    meshes: [{
      name: 'DualPartMesh',
      primitives: [
        { attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 },
        { attributes: { POSITION: 4, NORMAL: 5, TEXCOORD_0: 6 }, indices: 7, material: 1 },
      ],
    }],
    materials: [
      { name: 'body', pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], baseColorTexture: { index: 0 } } },
      { name: 'head', pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], baseColorTexture: { index: 1 } } },
    ],
    textures: [{ source: 0 }, { source: 1 }],
    images: [
      { bufferView: 8, mimeType: 'image/jpeg' },
      { bufferView: 9, mimeType: 'image/jpeg' },
    ],
    accessors: [
      { bufferView: bvPosB, componentType: 5126, count: 3, type: 'VEC3', min: [0, -0.8, 0], max: [0.2, 0.4, 0] },
      { bufferView: bvNormB, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: bvUvB, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: bvIdxB, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: bvPosH, componentType: 5126, count: 3, type: 'VEC3', min: [-0.1, 0.4, 0], max: [0.1, 0.85, 0] },
      { bufferView: bvNormH, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: bvUvH, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: bvIdxH, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    bufferViews,
    buffers: [{ byteLength: binLen }],
  };

  const rawJson = Buffer.from(JSON.stringify(gltf), 'utf8');
  const padJsonLen = (rawJson.length + 3) & ~3;
  const jsonChunk = Buffer.alloc(padJsonLen, 0x20);
  rawJson.copy(jsonChunk);
  const binChunk = Buffer.concat(chunks);
  const out = Buffer.alloc(12 + 8 + padJsonLen + 8 + binChunk.length);
  out.writeUInt32LE(0x46546c67, 0);
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(out.length, 8);
  out.writeUInt32LE(padJsonLen, 12);
  out.writeUInt32LE(0x4e4f534a, 16);
  jsonChunk.copy(out, 20);
  const bOff = 20 + padJsonLen;
  out.writeUInt32LE(binChunk.length, bOff);
  out.writeUInt32LE(0x004e4942, bOff + 4);
  binChunk.copy(out, bOff + 8);
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
}

function generateMultiMaterialGlbViaPythonOrFallback() {
  const pyCheck = [
    'import base64, io, os, tempfile',
    'import numpy as np',
    'from PIL import Image',
    'from bodyscan_pipeline.scan_service import write_multi_material_glb',
    'b1, b2 = io.BytesIO(), io.BytesIO()',
    'Image.new("RGB", (16, 16), (200, 120, 90)).save(b1, format="JPEG")',
    'Image.new("RGB", (16, 16), (60, 170, 90)).save(b2, format="JPEG")',
    'v_b = np.array([[0, -0.8, 0], [0.2, -0.8, 0], [0, 0.4, 0]], dtype=np.float32)',
    'v_h = np.array([[-0.1, 0.4, 0], [0.1, 0.4, 0], [0, 0.85, 0]], dtype=np.float32)',
    'f = np.array([[0, 1, 2]], dtype=np.int32)',
    'n = np.array([[0, 0, 1]] * 3, dtype=np.float32)',
    'uv = np.array([[0, 1], [1, 1], [0.5, 0]], dtype=np.float32)',
    'with tempfile.TemporaryDirectory() as tmp:',
    '    out_p = os.path.join(tmp, "multi.glb")',
    '    write_multi_material_glb([',
    '        {"verts": v_b, "faces": f, "normals": n, "uvs": uv, "texture_jpeg_bytes": b1.getvalue(), "name": "body"},',
    '        {"verts": v_h, "faces": f, "normals": n, "uvs": uv, "texture_jpeg_bytes": b2.getvalue(), "name": "head"},',
    '    ], out_p)',
    '    with open(out_p, "rb") as fp:',
    '        print(base64.b64encode(fp.read()).decode("ascii"))',
  ].join('\n');
  try {
    const b64 = execFileSync('python', ['-c', pyCheck], { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
    const buf = Buffer.from(b64, 'base64');
    return { source: 'python_write_multi_material_glb', arrayBuffer: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
  } catch (_) {
    return { source: 'js_2primitive_fallback', arrayBuffer: buildTwoPrimitiveMultiMaterialGlbBuffer() };
  }
}

test('validatePrivacyAndConnection enforces 4 cardinal slots for hi_res_head and dual_layer', () => {
  const { AlphaPlusPro } = loadThreeAndModule();
  const base = {
    consentMode: 'save',
    serverUrl: 'http://127.0.0.1:8000',
    privacyLevel: 2,
    consentChecked: true,
    heightCm: 170,
  };

  for (const mode of ['dual_layer', 'hi_res_head']) {
    const blocked3 = AlphaPlusPro.validatePrivacyAndConnection({
      ...base,
      photoCount: 3,
      cardinalSlotsCount: 3,
      reconstructionMode: mode,
    });
    assert.equal(blocked3.ok, false);
    assert.equal(blocked3.reasonKey, 'pp_err_4cardinal');

    // Even if 4 total photos are supplied (e.g. 3 cardinal + 1 extra), <4 cardinal slots must block
    const blockedMixed = AlphaPlusPro.validatePrivacyAndConnection({
      ...base,
      photoCount: 4,
      cardinalSlotsCount: 3,
      reconstructionMode: mode,
    });
    assert.equal(blockedMixed.ok, false);
    assert.equal(blockedMixed.reasonKey, 'pp_err_4cardinal');

    const ok4 = AlphaPlusPro.validatePrivacyAndConnection({
      ...base,
      photoCount: 4,
      cardinalSlotsCount: 4,
      reconstructionMode: mode,
    });
    assert.equal(ok4.ok, true);
  }

  // calibrated_mhr allows 1-8 photos even without 4 cardinal slots
  const okMhr1 = AlphaPlusPro.validatePrivacyAndConnection({
    ...base,
    photoCount: 1,
    cardinalSlotsCount: 1,
    reconstructionMode: 'calibrated_mhr',
  });
  assert.equal(okMhr1.ok, true);

  const okMhrExtraOnly = AlphaPlusPro.validatePrivacyAndConnection({
    ...base,
    photoCount: 2,
    cardinalSlotsCount: 0,
    reconstructionMode: 'calibrated_mhr',
  });
  assert.equal(okMhrExtraOnly.ok, true);
});

test('parseGlbBinary parses multi-material body+head GLB and disposes all GPU resources', async () => {
  const { THREE, AlphaPlusPro } = loadThreeAndModule();
  const { source, arrayBuffer } = generateMultiMaterialGlbViaPythonOrFallback();
  assert.equal(source, 'python_write_multi_material_glb');
  const loadedImages = [];
  const fakeImageFactory = async (bytes, mimeType) => {
    loadedImages.push({ size: bytes.byteLength, mimeType });
    return { width: 256, height: 256 };
  };

  const parsed = await AlphaPlusPro.parseGlbBinary(arrayBuffer, THREE, fakeImageFactory);
  assert.equal(parsed.stats.meshes, 2);
  assert.equal(parsed.stats.textures, 2);
  assert.deepEqual(Array.from(parsed.stats.flipYs), [false, false]);
  assert.equal(loadedImages.length, 2);

  let disposedGeo = 0;
  let disposedMat = 0;
  let disposedTex = 0;
  const meshes = [];
  parsed.group.traverse((obj) => {
    if (obj.isMesh) {
      meshes.push(obj);
      const origG = obj.geometry.dispose.bind(obj.geometry);
      obj.geometry.dispose = () => { disposedGeo++; origG(); };
      const origM = obj.material.dispose.bind(obj.material);
      obj.material.dispose = () => { disposedMat++; origM(); };
      if (obj.material.map) {
        assert.equal(obj.material.map.flipY, false);
        assert.equal(obj.material.map.encoding, THREE.sRGBEncoding);
        const origT = obj.material.map.dispose.bind(obj.material.map);
        obj.material.map.dispose = () => { disposedTex++; origT(); };
      }
    }
  });
  assert.equal(meshes.length, 2);

  AlphaPlusPro.disposeObject3D(parsed.group);
  assert.equal(disposedGeo, 2);
  assert.equal(disposedMat, 2);
  assert.equal(disposedTex, 2);

  const dualMock = generateMockGlbBytes(2, 166.0, 'dual_layer');
  assert.ok(dualMock.coreArrayBuffer, 'MockScanEngine dual_layer should output core_glb');
  const dualParsed = await AlphaPlusPro.parseGlbBinary(dualMock.arrayBuffer, THREE, async () => ({ width: 128, height: 128 }));
  assert.equal(dualParsed.stats.meshes, 2);
  assert.equal(dualParsed.stats.textures, 2);
  assert.deepEqual(Array.from(dualParsed.stats.flipYs), [false, false]);
  AlphaPlusPro.disposeObject3D(dualParsed.group);

  const coreParsed = await AlphaPlusPro.parseGlbBinary(dualMock.coreArrayBuffer, THREE, fakeImageFactory);
  assert.equal(coreParsed.stats.meshes, 1);
  assert.equal(coreParsed.stats.textures, 0);
  AlphaPlusPro.disposeObject3D(coreParsed.group);
});

test('extractAuditSummary and renderDetailHtml render engine pill, reconstruction_mode, proportion_audit, warnings, and outer/core toggle', () => {
  const { AlphaPlusPro } = loadThreeAndModule();

  const gpuPayload = {
    mode: 'gpu',
    test_only: false,
    reconstruction_mode: 'dual_layer',
    effective_reconstruction_mode: 'dual_layer',
    privacy_level: 2,
    warnings: [],
    artifacts: {
      mesh_url: '/api/v1/bodyscan/j1/files/scan_preview_blur.glb?exp=999&sig=abc',
      core_mesh_url: '/api/v1/bodyscan/j1/files/scan_core.glb?exp=999&sig=def',
      json_url: '/api/v1/bodyscan/j1/files/body_scan.json?exp=999&sig=ghi',
    },
    proportion_audit: {
      declared_weight_kg: 58.0,
      estimated_weight_kg: 57.2,
      weight_diff_pct: 1.4,
      mhr_silhouette_iou: 0.89,
      outer_silhouette_iou: 0.92,
      status: 'matched',
      warnings: [],
    },
    body_scan: { circumferences_cm: { neck: 33, chest: 88, waist: 68, hip: 94 } },
    deletion_audit: { deleted_count: 4 },
  };

  const gpuSummary = AlphaPlusPro.extractAuditSummary(gpuPayload);
  assert.equal(gpuSummary.engineMode, 'gpu');
  assert.equal(gpuSummary.reconstructionMode, 'dual_layer');
  assert.equal(gpuSummary.effectiveReconstructionMode, 'dual_layer');
  assert.equal(gpuSummary.wasDowngraded, false);
  assert.equal(gpuSummary.hasCoreMesh, true);
  assert.equal(gpuSummary.proportionAudit.status, 'matched');

  const gpuHtmlVi = AlphaPlusPro.renderDetailHtml({
    ...gpuPayload,
    realActive: true,
    hasLoadedMesh: true,
    meshViewLayer: 'outer',
  }, 'vi');
  assert.match(gpuHtmlVi, /GPU thật/);
  assert.match(gpuHtmlVi, /dual_layer/);
  assert.match(gpuHtmlVi, /57\.2\s*kg/);
  assert.match(gpuHtmlVi, /58\.0\s*kg/);
  assert.match(gpuHtmlVi, /89\.0%/);
  assert.match(gpuHtmlVi, /92\.0%/);
  assert.match(gpuHtmlVi, /id="ppToggleCoreBtn"/);
  assert.match(gpuHtmlVi, /Xem lõi đo MHR/);

  const mockPayload = {
    mode: 'mock',
    test_only: true,
    reconstruction_mode: 'dual_layer',
    effective_reconstruction_mode: 'calibrated_mhr',
    privacy_level: 1,
    warnings: ['<4 cardinal views; downgraded to calibrated_mhr', 'Weight differs by >12%'],
    artifacts: {
      mesh_url: '/api/v1/bodyscan/j2/files/scan.glb?exp=999&sig=abc',
      json_url: '/api/v1/bodyscan/j2/files/body_scan.json?exp=999&sig=ghi',
    },
    proportion_audit: {
      declared_weight_kg: 80.0,
      estimated_weight_kg: 62.5,
      weight_diff_pct: 21.9,
      mhr_silhouette_iou: 0.72,
      outer_silhouette_iou: null,
      status: 'mismatch_warning',
      warnings: ['Weight differs by >12%'],
    },
    body_scan: { circumferences_cm: { neck: 33, chest: 88, waist: 68, hip: 94 } },
  };

  const mockSummary = AlphaPlusPro.extractAuditSummary(mockPayload);
  assert.equal(mockSummary.engineMode, 'mock');
  assert.equal(mockSummary.wasDowngraded, true);
  assert.equal(mockSummary.hasCoreMesh, false);
  assert.equal(mockSummary.warnings.length, 2);

  const mockHtmlEn = AlphaPlusPro.renderDetailHtml({
    ...mockPayload,
    realActive: true,
    hasLoadedMesh: true,
  }, 'en');
  assert.match(mockHtmlEn, /MOCK TEST/);
  assert.match(mockHtmlEn, /dual_layer/);
  assert.match(mockHtmlEn, /calibrated_mhr/);
  assert.match(mockHtmlEn, /62\.5\s*kg/);
  assert.match(mockHtmlEn, /72\.0%/);
  assert.match(mockHtmlEn, /&lt;4 cardinal views; downgraded to calibrated_mhr/);
  assert.doesNotMatch(mockHtmlEn, /id="ppToggleCoreBtn"/);

  const estHtmlEn = AlphaPlusPro.renderDetailHtml({
    bodyScan: { circumferences_cm: {} },
    proportionAudit: { status: 'estimated_only', estimated_weight_kg: 65.0 },
    hasLoadedMesh: true,
    realActive: true,
  }, 'en');
  assert.match(estHtmlEn, /data-audit-status="estimated_only"/);
  assert.match(estHtmlEn, /Estimated only/);

  // Non-contract status falls back to estimated_only (never minor_offset)
  const fallbackHtmlEn = AlphaPlusPro.renderDetailHtml({
    bodyScan: { circumferences_cm: {} },
    proportionAudit: { status: 'minor_offset', estimated_weight_kg: 65.0 },
    hasLoadedMesh: true,
    realActive: true,
  }, 'en');
  assert.match(fallbackHtmlEn, /data-audit-status="estimated_only"/);
  assert.match(fallbackHtmlEn, /Estimated only/);
});

test('toggleCoreMeshView switches realPivot cleanly with disposeObject3D and resetSession cleans state', async () => {
  const { THREE, AlphaPlusPro } = loadThreeAndModule();
  const dualMock = generateMockGlbBytes(2, 170.0, 'dual_layer');
  assert.ok(dualMock.coreArrayBuffer, 'MockScanEngine dual_layer should emit scan_core.glb');
  const outerBuf = dualMock.arrayBuffer;
  const coreBuf = dualMock.coreArrayBuffer;
  const fakeImageFactory = async () => ({ width: 64, height: 64 });

  const realPivot = new THREE.Group();
  let detachedCount = 0;
  let attachedCount = 0;
  const client = { id: 'c1', name: 'Test', sex: 'female', height: 170 };

  AlphaPlusPro.init({
    getClient: () => client,
    getConsentMode: () => 'save',
    isInBodyClient: () => false,
    detachRealGroup: (grp) => {
      detachedCount++;
      realPivot.remove(grp);
    },
    attachRealGroup: (grp) => {
      attachedCount++;
      while (realPivot.children.length) realPivot.remove(realPivot.children[0]);
      realPivot.add(grp);
    },
    onScanCompleted: (grp) => {
      attachedCount++;
      while (realPivot.children.length) realPivot.remove(realPivot.children[0]);
      realPivot.add(grp);
    },
    refreshAppearance: () => {},
    getLang: () => 'vi',
    t: (k, fb) => fb,
  });

  const fetchMock = async (url) => {
    if (String(url).includes('scan_core.glb')) {
      return { ok: true, status: 200, arrayBuffer: async () => coreBuf };
    }
    return { ok: true, status: 200, arrayBuffer: async () => outerBuf };
  };

  await AlphaPlusPro.applyCompletedPayload('http://127.0.0.1:8000', {
    mode: 'gpu',
    test_only: false,
    reconstruction_mode: 'dual_layer',
    effective_reconstruction_mode: 'dual_layer',
    privacy_level: 2,
    warnings: [],
    proportion_audit: {
      declared_weight_kg: 58.0,
      estimated_weight_kg: 57.5,
      weight_diff_pct: 0.9,
      mhr_silhouette_iou: 0.91,
      outer_silhouette_iou: 0.93,
      status: 'matched',
      warnings: [],
    },
    artifacts: {
      mesh_url: '/api/v1/bodyscan/j9/files/scan_preview_blur.glb?exp=999&sig=1',
      core_mesh_url: '/api/v1/bodyscan/j9/files/scan_core.glb?exp=999&sig=2',
      json_url: '/api/v1/bodyscan/j9/files/body_scan.json?exp=999&sig=3',
    },
    body_scan: { circumferences_cm: { neck: 33, chest: 88, waist: 68, hip: 94 } },
    deletion_audit: { deleted_count: 4 },
  }, { fetchFn: fetchMock, imageFactory: fakeImageFactory });

  assert.equal(realPivot.children.length, 1);
  const detailStub = { innerHTML: '' };
  assert.equal(AlphaPlusPro.renderDetailCard(detailStub), true);
  assert.match(detailStub.innerHTML, /GPU thật/);
  assert.match(detailStub.innerHTML, /Xem lõi đo MHR/);

  const outerGroup = realPivot.children[0];
  let outerDisposedGeo = 0;
  let outerDisposedMat = 0;
  let outerDisposedTex = 0;
  outerGroup.traverse((obj) => {
    if (obj.isMesh) {
      const og = obj.geometry.dispose.bind(obj.geometry);
      obj.geometry.dispose = () => { outerDisposedGeo++; og(); };
      const om = obj.material.dispose.bind(obj.material);
      obj.material.dispose = () => { outerDisposedMat++; om(); };
      if (obj.material.map) {
        const ot = obj.material.map.dispose.bind(obj.material.map);
        obj.material.map.dispose = () => { outerDisposedTex++; ot(); };
      }
    }
  });

  // Switch outer -> core
  const layerAfterCore = await AlphaPlusPro.toggleCoreMeshView({ fetchFn: fetchMock, imageFactory: fakeImageFactory });
  assert.equal(layerAfterCore, 'core');
  assert.equal(outerDisposedGeo, 2);
  assert.equal(outerDisposedMat, 2);
  assert.equal(outerDisposedTex, 2);
  assert.equal(realPivot.children.length, 1);
  assert.notEqual(realPivot.children[0], outerGroup);
  assert.equal(AlphaPlusPro.getStateSnapshot().meshViewLayer, 'core');
  assert.equal(AlphaPlusPro.renderDetailCard(detailStub), true);
  assert.match(detailStub.innerHTML, /Xem lớp ngoài thật/);

  const coreGroup = realPivot.children[0];
  let coreDisposedGeo = 0;
  coreGroup.traverse((obj) => {
    if (obj.isMesh) {
      const cg = obj.geometry.dispose.bind(obj.geometry);
      obj.geometry.dispose = () => { coreDisposedGeo++; cg(); };
    }
  });

  // Switch core -> outer
  const layerAfterOuter = await AlphaPlusPro.toggleCoreMeshView({ fetchFn: fetchMock, imageFactory: fakeImageFactory });
  assert.equal(layerAfterOuter, 'outer');
  assert.equal(coreDisposedGeo, 1);
  assert.equal(AlphaPlusPro.getStateSnapshot().meshViewLayer, 'outer');
  assert.ok(detachedCount >= 2);
  assert.ok(attachedCount >= 3);

  // resetSession clears active mesh, audit, warnings, layer, and defaults mode back to dual_layer
  AlphaPlusPro.resetSession();
  const snap = AlphaPlusPro.getStateSnapshot();
  assert.equal(snap.bodyScan, null);
  assert.equal(snap.proportionAudit, null);
  assert.deepEqual(Array.from(snap.warnings), []);
  assert.equal(snap.meshViewLayer, 'outer');
  assert.equal(snap.reconstructionMode, 'dual_layer');
  assert.equal(realPivot.children.length, 0);
});

test('#plusModal renders capture-protocol banner, reconstruction_mode defaulting to dual_layer, and submits reconstruction_mode in FormData', async () => {
  const { AlphaPlusPro } = loadThreeAndModule();
  assert.equal(AlphaPlusPro.DEFAULT_RECONSTRUCTION_MODE, 'dual_layer');
  assert.deepEqual(Array.from(AlphaPlusPro.RECONSTRUCTION_MODES), ['calibrated_mhr', 'hi_res_head', 'dual_layer']);

  const modalHtml = AlphaPlusPro.buildModalHtml();
  assert.match(modalHtml, /Quy chuẩn: mặc đồ tập sát form · buộc tóc gọn lộ cổ–vai/);
  assert.match(modalHtml, /data-rmode="dual_layer"/);
  assert.match(modalHtml, /data-rmode="hi_res_head"/);
  assert.match(modalHtml, /data-rmode="calibrated_mhr"/);
  assert.match(modalHtml, /class="pp-rmode on"[^>]*data-rmode="dual_layer"/);
  assert.match(modalHtml, /name="ppReconstructionMode" value="dual_layer" checked/);

  const dummyBlob = new Blob([new Uint8Array([1, 2, 3])], { type: 'image/jpeg' });
  const fdDefault = AlphaPlusPro.buildScanFormData({
    photos: [dummyBlob, dummyBlob, dummyBlob, dummyBlob],
    heightCm: 168,
    sex: 'female',
    weightKg: 54,
    personId: 'p1',
    privacyLevel: 2,
  });
  assert.equal(fdDefault.get('reconstruction_mode'), 'dual_layer');
  assert.equal(fdDefault.getAll('photos').length, 4);

  const fdMhr = AlphaPlusPro.buildScanFormData({
    photos: [dummyBlob],
    heightCm: 175,
    sex: 'male',
    weightKg: 0,
    personId: 'p2',
    privacyLevel: 1,
    reconstructionMode: 'calibrated_mhr',
  });
  assert.equal(fdMhr.get('reconstruction_mode'), 'calibrated_mhr');
  assert.equal(fdMhr.get('weight_kg'), '0');

  // Verify Three.js remains 100% offline with zero external three.js CDN script tags
  const html = fs.readFileSync(path.join(ROOT, 'frontend_lab', 'index.html'), 'utf8');
  assert.doesNotMatch(html, /https?:\/\/[^"']*three(?:\.min)?\.js/i);
});

function buildMiniTestGlb(scale = 1) {
  const floats = new Float32Array([0, 0, 0, scale, 0, 0, 0, scale, 0]);
  const binRaw = new Uint8Array(floats.buffer);
  const binPad = (4 - (binRaw.length % 4)) % 4;
  const binLen = binRaw.length + binPad;

  const jsonStr = JSON.stringify({
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    buffers: [{ byteLength: binRaw.length }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: binRaw.length }],
    accessors: [{
      bufferView: 0,
      componentType: 5126,
      count: 3,
      type: 'VEC3',
      min: [0, 0, 0],
      max: [scale, scale, 0],
    }],
  });
  const enc = new TextEncoder();
  const jsonRaw = enc.encode(jsonStr);
  const jsonPad = (4 - (jsonRaw.length % 4)) % 4;
  const jsonLen = jsonRaw.length + jsonPad;

  const totalLen = 12 + 8 + jsonLen + 8 + binLen;
  const out = new Uint8Array(totalLen);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546C67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, totalLen, true);
  dv.setUint32(12, jsonLen, true);
  dv.setUint32(16, 0x4E4F534A, true);
  out.set(jsonRaw, 20);
  for (let i = 0; i < jsonPad; i += 1) out[20 + jsonRaw.length + i] = 0x20;
  const binHeaderOff = 20 + jsonLen;
  dv.setUint32(binHeaderOff, binLen, true);
  dv.setUint32(binHeaderOff + 4, 0x004E4942, true);
  out.set(binRaw, binHeaderOff + 8);
  return out;
}

function makeFakeImageFactory() {
  return function FakeImage() {
    this.onload = null;
    this.onerror = null;
  };
}

test('R07: auto-refresh signed artifact URLs (>240s), 403 single retry, and 410 stop', async () => {
  const sandbox = loadThreeAndModule();
  const pp = sandbox.AlphaPlusPro;
  pp.resetSession();
  pp.saveCfg('http://127.0.0.1:8000', 'secret-tok');

  const outerGlb = buildMiniTestGlb(1);
  const coreGlb = buildMiniTestGlb(0.8);
  const fetchCalls = [];
  let jobVersion = 1;
  let coreMode = 'ok'; // 'ok' | '403-once' | '403-always' | '410'
  let core403Count = 0;

  const mockFetch = async (url, init = {}) => {
    fetchCalls.push({ url, headers: init.headers || {} });
    if (url.includes('/api/v1/bodyscan/job-r07') && !url.includes('/artifacts/')) {
      jobVersion += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job-r07',
          status: 'done',
          artifacts: {
            mesh_url: `/api/v1/bodyscan/job-r07/artifacts/outer.glb?sig=v${jobVersion}`,
            core_mesh_url: `/api/v1/bodyscan/job-r07/artifacts/core.glb?sig=v${jobVersion}`,
            json_url: `/api/v1/bodyscan/job-r07/artifacts/summary.json?sig=v${jobVersion}`,
          },
          body_scan: { measurements_cm: { chest: 96.4, waist: 80.1, hip: 98.0 } },
        }),
      };
    }
    if (url.includes('/artifacts/outer.glb')) {
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => outerGlb.buffer.slice(0),
        blob: async () => new Blob([outerGlb]),
      };
    }
    if (url.includes('/artifacts/core.glb')) {
      if (coreMode === '403-once') {
        core403Count += 1;
        if (core403Count === 1) {
          return { ok: false, status: 403, text: async () => 'Expired signature' };
        }
      } else if (coreMode === '403-always') {
        return { ok: false, status: 403, text: async () => 'Forbidden' };
      } else if (coreMode === '410') {
        return { ok: false, status: 410, text: async () => 'Assets purged' };
      }
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => coreGlb.buffer.slice(0),
        blob: async () => new Blob([coreGlb]),
      };
    }
    throw new Error(`Unexpected fetch URL: ${url}`);
  };

  await pp.applyCompletedPayload('http://127.0.0.1:8000', {
    job_id: 'job-r07',
    status: 'done',
    artifacts: {
      mesh_url: '/api/v1/bodyscan/job-r07/artifacts/outer.glb?sig=v1',
      core_mesh_url: '/api/v1/bodyscan/job-r07/artifacts/core.glb?sig=v1',
      json_url: '/api/v1/bodyscan/job-r07/artifacts/summary.json?sig=v1',
    },
    body_scan: { measurements_cm: { chest: 96.4, waist: 80.1, hip: 98.0 } },
  }, { fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 1000 });

  assert.equal(pp.ARTIFACT_URL_REFRESH_MS, 240000);
  let snap = pp.getStateSnapshot();
  assert.equal(snap.lastJobId, 'job-r07');
  assert.equal(snap.artifactUrlsFetchedAt, 1000);

  // Advance past 240s TTL -> toggleCoreMeshView refreshes job status first with Bearer auth
  fetchCalls.length = 0;
  await pp.toggleCoreMeshView({
    fetchFn: mockFetch,
    imageFactory: makeFakeImageFactory(),
    now: 250000,
  });

  const refreshCall = fetchCalls.find((c) => c.url === 'http://127.0.0.1:8000/api/v1/bodyscan/job-r07');
  assert.ok(refreshCall, 'Should refresh GET /api/v1/bodyscan/job-r07 after >= 240s');
  assert.equal(refreshCall.headers.Authorization, 'Bearer secret-tok');
  const coreCall = fetchCalls.find((c) => c.url.includes('/artifacts/core.glb?sig=v2'));
  assert.ok(coreCall, 'Should fetch newly refreshed core_mesh_url sig=v2');
  const summary = pp.extractAuditSummary();
  assert.match(summary.jsonUrl, /sig=v2/);
  assert.equal(pp.getStateSnapshot().artifactUrlsFetchedAt, 250000);

  // Test 1 retry on 403 -> refresh -> succeed
  pp.resetSession();
  pp.saveCfg('http://127.0.0.1:8000', 'secret-tok');
  coreMode = '403-once';
  core403Count = 0;
  await pp.applyCompletedPayload('http://127.0.0.1:8000', {
    job_id: 'job-r07',
    status: 'done',
    artifacts: {
      mesh_url: '/api/v1/bodyscan/job-r07/artifacts/outer.glb?sig=v1',
      core_mesh_url: '/api/v1/bodyscan/job-r07/artifacts/core.glb?sig=v1',
    },
    body_scan: { measurements_cm: { chest: 96.4, waist: 80.1, hip: 98.0 } },
  }, { fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 1000 });

  fetchCalls.length = 0;
  const retryLayer = await pp.toggleCoreMeshView({
    fetchFn: mockFetch,
    imageFactory: makeFakeImageFactory(),
    now: 2000, // fresh (<240s), so 403 triggers single refresh + retry
  });
  assert.equal(retryLayer, 'core');
  assert.equal(fetchCalls.filter((c) => c.url === 'http://127.0.0.1:8000/api/v1/bodyscan/job-r07').length, 1);
  assert.equal(fetchCalls.filter((c) => c.url.includes('/artifacts/core.glb')).length, 2);

  // Permanent 403 must stop after 1 retry (2 core fetches total, not looping)
  pp.resetSession();
  pp.saveCfg('http://127.0.0.1:8000', 'secret-tok');
  coreMode = '403-always';
  await pp.applyCompletedPayload('http://127.0.0.1:8000', {
    job_id: 'job-r07',
    status: 'done',
    artifacts: {
      mesh_url: '/api/v1/bodyscan/job-r07/artifacts/outer.glb?sig=v1',
      core_mesh_url: '/api/v1/bodyscan/job-r07/artifacts/core.glb?sig=v1',
    },
    body_scan: { measurements_cm: { chest: 96.4 } },
  }, { fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 1000 });
  fetchCalls.length = 0;
  await assert.rejects(() => pp.toggleCoreMeshView({
    fetchFn: mockFetch,
    imageFactory: makeFakeImageFactory(),
    now: 2000,
  }), /403/);
  assert.equal(fetchCalls.filter((c) => c.url.includes('/artifacts/core.glb')).length, 2);

  // 410 Gone marks assetsDeleted=true, keeps measurements, clears mesh, and blocks future 403 retries
  pp.resetSession();
  pp.saveCfg('http://127.0.0.1:8000', 'secret-tok');
  coreMode = '410';
  await pp.applyCompletedPayload('http://127.0.0.1:8000', {
    job_id: 'job-r07',
    status: 'done',
    artifacts: {
      mesh_url: '/api/v1/bodyscan/job-r07/artifacts/outer.glb?sig=v1',
      core_mesh_url: '/api/v1/bodyscan/job-r07/artifacts/core.glb?sig=v1',
    },
    body_scan: { measurements_cm: { chest: 96.4, waist: 80.1 } },
  }, { fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 1000 });
  fetchCalls.length = 0;
  await assert.rejects(() => pp.toggleCoreMeshView({
    fetchFn: mockFetch,
    imageFactory: makeFakeImageFactory(),
    now: 2000,
  }), /410/);
  snap = pp.getStateSnapshot();
  assert.equal(snap.assetsDeleted, true);
  assert.equal(pp.hasMesh(), false);
  assert.equal(pp.getResult().measurements_cm.chest, 96.4);
  assert.equal(fetchCalls.filter((c) => c.url === 'http://127.0.0.1:8000/api/v1/bodyscan/job-r07').length, 0);

  // Subsequent download when assetsDeleted === true does not refresh or fetch
  coreMode = '403-always';
  fetchCalls.length = 0;
  await assert.rejects(() => pp.downloadArtifact('glb', {
    fetchFn: mockFetch,
    now: 300000,
  }));
  assert.equal(fetchCalls.length, 0);
});

test('Medium: Dynamic Tải .glb URL & filename switches between alpha_scan.glb and alpha_scan_core.glb with click refresh', async () => {
  const sandbox = loadThreeAndModule();
  const pp = sandbox.AlphaPlusPro;

  const domMap = {
    ppDlGlb: {
      tagName: 'A',
      href: '',
      download: '',
      onclick: null,
      style: {},
      setAttribute(k, v) { this[k] = String(v); },
      addEventListener() {},
    },
    ppDlJson: {
      tagName: 'A',
      href: '',
      download: '',
      onclick: null,
      style: {},
      setAttribute(k, v) { this[k] = String(v); },
      addEventListener() {},
    },
    ppToggleCoreBtn: { textContent: '', disabled: false, style: {}, onclick: null, addEventListener() {} },
    ppMeshLayerBadge: { textContent: '', style: {} },
    ppDelStatus: { textContent: '', style: {} },
  };
  const clickedAnchors = [];
  const detailEl = { innerHTML: '', style: {} };
  sandbox.document.body = { appendChild() {}, removeChild() {} };
  sandbox.document.createElement = (tag) => {
    const el = {
      tagName: String(tag).toUpperCase(),
      href: '',
      download: '',
      style: {},
      addEventListener() {},
      click() { clickedAnchors.push({ download: this.download, href: this.href }); },
    };
    return el;
  };
  sandbox.document.getElementById = (id) => domMap[id] || null;
  if (typeof sandbox.URL.createObjectURL !== 'function') {
    sandbox.URL.createObjectURL = () => 'blob:mock-download';
    sandbox.URL.revokeObjectURL = () => {};
  }

  pp.resetSession();
  pp.saveCfg('http://127.0.0.1:8000', 'tok-dl');
  const outerGlb = buildMiniTestGlb(1);
  const coreGlb = buildMiniTestGlb(0.75);
  let sigCounter = 1;
  const requests = [];
  const downloads = [];

  const mockFetch = async (url, init = {}) => {
    requests.push({ url, headers: init.headers || {} });
    if (url.includes('/api/v1/bodyscan/job-dl') && !url.includes('/artifacts/')) {
      sigCounter += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job-dl',
          status: 'done',
          artifacts: {
            mesh_url: `/api/v1/bodyscan/job-dl/artifacts/outer.glb?sig=s${sigCounter}`,
            core_mesh_url: `/api/v1/bodyscan/job-dl/artifacts/core.glb?sig=s${sigCounter}`,
            json_url: `/api/v1/bodyscan/job-dl/artifacts/body.json?sig=s${sigCounter}`,
          },
          body_scan: { measurements_cm: { chest: 95.0, waist: 79.0 } },
        }),
      };
    }
    if (url.includes('/artifacts/outer.glb')) {
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => outerGlb.buffer.slice(0),
        blob: async () => new Blob([outerGlb]),
      };
    }
    if (url.includes('/artifacts/core.glb')) {
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => coreGlb.buffer.slice(0),
        blob: async () => new Blob([coreGlb]),
      };
    }
    if (url.includes('/artifacts/body.json')) {
      return {
        ok: true,
        status: 200,
        blob: async () => new Blob([JSON.stringify({ ok: true })]),
      };
    }
    throw new Error(`Unexpected URL: ${url}`);
  };
  sandbox.fetch = mockFetch;

  await pp.applyCompletedPayload('http://127.0.0.1:8000', {
    job_id: 'job-dl',
    status: 'done',
    artifacts: {
      mesh_url: '/api/v1/bodyscan/job-dl/artifacts/outer.glb?sig=s1',
      core_mesh_url: '/api/v1/bodyscan/job-dl/artifacts/core.glb?sig=s1',
      json_url: '/api/v1/bodyscan/job-dl/artifacts/body.json?sig=s1',
    },
    body_scan: { measurements_cm: { chest: 95.0, waist: 79.0 } },
  }, { fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 1000 });

  assert.equal(pp.renderDetailCard(detailEl), true);
  assert.match(detailEl.innerHTML, /id="ppDlGlb"/);
  assert.match(detailEl.innerHTML, /download="alpha_scan\.glb"/);
  assert.equal(domMap.ppDlGlb.download, 'alpha_scan.glb');
  assert.match(domMap.ppDlGlb.href, /outer\.glb\?sig=s1/);
  assert.equal(pp.getActiveGlbDownloadTarget().filename, 'alpha_scan.glb');

  // Toggle to core layer -> #ppDlGlb updates immediately to core_mesh_url & alpha_scan_core.glb
  await pp.toggleCoreMeshView({ fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 2000 });
  assert.equal(pp.getStateSnapshot().meshViewLayer, 'core');
  assert.equal(domMap.ppDlGlb.download, 'alpha_scan_core.glb');
  assert.match(domMap.ppDlGlb.href, /core\.glb\?sig=s1/);
  const coreDetailHtml = pp.renderDetailHtml();
  assert.match(coreDetailHtml, /download="alpha_scan_core\.glb"/);
  assert.match(coreDetailHtml, /core\.glb\?sig=s1/);

  // Clicking/invoking downloadArtifact('glb') after >240s refreshes URLs first and downloads alpha_scan_core.glb
  requests.length = 0;
  const dlCore = await pp.downloadArtifact('glb', {
    fetchFn: mockFetch,
    now: 260000,
    triggerDownload: ({ blob, filename }) => downloads.push({ filename, size: blob.size }),
  });
  assert.equal(dlCore.filename, 'alpha_scan_core.glb');
  assert.match(dlCore.url, /core\.glb\?sig=s2/);
  assert.equal(downloads[0].filename, 'alpha_scan_core.glb');
  assert.ok(requests.some((r) => r.url === 'http://127.0.0.1:8000/api/v1/bodyscan/job-dl'));

  // Toggle back to outer -> downloads alpha_scan.glb; test #ppDlGlb.onclick triggers browser anchor download
  await pp.toggleCoreMeshView({ fetchFn: mockFetch, imageFactory: makeFakeImageFactory(), now: 261000 });
  assert.equal(domMap.ppDlGlb.download, 'alpha_scan.glb');
  assert.match(domMap.ppDlGlb.href, /outer\.glb\?sig=s2/);
  let prevented = false;
  await domMap.ppDlGlb.onclick({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(clickedAnchors[clickedAnchors.length - 1].download, 'alpha_scan.glb');

  // downloadArtifact('json') refreshes after >240s and saves body_scan.json
  const dlJson = await pp.downloadArtifact('json', {
    fetchFn: mockFetch,
    now: 520000,
    triggerDownload: ({ blob, filename }) => downloads.push({ filename, size: blob.size }),
  });
  assert.equal(dlJson.filename, 'body_scan.json');
  assert.match(dlJson.url, /body\.json\?sig=s3/);
});

test('Low & Medium: I18N.en & I18N.vi completeness and zero duplicate be_plus', () => {
  const html = fs.readFileSync(path.join(ROOT, 'frontend_lab', 'index.html'), 'utf8');
  const match = html.match(/const I18N\s*=\s*\{/);
  const tIdx = html.indexOf('function t(key, vi)');
  assert.ok(match && tIdx > match.index, 'I18N block should be present in frontend_lab/index.html');
  const i18nSnippet = html.slice(match.index, tIdx);

  // Inspect raw `en: { ... }` literal inside I18N so duplicate object keys cannot slip past JS evaluation
  const enMatch = i18nSnippet.match(/\ben\s*:\s*\{/);
  assert.ok(enMatch, 'I18N.en block must exist');
  const enSlice = i18nSnippet.slice(enMatch.index);
  const bePlusOccurrences = enSlice.match(/\bbe_plus\s*:/g) || [];
  assert.equal(bePlusOccurrences.length, 1, 'I18N.en must contain be_plus exactly once (no duplicate key)');

  // Evaluate I18N + t() in vm to inspect runtime values in both languages
  const i18nCtx = { lang: 'en', I18N: null };
  vm.createContext(i18nCtx);
  vm.runInContext(
    `${i18nSnippet.replace(/const I18N\s*=/, 'I18N =')}\nfunction lookup(k, vi){ return (I18N[lang] && I18N[lang][k]) || vi; }`,
    i18nCtx
  );

  const requiredBoth = [
    'sex_m',
    'sex_f',
    'overview_title',
    'pp_dup_warn',
    'pp_health_loading',
    'pp_err_410',
    'pp_dl_err',
  ];
  assert.ok(typeof i18nCtx.I18N.en.be_plus === 'string' && i18nCtx.I18N.en.be_plus.length > 0);
  assert.match(i18nCtx.I18N.en.be_plus, /16 circumferences/i);
  assert.doesNotMatch(i18nCtx.I18N.en.be_plus, /coming soon/i);

  for (const key of requiredBoth) {
    assert.ok(typeof i18nCtx.I18N.en[key] === 'string' && i18nCtx.I18N.en[key].length > 0, `I18N.en missing ${key}`);
    assert.ok(typeof i18nCtx.I18N.vi[key] === 'string' && i18nCtx.I18N.vi[key].length > 0, `I18N.vi missing ${key}`);
  }
});

test('Medium: Batch upload role hints & duplicate warning, /health loading handling, and ngrok header', async () => {
  const sandbox = loadThreeAndModule();
  const pp = sandbox.AlphaPlusPro;
  pp.resetSession();

  // 1. Cardinal role detection from filenames & sequential fallback
  assert.equal(pp.inferCardinalRoleFromFilename('01_subject.jpg'), 0);
  assert.equal(pp.inferCardinalRoleFromFilename('anh_truoc.png'), 0);
  assert.equal(pp.inferCardinalRoleFromFilename('phai_02.jpg'), 1);
  assert.equal(pp.inferCardinalRoleFromFilename('03_back.jpg'), 2);
  assert.equal(pp.inferCardinalRoleFromFilename('left_side.jpeg'), 3);
  assert.equal(pp.inferCardinalRoleFromFilename('front_back_ambiguous.jpg'), null);
  assert.equal(pp.inferCardinalRoleFromFilename('IMG_9981.jpg'), null);

  const shuffled = [
    { name: '03_back.jpg', size: 1003 },
    { name: 'IMG_001.jpg', size: 1009 },
    { name: 'anh_trai.jpg', size: 1004 },
    { name: '01_front.jpg', size: 1001 },
  ];
  const assigned = pp.assignBatchFilesToSlots(shuffled, [null, null, null, null], []);
  assert.equal(assigned.slots[0].name, '01_front.jpg');
  assert.equal(assigned.slots[1].name, 'IMG_001.jpg'); // unambiguous fallback to remaining cardinal slot
  assert.equal(assigned.slots[2].name, '03_back.jpg');
  assert.equal(assigned.slots[3].name, 'anh_trai.jpg');
  assert.equal(assigned.duplicateInfo.hasDuplicates, false);

  // Modal UI integration: clicking #ppBatchBtn with duplicate file signatures warns in #ppDupWarn without blocking #ppSubmitBtn
  const dupAssigned = pp.assignBatchFilesToSlots([
    { name: 'same_photo.jpg', size: 2048 },
    { name: 'same_photo.jpg', size: 2048 },
    { name: '03_back.jpg', size: 1003 },
    { name: '04_left.jpg', size: 1004 },
  ], [null, null, null, null], []);
  assert.equal(dupAssigned.duplicateInfo.hasDuplicates, true);
  assert.equal(dupAssigned.duplicateInfo.duplicateNames.includes('same_photo.jpg'), true);

  function makeStubEl(id) {
    return {
      id,
      innerHTML: '',
      textContent: '',
      value: '',
      checked: false,
      disabled: false,
      title: '',
      className: '',
      style: {},
      dataset: {},
      classList: { toggle() {} },
      addEventListener() {},
      appendChild() {},
    };
  }
  const modalEls = {
    plusModal: makeStubEl('plusModal'),
    ppSlotsGrid: makeStubEl('ppSlotsGrid'),
    ppExtraCount: makeStubEl('ppExtraCount'),
    ppBatchBtn: makeStubEl('ppBatchBtn'),
    ppDupWarn: makeStubEl('ppDupWarn'),
    ppGuardMsg: makeStubEl('ppGuardMsg'),
    ppSubmitBtn: makeStubEl('ppSubmitBtn'),
    ppHeight: Object.assign(makeStubEl('ppHeight'), { value: '170' }),
    ppConsentChk: Object.assign(makeStubEl('ppConsentChk'), { checked: true }),
    ppServerUrl: Object.assign(makeStubEl('ppServerUrl'), { value: 'http://127.0.0.1:8000' }),
    ppHealthStatus: makeStubEl('ppHealthStatus'),
    akGpuHealthOut: makeStubEl('akGpuHealthOut'),
  };
  let nextBatchFiles = [];
  const privBtn2 = Object.assign(makeStubEl('privBtn2'), { dataset: { priv: '2' } });
  sandbox.document.head = { appendChild() {} };
  sandbox.document.body = { appendChild() {}, removeChild() {} };
  sandbox.document.querySelectorAll = (sel) => (sel === '#plusModal .pp-priv' ? [privBtn2] : []);
  sandbox.document.getElementById = (id) => modalEls[id] || null;
  sandbox.document.createElement = (tag) => {
    const el = makeStubEl(tag);
    if (tag === 'input') {
      el.click = () => {
        el.files = nextBatchFiles;
        if (typeof el.onchange === 'function') el.onchange();
      };
    }
    return el;
  };
  sandbox.URL.createObjectURL = () => 'blob:slot-preview';
  sandbox.URL.revokeObjectURL = () => {};

  pp.init({
    getClient: () => ({ name: 'Client A', sex: 'female', height: 170, weight: 58 }),
    getConsentMode: () => 'cloud',
    closeBodyEditModal() {},
  });
  pp.openModal();
  privBtn2.onclick();

  nextBatchFiles = [
    { name: 'same_photo.jpg', size: 2048 },
    { name: 'same_photo.jpg', size: 2048 },
    { name: '03_back.jpg', size: 1003 },
    { name: '04_left.jpg', size: 1004 },
  ];
  modalEls.ppBatchBtn.onclick();

  assert.match(modalEls.ppDupWarn.className, /\bwarn\b/);
  assert.match(modalEls.ppDupWarn.textContent, /same_photo\.jpg/);
  assert.ok(pp.getStateSnapshot().duplicateWarning, 'duplicateWarning state should be recorded');
  assert.equal(modalEls.ppSubmitBtn.disabled, false, 'duplicate warning must be non-blocking when 4 valid slots + consent + URL exist');

  // 2. /health loading state handling (status === 'loading' or model_ready === false)
  pp.saveCfg('https://demo.ngrok-free.app', 'ng-key');
  const seenHeaders = [];
  const loadingHealth = await pp.onHealthCheck('https://demo.ngrok-free.app', 'ng-key', {
    fetchFn: async (url, init = {}) => {
      seenHeaders.push({ url, headers: init.headers || {} });
      return {
        ok: true,
        status: 200,
        json: async () => ({ ok: true, status: 'loading', model_ready: false, mode: 'gpu' }),
      };
    },
  });
  assert.ok(loadingHealth, 'onHealthCheck should succeed cleanly on loading payload');
  assert.equal(pp.getStateSnapshot().healthStatus, 'loading');
  assert.notEqual(modalEls.ppHealthStatus.style.color, '#991b1b');
  assert.match(modalEls.ppHealthStatus.textContent, /GPU|loading|tải/i);

  // 3. ngrok-skip-browser-warning header included ONLY when URL host matches /ngrok/i
  assert.equal(seenHeaders[0].headers['ngrok-skip-browser-warning'], 'true');
  assert.equal(pp.isNgrokUrl('https://abc-123.ngrok-free.app/health'), true);
  assert.equal(pp.isNgrokUrl('http://127.0.0.1:8000/health?next=ngrok'), false);
  assert.equal(pp.isNgrokUrl('https://tunnel.trycloudflare.com/health'), false);

  const ngHeaders = pp.buildRequestHeaders('https://abc.ngrok.io/api/v1/bodyscan', { 'X-Custom': '1' }, true);
  assert.equal(ngHeaders['ngrok-skip-browser-warning'], 'true');
  assert.equal(ngHeaders.Authorization, 'Bearer ng-key');
  assert.equal(ngHeaders['X-Custom'], '1');

  const localHeaders = pp.buildRequestHeaders('http://127.0.0.1:8000/api/v1/bodyscan?host=ngrok', {}, true);
  assert.equal('ngrok-skip-browser-warning' in localHeaders, false);
  const cfHeaders = pp.buildRequestHeaders('https://foo.trycloudflare.com/api/v1/bodyscan', {}, true);
  assert.equal('ngrok-skip-browser-warning' in cfHeaders, false);
});

test('P0-1, P0-2.3 & P1-1: Zero-friction /lab auto-connect, 1-click consent switch, backend diagnostics, and 3D viewer polish', async () => {
  const sandbox = loadThreeAndModule();
  const pp = sandbox.AlphaPlusPro;
  const THREE = sandbox.THREE;

  // 1. Self-healing loadCfg() on non-loopback origin (e.g. https://fresh-colab.trycloudflare.com/lab/)
  pp.resetSession();
  sandbox.localStorage.setItem('ak_gpu_cfg', JSON.stringify({ url: 'http://127.0.0.1:8000', apiKey: '' }));
  sandbox.location = {
    protocol: 'https:',
    hostname: 'fresh-colab.trycloudflare.com',
    origin: 'https://fresh-colab.trycloudflare.com',
    pathname: '/lab/',
    search: '',
  };
  let cfg = pp.loadCfg();
  assert.equal(cfg.url, 'https://fresh-colab.trycloudflare.com', 'Stale loopback URL in localStorage must auto-bind to current tunnel origin');

  // Stale old tunnel URL in localStorage must also auto-bind to new tunnel origin
  pp.resetSession();
  sandbox.localStorage.setItem('ak_gpu_cfg', JSON.stringify({ url: 'https://old-dead-tunnel.trycloudflare.com', apiKey: 'old-k' }));
  cfg = pp.loadCfg();
  assert.equal(cfg.url, 'https://fresh-colab.trycloudflare.com', 'Stale old trycloudflare tunnel URL must auto-bind to new tunnel origin');

  // Query param ?gpu_url= & ?api_key= override
  pp.resetSession();
  sandbox.location = {
    protocol: 'https:',
    hostname: 'app.khoe.vn',
    origin: 'https://app.khoe.vn',
    pathname: '/lab/',
    search: '?gpu_url=https%3A%2F%2Fcustom-gpu.trycloudflare.com%2F&api_key=secret123',
  };
  cfg = pp.loadCfg();
  assert.equal(cfg.url, 'https://custom-gpu.trycloudflare.com');
  assert.equal(cfg.apiKey, 'secret123');

  // 2. Modal HTML collapses <details> on /lab, renders auto-connect badge & API key placeholder, and places #ppHealthStatus outside <details>
  sandbox.location = {
    protocol: 'https:',
    hostname: 'fresh-colab.trycloudflare.com',
    origin: 'https://fresh-colab.trycloudflare.com',
    pathname: '/lab/',
    search: '',
  };
  pp.resetSession();
  const modalHtml = pp.buildModalHtml();
  assert.doesNotMatch(modalHtml, /<details[^>]*\bopen\b/, '<details> must be collapsed by default when on /lab origin');
  assert.match(modalHtml, /Tự động kết nối — không cần điền/);
  assert.match(modalHtml, /Để trống \(chỉ điền nếu bạn tự đặt mật khẩu trên Colab\)/);
  assert.match(modalHtml, /<\/details>\s*<div id="ppHealthStatus"/, '#ppHealthStatus must stay visible outside collapsed <details>');

  // 3. Auto-health check on openModal() + 1-click consent switch (#ppBtnSwitchServerConsent) + backend tags in /health
  let currentConsent = 'local';
  const stubEls = {};
  const getOrMakeEl = (id) => {
    if (!stubEls[id]) {
      stubEls[id] = {
        id,
        innerHTML: '',
        textContent: '',
        value: id === 'ppHeight' ? '168' : (id === 'ppServerUrl' ? 'https://fresh-colab.trycloudflare.com' : ''),
        checked: id === 'ppConsentChk',
        disabled: false,
        style: {},
        dataset: {},
        className: '',
        classList: { toggle() {} },
        addEventListener() {},
        appendChild() {},
      };
    }
    return stubEls[id];
  };
  [
    'plusModal', 'ppSlotsGrid', 'ppExtraCount', 'ppBatchBtn', 'ppDupWarn',
    'ppGuardMsg', 'ppSubmitBtn', 'ppHeight', 'ppConsentChk', 'ppServerUrl',
    'ppApiKey', 'ppHealthStatus', 'akGpuHealthOut', 'ppBtnSwitchServerConsent',
  ].forEach(getOrMakeEl);

  sandbox.document.head = { appendChild() {} };
  sandbox.document.body = { appendChild() {}, removeChild() {} };
  sandbox.document.querySelectorAll = () => [];
  sandbox.document.getElementById = (id) => stubEls[id] || null;
  sandbox.document.createElement = (tag) => getOrMakeEl(tag);

  let healthCalls = 0;
  sandbox.fetch = async (url) => {
    if (url.endsWith('/health')) {
      healthCalls += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          ok: true,
          status: 'ready',
          model_ready: true,
          mode: 'gpu',
          test_only: true,
          backends: {
            gpu_name: 'Tesla T4',
            segmentation: 'sam2',
            pytorch3d_available: true,
            rasterizer: 'pytorch3d',
            moge2_available: true,
          },
        }),
      };
    }
    throw new Error(`Unexpected fetch: ${url}`);
  };

  pp.init({
    getClient: () => ({ name: 'Colab Tester', sex: 'female', height: 168, weight: 55 }),
    getConsentMode: () => currentConsent,
    setConsentMode: (m) => { currentConsent = m; },
    closeBodyEditModal() {},
  });

  pp.openModal();
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(healthCalls, 1, 'openModal() should auto-invoke onHealthCheck() once when _lastHealthOk === null');
  assert.match(stubEls.ppHealthStatus.textContent, /GPU thật/);
  assert.match(stubEls.ppHealthStatus.textContent, /TEST_ONLY/);
  assert.match(stubEls.ppHealthStatus.textContent, /Tesla T4/);
  assert.match(stubEls.ppHealthStatus.textContent, /Seg:sam2/);
  assert.match(stubEls.ppHealthStatus.textContent, /PyTorch3D/);
  assert.match(stubEls.ppHealthStatus.textContent, /MoGe2/);

  // Because currentConsent === 'local' and URL is remote https://fresh-colab.trycloudflare.com,
  // #ppGuardMsg must offer #ppBtnSwitchServerConsent and clicking it switches currentConsent to 'server'
  assert.match(stubEls.ppGuardMsg.innerHTML, /ppBtnSwitchServerConsent/);
  assert.equal(typeof stubEls.ppBtnSwitchServerConsent.onclick, 'function');
  stubEls.ppBtnSwitchServerConsent.onclick();
  assert.equal(currentConsent, 'server', 'Clicking #ppBtnSwitchServerConsent must switch consent mode to server');

  // 4. renderDetailHtml() surfaces backends_used chips (ok for GPU/SAM2/PyTorch3D, warn amber for fallbacks)
  const detailHtml = pp.renderDetailHtml({
    hasLoadedMesh: true,
    realActive: true,
    serverMode: 'gpu',
    testOnly: true,
    privacyLevel: 2,
    reconstructionMode: 'dual_layer',
    effectiveReconstructionMode: 'dual_layer',
    backendsUsed: {
      segmentation: 'sam2_hiera',
      texture_rasterizer: 'cpu_zbuf',
      moge2_depth_used: false,
      fitting_silhouette: 'pytorch3d_silhouette',
    },
    bodyScan: { measurements_cm: { chest: 88.0 } },
    artifacts: { mesh_url: '/a.glb', core_mesh_url: '/c.glb', json_url: '/j.json' },
  }, 'vi');
  assert.match(detailHtml, /class="pp-pill ok"[^>]*data-pp-backend="segmentation"[^>]*>seg: sam2_hiera</);
  assert.match(detailHtml, /class="pp-pill warn"[^>]*data-pp-backend="texture_rasterizer"[^>]*>tex: cpu_zbuf</);
  assert.match(detailHtml, /class="pp-pill warn"[^>]*data-pp-backend="moge2_depth_used"[^>]*>moge2:off</);
  assert.match(detailHtml, /class="pp-pill ok"[^>]*data-pp-backend="fitting_silhouette"[^>]*>fit: pytorch3d_silhouette</);

  // 5. parseGlbBinary sets anisotropy=4, LinearFilter, and untextured default roughness=0.58
  const untexturedGlb = buildMiniTestGlb(1.0);
  const parsedUntex = await pp.parseGlbBinary(untexturedGlb.buffer.slice(0), THREE, makeFakeImageFactory());
  let untexMat = null;
  parsedUntex.group.traverse((obj) => { if (obj.isMesh && obj.material && !untexMat) untexMat = obj.material; });
  assert.ok(untexMat, 'Expected parsed untextured material');
  assert.ok(Math.abs(untexMat.roughness - 0.58) < 1e-3, `Expected untextured roughness 0.58, got ${untexMat.roughness}`);
  pp.disposeObject3D(parsedUntex.group);

  const lvl2Tex = generateMockGlbBytes(2, 165.0);
  const parsedTex = await pp.parseGlbBinary(lvl2Tex.arrayBuffer, THREE, async () => ({ width: 512, height: 512 }));
  let texMap = null;
  parsedTex.group.traverse((obj) => { if (obj.isMesh && obj.material && obj.material.map) texMap = obj.material.map; });
  assert.ok(texMap, 'Expected textured map on Level 2 GLB');
  assert.equal(texMap.anisotropy, 4);
  assert.equal(texMap.magFilter, THREE.LinearFilter);
  assert.equal(texMap.encoding, THREE.sRGBEncoding);
  assert.equal(texMap.flipY, false);
  pp.disposeObject3D(parsedTex.group);

  // Verify index.html wires sRGBEncoding, ACESFilmicToneMapping, and rx=0.04 in frameRealModel
  const indexHtml = fs.readFileSync(path.join(ROOT, 'frontend_lab', 'index.html'), 'utf8');
  assert.match(indexHtml, /rnd\.outputEncoding\s*=\s*THREE\.sRGBEncoding/);
  assert.match(indexHtml, /rnd\.toneMapping\s*=\s*THREE\.ACESFilmicToneMapping/);
  assert.match(indexHtml, /rx\s*=\s*0\.04/);
});

function buildCustomJsonGlb(gltfObj, binBytes = new Uint8Array(0)) {
  const jsonText = JSON.stringify(gltfObj);
  const jsonRaw = Buffer.from(jsonText, 'utf8');
  const jsonPadLen = (4 - (jsonRaw.length % 4)) % 4;
  const jsonChunk = Buffer.concat([jsonRaw, Buffer.alloc(jsonPadLen, 0x20)]);

  const binRaw = Buffer.from(binBytes);
  const binPadLen = (4 - (binRaw.length % 4)) % 4;
  const binChunk = Buffer.concat([binRaw, Buffer.alloc(binPadLen, 0x00)]);

  const totalLen = 12 + 8 + jsonChunk.length + (binChunk.length > 0 ? 8 + binChunk.length : 0);
  const out = Buffer.alloc(totalLen);
  out.writeUInt32LE(0x46546C67, 0);
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(totalLen, 8);
  let off = 12;
  out.writeUInt32LE(jsonChunk.length, off);
  out.writeUInt32LE(0x4E4F534A, off + 4);
  jsonChunk.copy(out, off + 8);
  off += 8 + jsonChunk.length;
  if (binChunk.length > 0) {
    out.writeUInt32LE(binChunk.length, off);
    out.writeUInt32LE(0x004E4942, off + 4);
    binChunk.copy(out, off + 8);
  }
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
}

test('GLB fidelity: COLOR_0, doubleSided, alphaMode MASK/BLEND/OPAQUE, bufferView-omitted zero accessor, samplers & NPOT', async () => {
  const sandbox = loadThreeAndModule();
  const pp = sandbox.AlphaPlusPro;
  const THREE = sandbox.THREE;

  // 3 vertices (vec3 float32 = 36 bytes) + 3 colors (vec3 float32 = 36 bytes) + 3 UVs (vec2 float32 = 24 bytes) + 1x1 fake JPEG bytes (4 bytes)
  const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const col = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const uv = new Float32Array([0, 0, 1, 0, 0, 1]);
  const imgBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
  const bin = Buffer.concat([
    Buffer.from(pos.buffer),
    Buffer.from(col.buffer),
    Buffer.from(uv.buffer),
    Buffer.from(imgBytes),
  ]);

  const gltf = {
    asset: { version: '2.0' },
    scenes: [{ nodes: [0, 1, 2] }],
    nodes: [
      { mesh: 0 },
      { mesh: 1 },
      { mesh: 2 },
    ],
    samplers: [
      { wrapS: 33648, wrapT: 33071, magFilter: 9728, minFilter: 9728 },
    ],
    images: [
      { bufferView: 3, mimeType: 'image/jpeg' },
    ],
    textures: [
      { source: 0, sampler: 0 },
      { source: 0 },
    ],
    materials: [
      {
        doubleSided: false,
        alphaMode: 'MASK',
        alphaCutoff: 0.42,
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0 },
          baseColorFactor: [1, 1, 1, 0.2],
        },
      },
      {
        doubleSided: true,
        alphaMode: 'BLEND',
        pbrMetallicRoughness: {
          baseColorTexture: { index: 1 },
          baseColorFactor: [1, 1, 1, 0.5],
        },
      },
      {
        alphaMode: 'OPAQUE',
        pbrMetallicRoughness: {
          baseColorFactor: [0.8, 0.7, 0.6, 0.3],
        },
      },
    ],
    meshes: [
      {
        name: 'mesh_mask_pot',
        primitives: [{ attributes: { POSITION: 0, COLOR_0: 1, TEXCOORD_0: 2 }, material: 0 }],
      },
      {
        name: 'mesh_blend_zero_norm',
        primitives: [{ attributes: { POSITION: 0, NORMAL: 3, TEXCOORD_0: 2 }, material: 1 }],
      },
      {
        name: 'mesh_opaque_untex',
        primitives: [{ attributes: { POSITION: 0 }, material: 2 }],
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 36 },
      { buffer: 0, byteOffset: 72, byteLength: 24 },
      { buffer: 0, byteOffset: 96, byteLength: 4 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      // Omitted bufferView => zero-filled accessor per glTF 2.0 spec
      { componentType: 5126, count: 3, type: 'VEC3' },
    ],
  };

  // First test with POT texture (128x128) honoring sampler
  const glbBuf = buildCustomJsonGlb(gltf, bin);
  const parsedPot = await pp.parseGlbBinary(glbBuf, THREE, async () => ({ width: 128, height: 128 }));
  const meshesPot = [];
  parsedPot.group.traverse((o) => { if (o.isMesh) meshesPot.push(o); });
  assert.equal(meshesPot.length, 3);

  const m0 = meshesPot[0];
  assert.ok(m0.geometry.getAttribute('color'), 'COLOR_0 must populate geometry color attribute');
  assert.equal(m0.material.vertexColors, true);
  assert.equal(m0.material.side, THREE.FrontSide);
  assert.equal(m0.material.alphaTest, 0.42);
  assert.equal(m0.material.transparent, false);
  assert.equal(m0.material.map.wrapS, THREE.MirroredRepeatWrapping);
  assert.equal(m0.material.map.wrapT, THREE.ClampToEdgeWrapping);
  assert.equal(m0.material.map.magFilter, THREE.NearestFilter);

  const m1 = meshesPot[1];
  assert.equal(m1.material.side, THREE.DoubleSide);
  assert.equal(m1.material.transparent, true);
  assert.equal(m1.material.opacity, 0.5);
  assert.ok(m1.geometry.getAttribute('normal'), 'Zero-filled accessor without bufferView must load');

  const m2 = meshesPot[2];
  assert.equal(m2.material.side, THREE.FrontSide);
  assert.equal(m2.material.transparent, false, 'alphaMode=OPAQUE must ignore alpha<1');
  pp.disposeObject3D(parsedPot.group);

  // Now test NPOT texture (300x200) forces WebGL1-safe ClampToEdgeWrapping + non-mipmap minFilter
  const parsedNpot = await pp.parseGlbBinary(glbBuf, THREE, async () => ({ width: 300, height: 200 }));
  const meshesNpot = [];
  parsedNpot.group.traverse((o) => { if (o.isMesh) meshesNpot.push(o); });
  const npotTex0 = meshesNpot[0].material.map;
  const npotTex1 = meshesNpot[1].material.map;
  assert.equal(npotTex0.wrapS, THREE.ClampToEdgeWrapping);
  assert.equal(npotTex0.wrapT, THREE.ClampToEdgeWrapping);
  assert.equal(npotTex0.generateMipmaps, false);
  assert.equal(npotTex0.minFilter, THREE.NearestFilter);
  assert.equal(npotTex1.wrapS, THREE.ClampToEdgeWrapping);
  assert.equal(npotTex1.wrapT, THREE.ClampToEdgeWrapping);
  assert.equal(npotTex1.generateMipmaps, false);
  assert.equal(npotTex1.minFilter, THREE.LinearFilter);
  pp.disposeObject3D(parsedNpot.group);
});

test('Step 7 viewer polish: neutral real-3D lights, viewSeg front/back, preserveCamera, PDF front snap, touch-action', () => {
  const html = fs.readFileSync(path.join(ROOT, 'frontend_lab', 'index.html'), 'utf8');
  assert.match(html, /#scene\{[^}]*touch-action:\s*none/);
  assert.match(html, /function applySceneLights\(isReal3D\)/);
  assert.match(html, /rnd\.toneMappingExposure\s*=\s*0\.88/);
  assert.match(html, /frameRealModel\(opts\)/);
  assert.match(html, /frameModel\(\{\s*preserveCamera:\s*true\s*\}\)/);
  assert.match(html, /attachRealGroup:\s*\(grp,\s*metrics,\s*opts\)/);
  assert.match(html, /const v=b\.dataset\.view,\s*isReal3D=.*AlphaPlusPro\.isReal3DActive\(\)/);
  assert.match(html, /ry=isReal3D\?0:-0\.45/);
  assert.match(html, /ry=isReal3D\?Math\.PI:\(Math\.PI-0\.45\)/);
  assert.match(html, /if\(hasReal3D\)\{\s*spin=false;\s*ry=0;\s*rx=0;\s*realPivot\.rotation\.set\(0,0,0\);/);
});

test('Polling resilience (502/network retry), slot_roles FormData, content-fingerprint dup detection, assetsDeleted JSON & auth hint', async () => {
  const sandbox = loadThreeAndModule();
  const pp = sandbox.AlphaPlusPro;
  const THREE = sandbox.THREE;
  pp.resetSession();

  // 1. buildScanFormData emits slot_roles JSON array matching photo order
  const fd = pp.buildScanFormData({
    photos: [
      new Blob(['a'], { type: 'image/jpeg' }),
      new Blob(['b'], { type: 'image/jpeg' }),
      new Blob(['c'], { type: 'image/jpeg' }),
      new Blob(['d'], { type: 'image/jpeg' }),
    ],
    slotRoles: ['front', 'right', 'back', 'left'],
    heightCm: 165,
    sex: 'female',
    weightKg: 54,
    privacyLevel: 2,
    reconstructionMode: 'dual_layer',
  });
  assert.equal(fd.get('slot_roles'), '["front","right","back","left"]');

  // 2. Content-fingerprint duplicate detection catches renamed identical files
  const bytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const fA = new Blob([bytes], { type: 'image/jpeg' });
  fA.name = 'front_cam.jpg';
  const fB = new Blob([bytes], { type: 'image/jpeg' });
  fB.name = 'right_renamed_copy.jpg';
  const fC = new Blob([new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])], { type: 'image/jpeg' });
  fC.name = 'back_cam.jpg';
  await pp.enrichFileContentFingerprints([fA, fB, fC]);
  const dupRes = pp.findDuplicateFileSignatures([fA, fB, fC]);
  assert.equal(dupRes.hasDuplicates, true);
  assert.match(dupRes.duplicateNames[0], /front_cam\.jpg/);
  assert.match(dupRes.duplicateNames[0], /right_renamed_copy\.jpg/);

  // 3. assetsDeleted detail card keeps #ppDlJson download button
  const deletedHtml = pp.renderDetailHtml({
    assetsDeleted: true,
    serverMode: 'gpu',
    testOnly: false,
    bodyScan: { circumferences_cm: { waist: 70 } },
    artifacts: { json_url: 'https://srv/api/v1/bodyscan/j1/files/body_scan.json?sig=ok' },
  }, 'vi');
  assert.match(deletedHtml, /id="ppDlJson"/);
  assert.match(deletedHtml, /body_scan\.json/);

  // 4. pollJobUntilDone survives transient network error + 502 before succeeding
  const lvl1 = generateMockGlbBytes(1, 168.0, 'calibrated_mhr');
  let pollAttempt = 0;
  const mockFetch = async (url) => {
    if (url.includes('/api/v1/bodyscan/job-retry-1') && !url.includes('/files/')) {
      pollAttempt += 1;
      if (pollAttempt === 1) throw new Error('ECONNRESET');
      if (pollAttempt === 2) return { ok: false, status: 502, json: async () => ({}) };
      if (pollAttempt === 3) return { ok: true, status: 200, json: async () => ({ status: 'running', progress: 55, step: 'fitting' }) };
      return {
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job-retry-1',
          status: 'completed',
          progress: 100,
          step: 'completed',
          mode: 'mock',
          body_scan: lvl1.bodyScan,
          artifacts: { mesh_url: 'https://srv/api/v1/bodyscan/job-retry-1/files/scan.glb?exp=9999999999&sig=s' },
        }),
      };
    }
    if (url.includes('/files/scan.glb')) {
      return { ok: true, status: 200, arrayBuffer: async () => lvl1.arrayBuffer };
    }
    throw new Error('Unexpected URL: ' + url);
  };
  const completed = await pp.pollJobUntilDone('https://srv', 'job-retry-1', {
    fetchFn: mockFetch,
    THREE,
    pollIntervalMs: 10,
    retryDelayMs: 10,
    maxRetries: 4,
    imageFactory: async () => ({ width: 32, height: 32 }),
  });
  assert.equal(completed.status, 'completed');
  assert.equal(pollAttempt, 4);
  pp.resetSession();

  // Fail fast on 410 Gone without retrying 5 times
  let goneCalls = 0;
  await assert.rejects(
    () => pp.pollJobUntilDone('https://srv', 'job-410', {
      fetchFn: async () => { goneCalls += 1; return { ok: false, status: 410, json: async () => ({}) }; },
      pollIntervalMs: 10,
      retryDelayMs: 10,
      maxRetries: 4,
    }),
    /Poll HTTP 410/
  );
  assert.equal(goneCalls, 1);

  // 5. onHealthCheck surfaces API key hint when auth_required=true and key empty
  const healthEl = { textContent: '', style: {} };
  sandbox.document.getElementById = (id) => (id === 'ppHealthStatus' ? healthEl : null);
  await pp.onHealthCheck('http://127.0.0.1:8000', '', {
    fetchFn: async () => ({
      ok: true,
      status: 200,
      json: async () => ({ ok: true, status: 'ready', model_ready: true, mode: 'gpu', auth_required: true, backends: {} }),
    }),
  });
  assert.match(healthEl.textContent, /API Key/);
});

test('khoe-qr.js handles UTF-8 URLs and Colab notebook / requirements / README pass consistency lint', () => {
  const qrCode = fs.readFileSync(path.join(ROOT, 'frontend_lab', 'khoe-qr.js'), 'utf8');
  const qrSandbox = { console, TextEncoder };
  qrSandbox.window = qrSandbox;
  vm.createContext(qrSandbox);
  vm.runInContext(qrCode, qrSandbox);
  const svgOut = qrSandbox.KhoeQR.svg('https://app.appkhoe.vn/?src=alphascan&note=Khỏe', 'App KHOẺ');
  assert.match(svgOut, /^<svg /);
  assert.match(svgOut, /<path d="M/);

  const reqTxt = fs.readFileSync(path.join(ROOT, 'requirements.txt'), 'utf8');
  assert.match(reqTxt, /mediapipe>=0\.10\.14,<0\.11;\s*python_version\s*<\s*"3\.13"/);

  const readmeTxt = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
  assert.match(readmeTxt, /slot_roles/);
  assert.match(readmeTxt, /~44 MB/);

  // Validate all Colab code cells AST + key P0 guards via Python
  const pyOut = execFileSync(
    'python',
    [
      '-c',
      [
        'import ast, json',
        'nb = json.load(open("AlphaScan_Plan2A_Colab.ipynb", encoding="utf-8"))',
        'for i, c in enumerate(nb["cells"]):',
        '    if c["cell_type"] == "code":',
        '        ast.parse("".join(c["source"]), filename=f"cell_{i}")',
        'c1 = "".join(nb["cells"][2]["source"])',
        'assert "USE_GOOGLE_DRIVE = True" in c1',
        'c2 = "".join(nb["cells"][4]["source"])',
        'assert "numpy<2.3" in c2 and "abi_needle" in c2',
        'c3 = "".join(nb["cells"][6]["source"])',
        'assert "APPLY_HOTFIX_OVERLAY" in c3 and "_clean_stale_package_files" in c3 and "numpy<2.3" in c3',
        'c7 = "".join(nb["cells"][14]["source"])',
        'c7b = "".join(nb["cells"][16]["source"])',
        'assert "cf_proc.poll() is not None" in c7 and "cf_proc.poll() is not None" in c7b',
        'print("COLAB_OK")',
      ].join('\n'),
    ],
    { cwd: ROOT, encoding: 'utf8' }
  );
  assert.match(pyOut, /COLAB_OK/);
});


