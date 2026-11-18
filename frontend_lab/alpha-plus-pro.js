/* Alpha Scan Plus/PRO integration module for Alpha Scan Lab (Three.js r128, offline-ready, Plan 2A). */
(function (rootScope) {
  'use strict';

  const ANATOMICAL_ORDER = [
    'neck', 'chest', 'waist', 'hip',
    'bicep_l', 'bicep_r', 'forearm_l', 'forearm_r',
    'wrist_l', 'wrist_r', 'thigh_l', 'thigh_r',
    'calf_l', 'calf_r', 'ankle_l', 'ankle_r',
  ];

  const LABELS_VI = {
    neck: 'Vòng cổ',
    chest: 'Vòng ngực',
    waist: 'Vòng eo',
    hip: 'Vòng hông',
    bicep_l: 'Bắp tay trái',
    bicep_r: 'Bắp tay phải',
    forearm_l: 'Cẳng tay trái',
    forearm_r: 'Cẳng tay phải',
    wrist_l: 'Cổ tay trái',
    wrist_r: 'Cổ tay phải',
    thigh_l: 'Đùi trái',
    thigh_r: 'Đùi phải',
    calf_l: 'Bắp chân trái',
    calf_r: 'Bắp chân phải',
    ankle_l: 'Cổ chân trái',
    ankle_r: 'Cổ chân phải',
  };

  const LABELS_EN = {
    neck: 'Neck',
    chest: 'Chest',
    waist: 'Waist',
    hip: 'Hip',
    bicep_l: 'Left Bicep',
    bicep_r: 'Right Bicep',
    forearm_l: 'Left Forearm',
    forearm_r: 'Right Forearm',
    wrist_l: 'Left Wrist',
    wrist_r: 'Right Wrist',
    thigh_l: 'Left Thigh',
    thigh_r: 'Right Thigh',
    calf_l: 'Left Calf',
    calf_r: 'Right Calf',
    ankle_l: 'Left Ankle',
    ankle_r: 'Right Ankle',
  };

  const ANGLE_SLOTS = [
    { key: 'front', vi: '1. Chính diện', en: '1. Front' },
    { key: 'right', vi: '2. Nghiêng phải (vai phải→cam)', en: '2. Subject Right' },
    { key: 'back', vi: '3. Phía sau', en: '3. Back' },
    { key: 'left', vi: '4. Nghiêng trái (vai trái→cam)', en: '4. Subject Left' },
  ];

  const RECONSTRUCTION_MODES = ['calibrated_mhr', 'hi_res_head', 'dual_layer'];
  const DEFAULT_RECONSTRUCTION_MODE = 'calibrated_mhr';
  const ARTIFACT_URL_REFRESH_MS = 240 * 1000;

  const CFG_KEY = 'ak_gpu_cfg';

  function isLoopbackUrl(urlStr) {
    if (!urlStr || typeof urlStr !== 'string') return false;
    try {
      const u = new URL(urlStr.trim());
      if (u.protocol !== 'http:' && u.protocol !== 'https:') return false;
      const host = (u.hostname || '').replace(/^\[|\]$/g, '').toLowerCase();
      return host === 'localhost' || host === '127.0.0.1' || host === '::1';
    } catch (_) {
      return false;
    }
  }

  function isNgrokUrl(urlStr) {
    if (!urlStr || typeof urlStr !== 'string') return false;
    try {
      const base = (rootScope.location && rootScope.location.origin) || 'http://127.0.0.1:8000';
      const u = new URL(urlStr.trim(), base);
      return /ngrok/i.test(u.hostname || '');
    } catch (_) {
      return false;
    }
  }

  function _normalizeFilenameStem(name) {
    const raw = String(name || '');
    const dot = raw.lastIndexOf('.');
    const stem = (dot > 0 ? raw.slice(0, dot) : raw).toLowerCase();
    return stem.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd');
  }

  function inferCardinalRoleFromFilename(name) {
    const norm = _normalizeFilenameStem(name);
    if (!norm) return null;
    const matched = new Set();
    if (/(?:^|[^a-z])(front|truoc|chinh_?dien)(?:$|[^a-z])/.test(norm) || /(?:^|[^a-z0-9])01(?:$|[^a-z0-9])/.test(norm)) {
      matched.add(0);
    }
    if (/(?:^|[^a-z])(right|phai)(?:$|[^a-z])/.test(norm) || /(?:^|[^a-z0-9])02(?:$|[^a-z0-9])/.test(norm)) {
      matched.add(1);
    }
    if (/(?:^|[^a-z])(back|sau)(?:$|[^a-z])/.test(norm) || /(?:^|[^a-z0-9])03(?:$|[^a-z0-9])/.test(norm)) {
      matched.add(2);
    }
    if (/(?:^|[^a-z])(left|trai)(?:$|[^a-z])/.test(norm) || /(?:^|[^a-z0-9])04(?:$|[^a-z0-9])/.test(norm)) {
      matched.add(3);
    }
    if (matched.size !== 1) return null;
    return Array.from(matched)[0];
  }

  function _fnv1aHex(bytes) {
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
      h ^= bytes[i];
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return ('00000000' + h.toString(16)).slice(-8);
  }

  async function enrichFileContentFingerprints(files) {
    const list = Array.isArray(files) ? files : [];
    await Promise.all(list.map(async (f) => {
      if (!f || typeof f !== 'object' || f._contentFingerprint) return;
      try {
        const size = Number.isFinite(Number(f.size)) ? Number(f.size) : 0;
        let sampleBuf = null;
        if (typeof f.slice === 'function' && size > 0) {
          const headEnd = Math.min(size, 4096);
          const tailStart = Math.max(0, size - 4096);
          const headBuf = await f.slice(0, headEnd).arrayBuffer();
          const tailBuf = tailStart > headEnd ? await f.slice(tailStart, size).arrayBuffer() : new ArrayBuffer(0);
          const hArr = new Uint8Array(headBuf);
          const tArr = new Uint8Array(tailBuf);
          const merged = new Uint8Array(hArr.length + tArr.length);
          merged.set(hArr, 0);
          merged.set(tArr, hArr.length);
          sampleBuf = merged;
        } else if (typeof f.arrayBuffer === 'function') {
          sampleBuf = new Uint8Array(await f.arrayBuffer());
        }
        if (sampleBuf) {
          f._contentFingerprint = 'fp:' + size + ':' + _fnv1aHex(sampleBuf);
        }
      } catch (_) {}
    }));
    return list;
  }

  function findDuplicateFileSignatures(files) {
    const list = Array.isArray(files) ? files : [];
    const groups = new Map();
    list.forEach((f, idx) => {
      if (!f || typeof f !== 'object') return;
      const name = f.name != null ? String(f.name) : '';
      const fp = f._contentFingerprint || f.contentFingerprint || '';
      if (!name && !fp) return;
      const size = Number.isFinite(Number(f.size)) ? Number(f.size) : '';
      const sig = fp ? ('content::' + fp) : (name + '::' + size);
      if (!groups.has(sig)) {
        groups.set(sig, { signature: sig, name: name || fp, names: [], size, count: 0, indices: [] });
      }
      const entry = groups.get(sig);
      entry.count += 1;
      entry.indices.push(idx);
      if (name && !entry.names.includes(name)) entry.names.push(name);
    });
    const duplicates = [];
    groups.forEach((entry) => {
      if (entry.count > 1) {
        entry.displayName = entry.names.length > 1 ? entry.names.join(' = ') : (entry.names[0] || entry.name);
        duplicates.push(entry);
      }
    });
    return {
      hasDuplicates: duplicates.length > 0,
      duplicates,
      duplicateNames: duplicates.map((d) => d.displayName || d.name),
    };
  }

  function assignBatchFilesToSlots(incomingFiles, existingSlots, existingExtras) {
    const slots = [0, 1, 2, 3].map((i) => (existingSlots && existingSlots[i]) || null);
    const extras = Array.isArray(existingExtras) ? existingExtras.slice() : [];
    const files = Array.from(incomingFiles || []).slice(0, 8);
    const totalUsed = () => slots.filter(Boolean).length + extras.length;

    const roleCounts = new Map();
    const parsed = files.map((f) => {
      const role = inferCardinalRoleFromFilename(f && f.name);
      if (role !== null) roleCounts.set(role, (roleCounts.get(role) || 0) + 1);
      return { file: f, role };
    });

    const unplaced = [];
    parsed.forEach(({ file, role }) => {
      if (role !== null && roleCounts.get(role) === 1 && !slots[role] && totalUsed() < 8) {
        slots[role] = file;
      } else {
        unplaced.push(file);
      }
    });

    let cursor = 0;
    unplaced.forEach((f) => {
      while (cursor < 4 && slots[cursor]) cursor++;
      if (cursor < 4 && totalUsed() < 8) {
        slots[cursor++] = f;
      } else if (totalUsed() < 8) {
        extras.push(f);
      }
    });

    const all = slots.filter(Boolean).concat(extras);
    return {
      slots,
      extraPhotos: extras,
      duplicateInfo: findDuplicateFileSignatures(all),
    };
  }

  function classifyHealthPayload(payload, httpStatus) {
    const j = payload && typeof payload === 'object' ? payload : null;
    const statusNum = Number(httpStatus || (j ? 200 : 0));
    const isOkHttp = statusNum >= 200 && statusNum < 300;
    if (j && (String(j.status || '').toLowerCase() === 'loading' || j.model_ready === false)) {
      return {
        state: 'loading',
        ok: true,
        isLoading: true,
        mode: j.mode || 'gpu',
        testOnly: j.test_only !== false,
        detail: j.detail || j.step || j.message || '',
      };
    }
    if (!isOkHttp || !j || j.ok === false) {
      return {
        state: 'error',
        ok: false,
        isLoading: false,
        mode: (j && j.mode) || null,
        testOnly: j ? j.test_only !== false : true,
        detail: (j && (j.detail || j.error)) || (statusNum ? 'HTTP ' + statusNum : 'Network error'),
      };
    }
    return {
      state: 'ready',
      ok: true,
      isLoading: false,
      mode: j.mode || 'gpu',
      testOnly: j.test_only !== false,
      detail: '',
    };
  }

  function validatePrivacyAndConnection(opts) {
    const o = opts || {};
    const urlStr = (o.serverUrl || '').trim();
    try {
      const parsed = new URL(urlStr);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return { ok: false, reasonKey: 'pp_err_url' };
      }
    } catch (_) {
      return { ok: false, reasonKey: 'pp_err_url' };
    }
    if (o.consentMode === 'local' && !isLoopbackUrl(urlStr)) {
      return { ok: false, reasonKey: 'pp_err_local_only' };
    }
    const count = Number(o.photoCount || 0);
    if (!Number.isFinite(count) || count < 1 || count > 8) {
      return { ok: false, reasonKey: 'pp_err_photos' };
    }
    const mode = (o.reconstructionMode !== undefined && o.reconstructionMode !== null && o.reconstructionMode !== '')
      ? String(o.reconstructionMode)
      : DEFAULT_RECONSTRUCTION_MODE;
    if (!RECONSTRUCTION_MODES.includes(mode)) {
      return { ok: false, reasonKey: 'pp_err_mode' };
    }
    const requires4Cardinal = mode === 'hi_res_head' || mode === 'dual_layer';
    const cardinalCount = (o.cardinalSlotsCount !== undefined && o.cardinalSlotsCount !== null)
      ? Number(o.cardinalSlotsCount)
      : Math.min(4, count);
    if (requires4Cardinal && (!Number.isFinite(cardinalCount) || cardinalCount < 4)) {
      return { ok: false, reasonKey: 'pp_err_4cardinal' };
    }
    const h = Number(o.heightCm);
    if (!Number.isFinite(h) || h < 80 || h > 230) {
      return { ok: false, reasonKey: 'pp_err_height' };
    }
    const lvl = Number(o.privacyLevel);
    if (lvl !== 1 && lvl !== 2 && lvl !== 3) {
      return { ok: false, reasonKey: 'pp_err_privacy' };
    }
    if (!o.consentChecked) {
      return { ok: false, reasonKey: 'pp_err_consent' };
    }
    return { ok: true, reasonKey: '' };
  }

  function _componentCount(typeStr) {
    if (typeStr === 'VEC2') return 2;
    if (typeStr === 'VEC3') return 3;
    if (typeStr === 'VEC4') return 4;
    return 1;
  }

  function _componentByteSize(componentType) {
    if (componentType === 5120 || componentType === 5121) return 1;
    if (componentType === 5122 || componentType === 5123) return 2;
    return 4;
  }

  function _readComponent(dv, offset, componentType, normalized) {
    if (componentType === 5126) return dv.getFloat32(offset, true);
    if (componentType === 5125) return dv.getUint32(offset, true);
    if (componentType === 5123) {
      const v = dv.getUint16(offset, true);
      return normalized ? v / 65535.0 : v;
    }
    if (componentType === 5122) {
      const v = dv.getInt16(offset, true);
      return normalized ? Math.max(v / 32767.0, -1.0) : v;
    }
    if (componentType === 5121) {
      const v = dv.getUint8(offset);
      return normalized ? v / 255.0 : v;
    }
    if (componentType === 5120) {
      const v = dv.getInt8(offset);
      return normalized ? Math.max(v / 127.0, -1.0) : v;
    }
    return dv.getFloat32(offset, true);
  }

  function _makeOutArray(componentType, normalized, length) {
    if (normalized || componentType === 5126) return new Float32Array(length);
    if (componentType === 5125) return new Uint32Array(length);
    if (componentType === 5123) return new Uint16Array(length);
    if (componentType === 5121) return new Uint8Array(length);
    return new Float32Array(length);
  }

  function _readAccessor(gltf, binBuffer, accIdx) {
    const acc = (gltf.accessors || [])[accIdx];
    if (!acc) throw new Error('Missing glTF accessor #' + accIdx);
    const nComp = _componentCount(acc.type);
    const compType = acc.componentType;
    const compBytes = _componentByteSize(compType);
    const count = acc.count || 0;
    const normalized = Boolean(acc.normalized);
    const out = _makeOutArray(compType, normalized, count * nComp);

    if (acc.bufferView !== undefined && acc.bufferView !== null) {
      const bv = (gltf.bufferViews || [])[acc.bufferView];
      if (!bv) throw new Error('Missing glTF bufferView #' + acc.bufferView);
      const baseOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
      const stride = bv.byteStride || (nComp * compBytes);
      const dv = new DataView(binBuffer);
      for (let i = 0; i < count; i++) {
        const elemOffset = baseOffset + i * stride;
        for (let c = 0; c < nComp; c++) {
          out[i * nComp + c] = _readComponent(dv, elemOffset + c * compBytes, compType, normalized);
        }
      }
    }

    if (acc.sparse && acc.sparse.count > 0 && acc.sparse.indices && acc.sparse.values) {
      const sp = acc.sparse;
      const idxBv = (gltf.bufferViews || [])[sp.indices.bufferView];
      const valBv = (gltf.bufferViews || [])[sp.values.bufferView];
      if (idxBv && valBv) {
        const dv = new DataView(binBuffer);
        const idxType = sp.indices.componentType || 5123;
        const idxBytes = _componentByteSize(idxType);
        const idxBase = (idxBv.byteOffset || 0) + (sp.indices.byteOffset || 0);
        const valBase = (valBv.byteOffset || 0) + (sp.values.byteOffset || 0);
        const valStride = valBv.byteStride || (nComp * compBytes);
        for (let s = 0; s < sp.count; s++) {
          const elemIdx = Math.round(_readComponent(dv, idxBase + s * idxBytes, idxType, false));
          if (elemIdx >= 0 && elemIdx < count) {
            const vOff = valBase + s * valStride;
            for (let c = 0; c < nComp; c++) {
              out[elemIdx * nComp + c] = _readComponent(dv, vOff + c * compBytes, compType, normalized);
            }
          }
        }
      }
    }
    return { data: out, itemSize: nComp, count: count };
  }

  function _defaultImageFactory(bytes, mimeType) {
    return new Promise((resolve, reject) => {
      const blob = new Blob([bytes], { type: mimeType || 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        try { URL.revokeObjectURL(url); } catch (_) {}
        resolve(img);
      };
      img.onerror = (err) => {
        try { URL.revokeObjectURL(url); } catch (_) {}
        reject(err || new Error('Failed to decode glTF embedded texture'));
      };
      img.src = url;
    });
  }

  function _decodeDataUri(uri) {
    const comma = uri.indexOf(',');
    const meta = uri.slice(5, comma);
    const b64 = uri.slice(comma + 1);
    const mime = (meta.split(';')[0] || 'image/jpeg').trim();
    const binStr = (typeof atob === 'function')
      ? atob(b64)
      : Buffer.from(b64, 'base64').toString('binary');
    const out = new Uint8Array(binStr.length);
    for (let i = 0; i < binStr.length; i++) out[i] = binStr.charCodeAt(i);
    return { bytes: out, mimeType: mime };
  }

  async function parseGlbBinary(arrayBuffer, THREE, imageFactory) {
    if (!arrayBuffer || arrayBuffer.byteLength < 20) {
      throw new Error('GLB buffer too small');
    }
    const dv = new DataView(arrayBuffer);
    const magic = dv.getUint32(0, true);
    const version = dv.getUint32(4, true);
    if (magic !== 0x46546C67 || version !== 2) {
      throw new Error('Invalid glTF 2.0 binary header');
    }

    let offset = 12;
    let gltfJson = null;
    let binBuffer = new ArrayBuffer(0);
    const totalLen = Math.min(arrayBuffer.byteLength, dv.getUint32(8, true));

    while (offset + 8 <= totalLen) {
      const chunkLen = dv.getUint32(offset, true);
      const chunkType = dv.getUint32(offset + 4, true);
      offset += 8;
      const end = Math.min(totalLen, offset + chunkLen);
      if (chunkType === 0x4E4F534A) {
        const text = new TextDecoder('utf-8').decode(new Uint8Array(arrayBuffer, offset, end - offset));
        gltfJson = JSON.parse(text);
      } else if (chunkType === 0x004E4942) {
        binBuffer = arrayBuffer.slice(offset, end);
      }
      offset += (chunkLen + 3) & ~3;
    }

    if (!gltfJson || !gltfJson.meshes) {
      throw new Error('GLB does not contain valid glTF JSON/meshes');
    }

    const loaderFn = imageFactory || _defaultImageFactory;
    const loadedTextures = {};
    const flipYs = [];
    let textureCount = 0;

    async function resolveTexture(texIdx) {
      if (loadedTextures[texIdx]) return loadedTextures[texIdx];
      const texDef = (gltfJson.textures || [])[texIdx];
      if (!texDef || texDef.source === undefined) return null;
      const imgDef = (gltfJson.images || [])[texDef.source];
      if (!imgDef) return null;

      let imgBytes = null;
      let mimeType = imgDef.mimeType || 'image/jpeg';
      if (imgDef.bufferView !== undefined) {
        const bv = (gltfJson.bufferViews || [])[imgDef.bufferView];
        if (!bv) return null;
        const start = bv.byteOffset || 0;
        imgBytes = new Uint8Array(binBuffer, start, bv.byteLength);
      } else if (typeof imgDef.uri === 'string' && imgDef.uri.startsWith('data:')) {
        const decoded = _decodeDataUri(imgDef.uri);
        imgBytes = decoded.bytes;
        mimeType = decoded.mimeType;
      }
      if (!imgBytes) return null;

      const imgObj = await loaderFn(imgBytes, mimeType);
      const tex = new THREE.Texture(imgObj);
      tex.flipY = false;
      tex.encoding = THREE.sRGBEncoding;
      tex.anisotropy = 4;

      const sampler = texDef.sampler !== undefined ? (gltfJson.samplers || [])[texDef.sampler] : null;
      const w = Number(imgObj && imgObj.width) || 0;
      const h = Number(imgObj && imgObj.height) || 0;
      const isPot = w > 0 && h > 0 && (w & (w - 1)) === 0 && (h & (h - 1)) === 0;
      const mapWrap = (val) => {
        if (val === 33071 && THREE.ClampToEdgeWrapping !== undefined) return THREE.ClampToEdgeWrapping;
        if (val === 33648 && THREE.MirroredRepeatWrapping !== undefined) return THREE.MirroredRepeatWrapping;
        if (THREE.RepeatWrapping !== undefined) return THREE.RepeatWrapping;
        return THREE.ClampToEdgeWrapping;
      };
      if (!isPot && w > 0 && h > 0) {
        if (THREE.ClampToEdgeWrapping !== undefined) {
          tex.wrapS = THREE.ClampToEdgeWrapping;
          tex.wrapT = THREE.ClampToEdgeWrapping;
        }
        tex.generateMipmaps = false;
        if (THREE.LinearFilter) {
          tex.magFilter = (sampler && sampler.magFilter === 9728 && THREE.NearestFilter) ? THREE.NearestFilter : THREE.LinearFilter;
          tex.minFilter = (sampler && sampler.minFilter === 9728 && THREE.NearestFilter) ? THREE.NearestFilter : THREE.LinearFilter;
        }
      } else {
        if (sampler && sampler.wrapS !== undefined) tex.wrapS = mapWrap(sampler.wrapS);
        else if (THREE.RepeatWrapping !== undefined) tex.wrapS = THREE.RepeatWrapping;
        if (sampler && sampler.wrapT !== undefined) tex.wrapT = mapWrap(sampler.wrapT);
        else if (THREE.RepeatWrapping !== undefined) tex.wrapT = THREE.RepeatWrapping;
        if (THREE.LinearFilter) {
          tex.magFilter = (sampler && sampler.magFilter === 9728 && THREE.NearestFilter) ? THREE.NearestFilter : THREE.LinearFilter;
          if (sampler && (sampler.minFilter === 9728 || sampler.minFilter === 9984 || sampler.minFilter === 9986) && THREE.NearestFilter) {
            tex.minFilter = sampler.minFilter === 9728 ? THREE.NearestFilter : (THREE.NearestMipmapNearestFilter || THREE.LinearFilter);
          } else {
            tex.minFilter = THREE.LinearMipmapLinearFilter || THREE.LinearFilter;
          }
        }
      }
      tex.needsUpdate = true;
      loadedTextures[texIdx] = tex;
      flipYs.push(tex.flipY);
      textureCount += 1;
      return tex;
    }

    async function buildMaterial(matIdx, hasVertexColors) {
      const mDef = matIdx !== undefined ? (gltfJson.materials || [])[matIdx] : null;
      const pbr = (mDef && mDef.pbrMetallicRoughness) || {};
      const hasBaseTex = Boolean(pbr.baseColorTexture && pbr.baseColorTexture.index !== undefined);
      const bcf = Array.isArray(pbr.baseColorFactor)
        ? pbr.baseColorFactor
        : (hasBaseTex || hasVertexColors ? [1.0, 1.0, 1.0, 1.0] : [0.82, 0.78, 0.73, 1.0]);
      const defaultRoughness = hasBaseTex ? 0.78 : 0.58;
      const side = (mDef && mDef.doubleSided === true)
        ? THREE.DoubleSide
        : (THREE.FrontSide !== undefined ? THREE.FrontSide : THREE.DoubleSide);
      const matOpts = {
        color: new THREE.Color(bcf[0], bcf[1], bcf[2]),
        roughness: pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : defaultRoughness,
        metalness: pbr.metallicFactor !== undefined ? pbr.metallicFactor : 0.0,
        side: side,
      };
      if (hasVertexColors) {
        matOpts.vertexColors = true;
      }
      const alphaMode = mDef && mDef.alphaMode ? String(mDef.alphaMode).toUpperCase() : null;
      if (alphaMode === 'MASK') {
        matOpts.alphaTest = mDef.alphaCutoff !== undefined ? Number(mDef.alphaCutoff) : 0.5;
      } else if (alphaMode === 'BLEND') {
        matOpts.transparent = true;
        matOpts.opacity = bcf[3] !== undefined ? bcf[3] : 1.0;
      } else if (alphaMode !== 'OPAQUE' && bcf[3] !== undefined && bcf[3] < 0.99) {
        matOpts.transparent = true;
        matOpts.opacity = bcf[3];
      }
      if (pbr.baseColorTexture && pbr.baseColorTexture.index !== undefined) {
        const tex = await resolveTexture(pbr.baseColorTexture.index);
        if (tex) {
          matOpts.map = tex;
        }
      }
      return new THREE.MeshStandardMaterial(matOpts);
    }

    const contentGroup = new THREE.Group();
    let meshCount = 0;

    async function buildNode(nodeIdx) {
      const node = (gltfJson.nodes || [])[nodeIdx];
      if (!node) return null;
      const nodeGrp = new THREE.Group();
      if (Array.isArray(node.matrix) && node.matrix.length === 16) {
        const m4 = new THREE.Matrix4().fromArray(node.matrix);
        nodeGrp.applyMatrix4(m4);
      } else {
        if (Array.isArray(node.translation)) nodeGrp.position.fromArray(node.translation);
        if (Array.isArray(node.rotation)) nodeGrp.quaternion.fromArray(node.rotation);
        if (Array.isArray(node.scale)) nodeGrp.scale.fromArray(node.scale);
      }

      if (node.mesh !== undefined) {
        const mDef = gltfJson.meshes[node.mesh];
        for (const prim of (mDef && mDef.primitives) || []) {
          if (prim.mode !== undefined && prim.mode !== 4) continue;
          const geo = new THREE.BufferGeometry();
          const attrs = prim.attributes || {};
          if (attrs.POSITION === undefined) continue;
          const posAcc = _readAccessor(gltfJson, binBuffer, attrs.POSITION);
          geo.setAttribute('position', new THREE.BufferAttribute(posAcc.data, posAcc.itemSize));

          if (attrs.NORMAL !== undefined) {
            const normAcc = _readAccessor(gltfJson, binBuffer, attrs.NORMAL);
            geo.setAttribute('normal', new THREE.BufferAttribute(normAcc.data, normAcc.itemSize));
          }
          if (attrs.TEXCOORD_0 !== undefined) {
            const uvAcc = _readAccessor(gltfJson, binBuffer, attrs.TEXCOORD_0);
            geo.setAttribute('uv', new THREE.BufferAttribute(uvAcc.data, uvAcc.itemSize));
          }
          if (attrs.TEXCOORD_1 !== undefined) {
            const uv2Acc = _readAccessor(gltfJson, binBuffer, attrs.TEXCOORD_1);
            geo.setAttribute('uv2', new THREE.BufferAttribute(uv2Acc.data, uv2Acc.itemSize));
          }
          let hasVertexColors = false;
          if (attrs.COLOR_0 !== undefined) {
            const colAcc = _readAccessor(gltfJson, binBuffer, attrs.COLOR_0);
            geo.setAttribute('color', new THREE.BufferAttribute(colAcc.data, colAcc.itemSize));
            hasVertexColors = true;
          }
          if (prim.indices !== undefined) {
            const idxAcc = _readAccessor(gltfJson, binBuffer, prim.indices);
            geo.setIndex(new THREE.BufferAttribute(idxAcc.data, 1));
          }
          if (attrs.NORMAL === undefined) {
            geo.computeVertexNormals();
          }
          const mat = await buildMaterial(prim.material, hasVertexColors);
          const mesh = new THREE.Mesh(geo, mat);
          mesh.name = mDef.name || 'body_mesh';
          nodeGrp.add(mesh);
          meshCount += 1;
        }
      }
      for (const childIdx of node.children || []) {
        const child = await buildNode(childIdx);
        if (child) nodeGrp.add(child);
      }
      return nodeGrp;
    }

    const sceneDef = (gltfJson.scenes || [])[gltfJson.scene || 0];
    const rootNodes = (sceneDef && sceneDef.nodes) || (gltfJson.nodes || []).map((_, i) => i);
    for (const ni of rootNodes) {
      const built = await buildNode(ni);
      if (built) contentGroup.add(built);
    }

    const wrapper = new THREE.Group();
    wrapper.add(contentGroup);
    const box = new THREE.Box3().setFromObject(contentGroup);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    contentGroup.position.sub(center);

    const postBox = new THREE.Box3().setFromObject(wrapper);
    const postCenter = new THREE.Vector3();
    postBox.getCenter(postCenter);

    return {
      group: wrapper,
      metrics: {
        hY: Math.max(0.2, size.y),
        hX: Math.max(0.2, size.x),
        center: [postCenter.x, postCenter.y, postCenter.z],
      },
      stats: {
        meshes: meshCount,
        textures: textureCount,
        flipYs: flipYs,
      },
    };
  }

  function disposeObject3D(obj) {
    if (!obj || typeof obj.traverse !== 'function') return;
    const geos = new Set();
    const mats = new Set();
    const texs = new Set();
    obj.traverse((child) => {
      if (child.geometry && typeof child.geometry.dispose === 'function') {
        geos.add(child.geometry);
      }
      if (child.material) {
        const list = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of list) {
          if (!m) continue;
          for (const key of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap']) {
            if (m[key] && typeof m[key].dispose === 'function') texs.add(m[key]);
          }
          if (typeof m.dispose === 'function') mats.add(m);
        }
      }
    });
    texs.forEach((tx) => tx.dispose());
    mats.forEach((m) => m.dispose());
    geos.forEach((g) => g.dispose());
  }

  function extract16GirthRows(bodyScan, lang) {
    const circs = (bodyScan && bodyScan.circumferences_cm) || {};
    const dict = (lang === 'en') ? LABELS_EN : LABELS_VI;
    return ANATOMICAL_ORDER
      .filter((k) => circs[k] !== undefined && circs[k] !== null)
      .map((k) => ({ key: k, label: dict[k] || k, cm: Number(circs[k]) }));
  }

  function applyScanResultToClient(client, bodyScan, isInBodyClient) {
    if (!client) return { keptInBody: Boolean(isInBodyClient) };
    client.alpha_plus_scan = bodyScan;
    if (isInBodyClient) {
      return { keptInBody: true };
    }
    const circs = (bodyScan && bodyScan.circumferences_cm) || {};
    client.girth_src = client.girth_src || {};
    const out = { keptInBody: false };
    ['waist', 'neck', 'hip'].forEach((k) => {
      if (Number.isFinite(Number(circs[k])) && Number(circs[k]) > 0) {
        client[k] = Number(circs[k]);
        client.girth_src[k] = 'plus_pro';
        out[k] = client[k];
      }
    });
    return out;
  }

  function _esc(v) {
    return String(v == null ? '' : v)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderLeft16Html(scanState, lang, tFn) {
    if (!scanState || !scanState.bodyScan) return '';
    const rows = extract16GirthRows(scanState.bodyScan, lang);
    if (!rows.length) return '';
    const t = tFn || ((_, fb) => fb);
    const title = t('pp_left_title', '16 vòng đo cơ thể chính (3D)');
    const badge = scanState.testOnly
      ? `<span class="pp-pill warn">TEST_ONLY</span>`
      : `<span class="pp-pill">3D Plus/PRO</span>`;
    const sub = scanState.assetsDeleted
      ? `<div class="pp-sub">${_esc(t('pp_left_retained', 'Đã xoá file 3D/ảnh trên server · Giữ 16 số đo'))}</div>`
      : `<div class="pp-sub">${_esc(t('pp_left_live', 'Dựng từ 4 góc chụp · Chuẩn khớp A-pose'))}</div>`;

    const cells = rows.map((r) => {
      const mainCls = (r.key === 'neck' || r.key === 'chest' || r.key === 'waist' || r.key === 'hip') ? ' main' : '';
      return `<div class="pp-gitem${mainCls}"><span class="k">${_esc(r.label)}</span><b class="v">${r.cm.toFixed(1)} cm</b></div>`;
    }).join('');

    return `<div class="pp-left-card">
      <div class="pp-left-hd">
        <span>${_esc(title)}</span>
        ${badge}
      </div>
      ${sub}
      <div class="pp-ggrid">${cells}</div>
    </div>`;
  }

  function renderReport16Html(scanState, lang, tFn) {
    if (!scanState || !scanState.bodyScan) return '';
    const rows = extract16GirthRows(scanState.bodyScan, lang);
    if (!rows.length) return '';
    const t = tFn || ((_, fb) => fb);
    const trs = rows.map((r) => `<tr><td>${_esc(r.label)}</td><td style="text-align:right;font-weight:700">${r.cm.toFixed(1)} cm</td></tr>`).join('');
    return `<div class="pb" style="margin-top:14px">
      <div style="font-size:12px;font-weight:800;text-transform:uppercase;color:#8a6a2f;margin-bottom:6px">${_esc(t('pp_pdf_title', 'Số đo 16 vòng cơ thể (Alpha Scan Plus/PRO 3D)'))}</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;background:#fff;border:1px solid #e9e3d6">
        <thead><tr style="background:#f3ecdd"><th style="text-align:left;padding:6px 8px">${_esc(t('pp_pdf_part', 'Vùng đo'))}</th><th style="text-align:right;padding:6px 8px">${_esc(t('pp_pdf_cm', 'Chu vi (cm)'))}</th></tr></thead>
        <tbody>${trs}</tbody>
      </table>
    </div>`;
  }

  // ===== Browser Runtime Controller =====
  const state = {
    bridge: null,
    serverCfg: null,
    _explicitlySavedCfg: false,
    _lastHealthOk: null,
    health: null,
    healthStatus: null,
    serverMode: null,
    jobId: null,
    lastJobId: null,
    lastBaseUrl: null,
    artifactUrlsFetchedAt: 0,
    jobStatus: null,
    bodyScan: null,
    backendsUsed: null,
    artifacts: null,
    deletionAudit: null,
    privacyLevel: null,
    reconstructionMode: DEFAULT_RECONSTRUCTION_MODE,
    activeReconstructionMode: null,
    effectiveReconstructionMode: null,
    proportionAudit: null,
    warnings: [],
    duplicateWarning: null,
    meshViewLayer: 'outer',
    outerGlbBuffer: null,
    coreGlbBuffer: null,
    coreMeshLoading: false,
    testOnly: true,
    assetsDeleted: false,
    realActive: false,
    hasLoadedMesh: false,
    meshMetrics: null,
    meshGroup: null,
    slots: [null, null, null, null],
    extraPhotos: [],
    faceFile: null,
    pollTimer: null,
    stylesInjected: false,
  };

  function _t(key, fallback) {
    if (state.bridge && typeof state.bridge.t === 'function') return state.bridge.t(key, fallback);
    return fallback;
  }

  function _lang() {
    if (state.bridge && typeof state.bridge.getLang === 'function') return state.bridge.getLang();
    return 'vi';
  }

  function _queryConfigOverrides() {
    try {
      const loc = rootScope.location;
      const search = loc && typeof loc.search === 'string' ? loc.search : '';
      if (!search) return null;
      const cleanSearch = search.replace(/^\?/, '');
      if (!cleanSearch) return null;
      const sp = typeof URLSearchParams === 'function'
        ? new URLSearchParams(cleanSearch)
        : (typeof URL === 'function' ? new URL('http://localhost/?' + cleanSearch).searchParams : null);
      if (!sp) return null;
      const rawUrl = (sp.get('gpu_url') || sp.get('server') || '').trim().replace(/\/+$/, '');
      const hasKeyParam = sp.has('api_key');
      const rawKey = hasKeyParam ? String(sp.get('api_key') || '').trim() : null;
      if (!rawUrl && !hasKeyParam) return null;
      return { url: rawUrl || null, apiKey: rawKey };
    } catch (_) {
      return null;
    }
  }

  function _locationOrigin(loc) {
    if (!loc || (loc.protocol !== 'http:' && loc.protocol !== 'https:')) return '';
    const rawOrigin = loc.origin || (loc.host ? (loc.protocol + '//' + loc.host) : (loc.hostname ? (loc.protocol + '//' + loc.hostname + (loc.port ? ':' + loc.port : '')) : ''));
    return String(rawOrigin || '').trim().replace(/\/+$/, '');
  }

  function _defaultServerUrl() {
    try {
      const q = _queryConfigOverrides();
      if (q && q.url) return q.url;
      const loc = rootScope.location;
      const origin = _locationOrigin(loc);
      if (origin && ((loc.pathname && loc.pathname.startsWith('/lab')) || !isLoopbackUrl(origin))) {
        return origin;
      }
    } catch (_) {}
    return 'http://127.0.0.1:8000';
  }

  function _isStaleSavedUrlForOrigin(savedUrl, currentOrigin, loc) {
    if (!savedUrl || !currentOrigin || isLoopbackUrl(currentOrigin)) return false;
    if (isLoopbackUrl(savedUrl)) return true;
    try {
      const u = new URL(savedUrl);
      const host = (u.hostname || '').toLowerCase();
      const normSaved = savedUrl.replace(/\/+$/, '').toLowerCase();
      const normCurr = currentOrigin.replace(/\/+$/, '').toLowerCase();
      if (normSaved === normCurr) return false;
      if (host.endsWith('.trycloudflare.com') || /ngrok|loca\.lt|gradio\.live/i.test(host)) {
        return true;
      }
      if (loc && typeof loc.pathname === 'string' && loc.pathname.startsWith('/lab')) {
        return true;
      }
    } catch (_) {
      return true;
    }
    return false;
  }

  function loadCfg() {
    const q = _queryConfigOverrides();
    const loc = rootScope.location;
    const currentOrigin = _locationOrigin(loc);
    const nonLoopbackOrigin = currentOrigin && !isLoopbackUrl(currentOrigin) ? currentOrigin : '';

    if (state.serverCfg && state._explicitlySavedCfg && !q && !nonLoopbackOrigin) {
      return state.serverCfg;
    }

    let savedUrl = '';
    let savedKey = '';
    try {
      const raw = rootScope.localStorage && rootScope.localStorage.getItem(CFG_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          if (parsed.url) savedUrl = String(parsed.url).trim().replace(/\/+$/, '');
          if (parsed.apiKey) savedKey = String(parsed.apiKey);
        }
      }
    } catch (_) {}

    let healed = false;
    if (nonLoopbackOrigin && savedUrl && _isStaleSavedUrlForOrigin(savedUrl, nonLoopbackOrigin, loc)) {
      savedUrl = nonLoopbackOrigin;
      healed = true;
    }

    const finalUrl = (q && q.url) || savedUrl || _defaultServerUrl();
    const finalKey = (q && q.apiKey !== null && q.apiKey !== undefined) ? q.apiKey : (savedKey || '');
    state.serverCfg = { url: finalUrl, apiKey: finalKey };

    if (healed || (q && (q.url || q.apiKey !== null))) {
      try {
        rootScope.localStorage.setItem(CFG_KEY, JSON.stringify(state.serverCfg));
      } catch (_) {}
    }
    return state.serverCfg;
  }

  function saveCfg(url, apiKey) {
    const cleanUrl = String(url || '').trim().replace(/\/+$/, '') || _defaultServerUrl();
    const cleanKey = String(apiKey || '').trim();
    state.serverCfg = { url: cleanUrl, apiKey: cleanKey };
    state._explicitlySavedCfg = true;
    try {
      rootScope.localStorage.setItem(CFG_KEY, JSON.stringify(state.serverCfg));
    } catch (_) {}
    return state.serverCfg;
  }

  function resolveServerUrl(base, maybeRelative) {
    if (!maybeRelative) return '';
    try {
      return new URL(maybeRelative, base.replace(/\/+$/, '') + '/').toString();
    } catch (_) {
      return maybeRelative;
    }
  }

  function _headers() {
    const cfg = loadCfg();
    return cfg.apiKey ? { Authorization: 'Bearer ' + cfg.apiKey } : {};
  }

  function buildRequestHeaders(urlStr, extraHeaders, includeAuth) {
    const headers = Object.assign({}, extraHeaders || {});
    if (includeAuth) {
      const cfg = loadCfg();
      if (cfg.apiKey) headers.Authorization = 'Bearer ' + cfg.apiKey;
    }
    if (isNgrokUrl(urlStr)) {
      headers['ngrok-skip-browser-warning'] = 'true';
    }
    return headers;
  }

  async function _performFetch(fetchFn, url, fetchOpts) {
    const o = fetchOpts || {};
    const method = (o.method || 'GET').toUpperCase();
    const headers = buildRequestHeaders(url, o.headers, Boolean(o.includeAuth));
    const hasHeaders = Object.keys(headers).length > 0;
    if (method === 'GET' && o.body === undefined && !hasHeaders) {
      return fetchFn(url);
    }
    const init = { method };
    if (hasHeaders) init.headers = headers;
    if (o.body !== undefined) init.body = o.body;
    return fetchFn(url, init);
  }

  function _injectStyles() {
    if (state.stylesInjected || typeof document === 'undefined' || !document.head) return;
    state.stylesInjected = true;
    const style = document.createElement('style');
    style.textContent = `
      .pp-left-card{background:#fbf8f2;border:1px solid #d9cbb1;border-radius:12px;padding:11px 12px;margin:10px 0 12px}
      .pp-left-hd{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:12px;font-weight:800;color:#6e5324;text-transform:uppercase;letter-spacing:.4px}
      .pp-pill{font-size:10px;padding:2px 7px;border-radius:999px;background:#f3ecdd;color:#8a6a2f;border:1px solid #dfd2b8;font-weight:700}
      .pp-pill.ok{background:#ecfdf5;color:#065f46;border-color:#6ee7b7}
      .pp-pill.warn{background:#fff7ed;color:#9a3412;border-color:#fdba74}
      .pp-pill.bad{background:#fef2f2;color:#991b1b;border-color:#fca5a5}
      .pp-sub{font-size:11px;color:#7a7266;margin:4px 0 8px}
      .pp-ggrid{display:grid;grid-template-columns:1fr 1fr;gap:5px 8px}
      .pp-gitem{display:flex;justify-content:space-between;align-items:baseline;background:#fff;border:1px solid #e9e3d6;border-radius:8px;padding:5px 7px;font-size:11.5px;gap:4px}
      .pp-gitem.main{background:#fffdf8;border-color:#d8c49e}
      .pp-gitem .k{color:#5f574c;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .pp-gitem .v{color:#201d18;font-variant-numeric:tabular-nums;white-space:nowrap}
      .pp-detail-card{margin-top:10px;padding:11px;border-radius:11px;background:#fbf8f2;border:1px solid #e0d4be;font-size:12px}
      .pp-detail-actions{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
      .pp-audit-box{margin-top:8px;padding:8px 10px;border-radius:8px;background:#fff;border:1px solid #e6dcc8}
      .pp-audit-row{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:11.5px;padding:2px 0}
      .pp-warn-list{margin-top:8px;padding:8px 10px;border-radius:8px;background:#fff7ed;border:1px solid #fdba74;color:#9a3412;font-size:11.5px}
      .pp-warn-list ul{margin:4px 0 0;padding-left:16px}
      .pp-btn{border:1px solid #d6c8ad;background:#fff;color:#201d18;border-radius:8px;padding:6px 10px;font-size:12px;font-weight:600;cursor:pointer}
      .pp-btn.pri{background:#8a6a2f;border-color:#8a6a2f;color:#fff}
      .pp-btn.danger{background:#fef2f2;border-color:#fca5a5;color:#991b1b}
      .pp-btn:disabled{opacity:.45;cursor:not-allowed}
      #plusModal{position:fixed;inset:0;z-index:100000;background:rgba(32,29,24,.55);display:none;align-items:center;justify-content:center;padding:16px;overflow:auto}
      #plusModal .pp-sheet{width:min(760px,100%);max-height:92vh;overflow:auto;background:#fbf8f2;border:1px solid #e0d4be;border-radius:16px;box-shadow:0 24px 60px rgba(0,0,0,.35);color:#201d18}
      #plusModal .pp-mhd{padding:14px 18px;background:linear-gradient(120deg,#5a4524,#8a6a34);color:#fff;display:flex;justify-content:space-between;align-items:center}
      #plusModal .pp-mbd{padding:16px 18px}
      .pp-protocol-banner{background:#fffaf0;border:1px solid #dcc89a;border-radius:10px;padding:9px 11px;font-size:12px;color:#5a4524;margin-bottom:10px}
      .pp-hw-hint{font-size:11px;color:#7a7266;margin:-4px 0 10px}
      .pp-slots{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
      @media(max-width:640px){.pp-slots{grid-template-columns:1fr 1fr}}
      .pp-slot{border:1.5px dashed #cbbba0;border-radius:10px;background:#fff;padding:8px;text-align:center;min-height:112px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;position:relative;cursor:pointer}
      .pp-slot img{width:100%;height:74px;object-fit:cover;border-radius:6px}
      .pp-slot .rm{position:absolute;top:4px;right:4px;border:0;background:rgba(0,0,0,.65);color:#fff;width:20px;height:20px;border-radius:50%;cursor:pointer;font-size:11px;line-height:20px}
      .pp-privs,.pp-rmodes{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}
      @media(max-width:640px){.pp-privs,.pp-rmodes{grid-template-columns:1fr}}
      .pp-priv,.pp-rmode{border:1.5px solid #dfd3bd;border-radius:10px;background:#fff;padding:9px;cursor:pointer;text-align:left;font-size:12px;display:block}
      .pp-priv.on,.pp-rmode.on{border-color:#8a6a2f;background:#fffaf0;box-shadow:0 0 0 2px rgba(138,106,47,.18)}
      .pp-flag{padding:9px 11px;border-radius:9px;font-size:12px;margin-top:8px;display:none}
      .pp-flag.warn{display:block;background:#fff7ed;border:1px solid #fdba74;color:#9a3412}
      .pp-flag.ok{display:block;background:#ecfdf5;border:1px solid #6ee7b7;color:#065f46}
      .pp-prog{margin-top:10px;display:none}
      .pp-pbar{height:8px;background:#e4dccb;border-radius:99px;overflow:hidden;margin-top:4px}
      .pp-pbar i{display:block;height:100%;width:0;background:#8a6a2f;transition:width .2s}
    `;
    document.head.appendChild(style);
  }

  function _clearSlotsObjectUrls() {
    document.querySelectorAll('#plusModal img[data-objurl]').forEach((im) => {
      try { URL.revokeObjectURL(im.dataset.objurl); } catch (_) {}
    });
  }

  function _ensureModal() {
    if (typeof document === 'undefined' || !document.body) return null;
    let el = document.getElementById('plusModal');
    if (el) return el;
    _injectStyles();
    el = document.createElement('div');
    el.id = 'plusModal';
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-modal', 'true');
    document.body.appendChild(el);
    return el;
  }

  function _cardinalSlotsCount() {
    return state.slots.filter(Boolean).length;
  }

  function _collectedPhotosWithRoles() {
    const photos = [];
    const slotRoles = [];
    state.slots.forEach((f, idx) => {
      if (f) {
        photos.push(f);
        slotRoles.push(ANGLE_SLOTS[idx] ? ANGLE_SLOTS[idx].key : 'extra');
      }
    });
    state.extraPhotos.forEach((f) => {
      if (f && photos.length < 8) {
        photos.push(f);
        const inferred = inferCardinalRoleFromFilename(f && f.name);
        slotRoles.push(inferred !== null && ANGLE_SLOTS[inferred] ? ANGLE_SLOTS[inferred].key : 'extra');
      }
    });
    return { photos, slotRoles };
  }

  function _collectedPhotos() {
    return _collectedPhotosWithRoles().photos;
  }

  function _paintDuplicateWarning(allPhotos) {
    const dupInfo = findDuplicateFileSignatures(allPhotos);
    let msg = '';
    if (dupInfo.hasDuplicates) {
      const tpl = _t(
        'pp_dup_warn',
        'Cảnh báo trùng ảnh (trùng nội dung hoặc tên & dung lượng): {files}. Tải ảnh trùng góc có thể khiến server tự hạ về chế độ calibrated_mhr.'
      );
      msg = tpl.replace('{files}', dupInfo.duplicateNames.join(', '));
    }
    state.duplicateWarning = msg || null;
    if (typeof document !== 'undefined') {
      const dupEl = document.getElementById('ppDupWarn');
      if (dupEl) {
        dupEl.className = msg ? 'pp-flag warn' : 'pp-flag';
        dupEl.textContent = msg;
      }
    }
    return dupInfo;
  }

  function _updateDuplicateWarning() {
    const allPhotos = _collectedPhotos();
    const dupInfo = _paintDuplicateWarning(allPhotos);
    if (allPhotos.some((f) => f && !f._contentFingerprint && (typeof f.slice === 'function' || typeof f.arrayBuffer === 'function'))) {
      enrichFileContentFingerprints(allPhotos).then(() => {
        _paintDuplicateWarning(_collectedPhotos());
      }).catch(() => {});
    }
    return dupInfo;
  }

  function _formatHealthBackendsTag(backends) {
    if (!backends || typeof backends !== 'object') return '';
    const tags = [];
    if (backends.gpu_name && String(backends.gpu_name) !== 'CPU') {
      tags.push(String(backends.gpu_name));
    }
    if (backends.segmentation) {
      tags.push('Seg:' + String(backends.segmentation));
    }
    const rast = backends.rasterizer || backends.texture_rasterizer;
    if (rast) {
      tags.push(rast === 'pytorch3d' ? 'PyTorch3D' : String(rast));
    } else if (backends.pytorch3d_available) {
      tags.push('PyTorch3D');
    }
    if (backends.moge2_available || backends.moge2) {
      tags.push('MoGe2');
    }
    return tags.join(' · ');
  }

  async function onHealthCheck(urlOverride, keyOverride, opts) {
    const o = opts || {};
    const cfg = urlOverride !== undefined ? saveCfg(urlOverride, keyOverride) : loadCfg();
    const fetchFn = o.fetchFn || rootScope.fetch;
    const resEl = document.getElementById('ppHealthStatus');
    const akResEl = document.getElementById('akGpuHealthOut');
    const setMsg = (txt, tone) => {
      const color = tone === 'ok' ? '#065f46' : (tone === 'loading' ? '#b07a1e' : '#991b1b');
      [resEl, akResEl].forEach((n) => {
        if (!n) return;
        n.textContent = txt;
        n.style.color = color;
      });
    };
    setMsg(_t('pp_checking', 'Đang kiểm tra kết nối...'), 'ok');
    try {
      if (typeof fetchFn !== 'function') throw new Error('fetch is not available');
      const r = await _performFetch(fetchFn, cfg.url + '/health', { method: 'GET', includeAuth: false });
      const j = await r.json().catch(() => null);
      const info = classifyHealthPayload(j, r.status);
      if (info.state === 'error') {
        throw new Error(info.detail || ('HTTP ' + r.status));
      }
      state.health = j;
      state.healthStatus = info.state;
      state._lastHealthOk = true;
      state.serverMode = info.mode || state.serverMode;
      state.testOnly = info.testOnly;
      const backendTag = j && j.backends ? _formatHealthBackendsTag(j.backends) : '';
      if (info.isLoading) {
        const extra = info.detail ? (' (' + info.detail + ')') : '';
        const bSuffix = backendTag ? (' · ' + backendTag) : '';
        const msg = '⏳ ' + _t('pp_health_loading', 'Server đã kết nối — đang nạp mô hình AI 3D vào GPU, vui lòng đợi...') + extra + bSuffix;
        setMsg(msg, 'loading');
        _refreshGuardBanner();
        return j;
      }
      const modeTxt = info.mode === 'mock'
        ? _t('pp_ready_mock', 'Sẵn sàng (Mock test CPU)')
        : _t('pp_ready_gpu', 'Sẵn sàng (GPU thật)');
      const testTag = state.testOnly ? ' · TEST_ONLY' : '';
      const bSuffix = backendTag ? (' · ' + backendTag) : '';
      const needsKey = Boolean(j && j.auth_required && !cfg.apiKey);
      const authSuffix = needsKey ? (' · 🔑 ' + _t('pp_auth_hint', 'Server yêu cầu API Key — nhập API Key trước khi gửi dựng 3D.')) : '';
      setMsg('● ' + modeTxt + testTag + bSuffix + authSuffix, needsKey ? 'loading' : 'ok');
      _refreshGuardBanner();
      return j;
    } catch (err) {
      state.healthStatus = 'error';
      state._lastHealthOk = false;
      let msg = _t('pp_conn_fail', 'Không kết nối được server: ') + (err && err.message ? err.message : err);
      try {
        if (rootScope.location && rootScope.location.protocol === 'https:' && cfg.url.startsWith('http://')) {
          msg += ' · ' + _t('pp_mixed_hint', 'Trang đang chạy HTTPS — cần link Cloudflare Tunnel HTTPS nếu server nằm từ xa.');
        }
      } catch (_) {}
      setMsg('✕ ' + msg, 'error');
      return null;
    }
  }

  function _switchToServerConsent() {
    if (rootScope.SAFETY && typeof rootScope.SAFETY === 'object') {
      rootScope.SAFETY.consent = 'server';
    }
    if (state.bridge && typeof state.bridge.setConsentMode === 'function') {
      state.bridge.setConsentMode('server');
    }
    if (typeof rootScope.showConsentBanner === 'function') {
      try { rootScope.showConsentBanner(); } catch (_) {}
    }
    _refreshGuardBanner();
  }

  function _refreshGuardBanner() {
    _updateDuplicateWarning();
    const warnEl = document.getElementById('ppGuardMsg');
    const goBtn = document.getElementById('ppSubmitBtn');
    if (!warnEl || !goBtn) return;

    const client = state.bridge ? state.bridge.getClient() : null;
    const consentMode = state.bridge
      ? state.bridge.getConsentMode()
      : ((rootScope.SAFETY && rootScope.SAFETY.consent) || 'local');
    const hIn = document.getElementById('ppHeight');
    const consentCk = document.getElementById('ppConsentChk');
    const urlIn = document.getElementById('ppServerUrl');
    const cfg = loadCfg();
    const urlVal = urlIn ? urlIn.value : cfg.url;
    const photoCount = _collectedPhotos().length;
    const cardinalCount = _cardinalSlotsCount();

    const check = validatePrivacyAndConnection({
      consentMode: consentMode,
      serverUrl: urlVal,
      privacyLevel: state.privacyLevel,
      reconstructionMode: state.reconstructionMode,
      cardinalSlotsCount: cardinalCount,
      consentChecked: consentCk ? consentCk.checked : false,
      photoCount: photoCount,
      heightCm: hIn ? Number(hIn.value) : Number(client && client.height || 0),
    });

    const msgMap = {
      pp_err_local_only: _t('pp_err_local_only', 'Chế độ LOCAL_ONLY chỉ cho phép gửi ảnh về máy chủ trên chính thiết bị này (localhost / 127.0.0.1). Hãy đổi URL về localhost hoặc chuyển đồng thuận khách sang Lưu hồ sơ ở cổng đầu.'),
      pp_err_photos: _t('pp_err_photos', 'Vui lòng chọn 1–8 ảnh toàn thân (khuyên dùng đủ 4 góc: Chính diện, Nghiêng phải, Phía sau, Nghiêng trái).'),
      pp_err_4cardinal: _t('pp_err_4cardinal', 'Chế độ Mặt–Mũi–Tóc / 2 lớp cần đủ cả 4 góc chính (Chính diện · Nghiêng phải · Sau · Nghiêng trái). Hãy chọn đủ 4 góc hoặc đổi sang Lõi chuẩn đo nhanh.'),
      pp_err_mode: _t('pp_err_mode', 'Chế độ dựng 3D không hợp lệ.'),
      pp_err_height: _t('pp_err_height', 'Chiều cao khách cần nằm trong khoảng 80–230 cm.'),
      pp_err_privacy: _t('pp_err_privacy', 'Vui lòng chọn 1 trong 3 mức hiển thị khuôn mặt bên trên.'),
      pp_err_consent: _t('pp_err_consent', 'Cần tick ô xác nhận khách hàng đã đồng ý dựng hình 3D trước khi gửi.'),
      pp_err_url: _t('pp_err_url', 'Địa chỉ GPU Server không hợp lệ (cần http:// hoặc https://).'),
    };

    if (!client || !client.sex) {
      warnEl.className = 'pp-flag warn';
      const clientMsg = _t('pp_err_client', 'Vui lòng mở cổng tạo/chọn khách hàng có giới tính trước khi quét.');
      warnEl.textContent = clientMsg;
      warnEl.innerHTML = _esc(clientMsg);
      goBtn.disabled = true;
      return;
    }

    if (!check.ok) {
      const showInlineWarn = (
        check.reasonKey === 'pp_err_local_only' ||
        check.reasonKey === 'pp_err_url' ||
        check.reasonKey === 'pp_err_mode' ||
        (check.reasonKey === 'pp_err_4cardinal' && photoCount > 0)
      );
      warnEl.className = showInlineWarn ? 'pp-flag warn' : 'pp-flag';
      const warnText = showInlineWarn ? (msgMap[check.reasonKey] || '') : '';
      if (check.reasonKey === 'pp_err_local_only') {
        const switchLabel = _t('pp_switch_server_consent_btn', '⚡ Bật chế độ Server GPU (cho phép gửi ảnh tới Colab)');
        warnEl.textContent = warnText + ' ' + switchLabel;
        warnEl.innerHTML = `<div>${_esc(warnText)}</div><div style="margin-top:6px"><button type="button" id="ppBtnSwitchServerConsent" class="pp-btn pri">${_esc(switchLabel)}</button></div>`;
        const switchBtn = (typeof warnEl.querySelector === 'function' && warnEl.querySelector('#ppBtnSwitchServerConsent'))
          || (typeof document !== 'undefined' && document.getElementById('ppBtnSwitchServerConsent'));
        if (switchBtn) {
          switchBtn.onclick = _switchToServerConsent;
        }
      } else {
        warnEl.textContent = warnText;
        warnEl.innerHTML = _esc(warnText);
      }
      goBtn.disabled = true;
      goBtn.title = msgMap[check.reasonKey] || '';
    } else {
      warnEl.className = 'pp-flag';
      warnEl.textContent = '';
      warnEl.innerHTML = '';
      goBtn.disabled = false;
      goBtn.title = '';
    }
  }

  function _renderSlots() {
    const wrap = document.getElementById('ppSlotsGrid');
    if (!wrap) return;
    _clearSlotsObjectUrls();
    const lang = _lang();
    wrap.innerHTML = '';

    ANGLE_SLOTS.forEach((slot, idx) => {
      const box = document.createElement('div');
      box.className = 'pp-slot';
      const file = state.slots[idx];
      if (file) {
        const img = document.createElement('img');
        const u = URL.createObjectURL(file);
        img.src = u;
        img.dataset.objurl = u;
        img.alt = slot[lang] || slot.vi;
        const rm = document.createElement('button');
        rm.type = 'button';
        rm.className = 'rm';
        rm.textContent = '×';
        rm.onclick = (ev) => {
          ev.stopPropagation();
          state.slots[idx] = null;
          _renderSlots();
          _refreshGuardBanner();
        };
        const lbl = document.createElement('span');
        lbl.style.fontSize = '11px';
        lbl.style.fontWeight = '700';
        lbl.textContent = slot[lang] || slot.vi;
        box.appendChild(rm);
        box.appendChild(img);
        box.appendChild(lbl);
      } else {
        box.innerHTML = `<div style="font-size:20px;color:#8a6a2f">+</div>
          <div style="font-size:11.5px;font-weight:700">${_esc(slot[lang] || slot.vi)}</div>
          <div style="font-size:10.5px;color:#8b8173">${_esc(_t('pp_pick_photo', 'Chọn ảnh'))}</div>`;
      }
      box.onclick = () => {
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = 'image/*';
        inp.onchange = () => {
          if (inp.files && inp.files[0]) {
            state.slots[idx] = inp.files[0];
            _renderSlots();
            _refreshGuardBanner();
          }
        };
        inp.click();
      };
      wrap.appendChild(box);
    });

    const extraInfo = document.getElementById('ppExtraCount');
    if (extraInfo) {
      const cardinalCount = _cardinalSlotsCount();
      const totalCount = _collectedPhotos().length;
      extraInfo.textContent = state.extraPhotos.length
        ? `${cardinalCount}/4 ${_t('pp_cardinal_short', 'góc chính')} · +${state.extraPhotos.length} ${_t('pp_extra_added', 'ảnh góc phụ')}`
        : `${cardinalCount}/4 ${_t('pp_cardinal_short', 'góc chính')} (${totalCount}/8 ${_t('pp_photos_unit', 'ảnh')})`;
    }
    _updateDuplicateWarning();
  }

  function _buildModalHtml() {
    const cfg = loadCfg();
    const client = state.bridge ? state.bridge.getClient() : null;
    const hVal = client && Number.isFinite(Number(client.height)) ? Number(client.height) : 168;
    const wVal = client && Number.isFinite(Number(client.weight)) ? Number(client.weight) : 0;
    const sexTxt = client && client.sex === 'male' ? _t('sex_m', 'Nam') : _t('sex_f', 'Nữ');
    const nm = client && client.name ? client.name : _t('pp_guest', 'Khách mới');
    const faceDisplay = state.privacyLevel === 3 ? 'block' : 'none';
    const loc = rootScope.location;
    const isOnLabOrigin = Boolean(
      loc &&
      ((typeof loc.pathname === 'string' && loc.pathname.startsWith('/lab')) ||
        (_locationOrigin(loc) && !isLoopbackUrl(_locationOrigin(loc))))
    );
    const hasCustomCfg = Boolean((rootScope.localStorage && rootScope.localStorage.getItem(CFG_KEY)) || isOnLabOrigin);
    const srvAutoBadge = hasCustomCfg
      ? ` <span class="pp-pill ok" style="font-weight:700">${_esc(_t('pp_srv_auto_badge', 'Tự động kết nối — không cần điền'))}</span>`
      : '';
    const apiKeyPh = _t('pp_apikey_ph', 'Để trống (chỉ điền nếu bạn tự đặt mật khẩu trên Colab)');
    const activeRmode = RECONSTRUCTION_MODES.includes(state.reconstructionMode)
      ? state.reconstructionMode
      : DEFAULT_RECONSTRUCTION_MODE;

    return `<div class="pp-sheet">
      <div class="pp-mhd">
        <div>
          <b style="font-size:15px">${_esc(_t('pp_modal_title', 'Alpha Scan Plus/PRO · Dựng 3D & 16 vòng đo'))}</b>
          <div style="font-size:11.5px;opacity:.9">${_esc(nm)} · ${_esc(sexTxt)} · ${_esc(_t('pp_internal_note', 'Bản test nội bộ — Ảnh gốc xoá ngay sau khi xử lý'))}</div>
        </div>
        <button type="button" id="ppCloseBtn" class="pp-btn" style="padding:4px 10px">✕</button>
      </div>
      <div class="pp-mbd">
        <div id="ppProtocolBanner" class="pp-protocol-banner">
          <b>${_esc(_t('pp_protocol_label', 'Chuẩn chụp:'))}</b> ${_esc(_t('pp_protocol_hint', 'Quy chuẩn: mặc đồ tập sát form · buộc tóc gọn lộ cổ–vai · "Nghiêng phải/trái" = xoay vai PHẢI/TRÁI của người mẫu hướng về camera.'))}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px">
          <div>
            <label style="font-size:11.5px;color:#5f574c;display:block;margin-bottom:3px">${_esc(_t('pp_height_lbl', 'Chiều cao chuẩn hoá (cm) *'))}</label>
            <input id="ppHeight" type="number" step="0.5" min="80" max="230" value="${hVal}" class="ik-in">
          </div>
          <div>
            <label style="font-size:11.5px;color:#5f574c;display:block;margin-bottom:3px">${_esc(_t('pp_weight_lbl', 'Cân nặng hiện tại (kg, tuỳ chọn)'))}</label>
            <input id="ppWeight" type="number" step="0.1" min="0" max="250" value="${wVal || ''}" placeholder="0" class="ik-in">
          </div>
        </div>
        <div class="pp-hw-hint">${_esc(_t('pp_hw_hint', 'Chiều cao chuẩn (80–230 cm) giúp định tỉ lệ mét thật; cân nặng để trống = tự ước lượng từ dáng 3D, nhập số = kiểm tra sai lệch %.'))}</div>

        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
          <b style="font-size:13px">${_esc(_t('pp_upload_title', '1. Tải 4 ảnh toàn thân (Chính diện · Vai phải → camera · Sau · Vai trái → camera)'))}</b>
          <div style="display:flex;align-items:center;gap:8px">
            <span id="ppExtraCount" style="font-size:11.5px;color:#7a7266">0/4</span>
            <button type="button" id="ppBatchBtn" class="pp-btn">${_esc(_t('pp_batch_btn', 'Chọn nhanh 4–8 ảnh'))}</button>
          </div>
        </div>
        <div id="ppSlotsGrid" class="pp-slots"></div>
        <div id="ppDupWarn" class="pp-flag" role="status" aria-live="polite"></div>

        <div id="ppFaceRow" style="display:${faceDisplay};margin-top:10px;background:#fff;border:1px solid #e0d4be;border-radius:10px;padding:9px 11px">
          <label style="font-size:12px;font-weight:700;display:block;margin-bottom:4px">${_esc(_t('pp_face_lbl', 'Ảnh cận mặt chính diện (~1m, tuỳ chọn cho Mức 3)'))}</label>
          <input id="ppFaceInput" type="file" accept="image/*" style="font-size:12px">
        </div>

        <div style="margin-top:14px;background:#f8f9fa;border:1.5px solid #d0d7de;border-radius:11px;padding:12px 14px">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:16px">🎯</span>
            <b style="font-size:13px;color:#1f2328">${_esc(_t('pp_single_mode_title', 'Chế độ Chuẩn: Tái tạo Hình thể 3D Thực tế & 16 Vòng đo'))}</b>
          </div>
          <div style="font-size:11.5px;color:#57606a;margin-top:6px;line-height:1.55">
            • <b>Hình thể thực tế:</b> AI trích xuất và tối ưu chính xác tỉ lệ vai, ngực, eo, hông, đùi theo ảnh chụp thực tế.<br>
            • <b>Khuôn mặt mỹ thuật:</b> Mô phỏng đường nét đầu, hàm và góc nghiêng chuẩn giải phẫu, tinh gọn, không biến dạng.<br>
            • <b>Vật liệu Studio Clay:</b> Tone xám đất sét điêu khắc kèm quần thể thao than chì tối giản, tôn rõ từng đường cong cơ thể.
          </div>
        </div>

        <label style="display:flex;align-items:flex-start;gap:8px;margin-top:11px;font-size:12px;background:#fff;border:1px solid #dfd3bd;border-radius:10px;padding:9px 11px;cursor:pointer">
          <input id="ppConsentChk" type="checkbox" style="margin-top:2px">
          <span>${_esc(_t('pp_consent_lbl', 'Khách hàng đã được giải thích và đồng ý dựng hình 3D theo mức đã chọn; ảnh gốc sẽ xoá ngay sau khi xử lý.'))}</span>
        </label>

        <details style="margin-top:10px;background:#fff;border:1px solid #dfd3bd;border-radius:10px;padding:8px 11px" ${hasCustomCfg ? '' : 'open'}>
          <summary style="cursor:pointer;font-size:12px;font-weight:700">${_esc(_t('pp_srv_summary', 'Cấu hình Server GPU Alpha Scan'))}${srvAutoBadge}</summary>
          <div style="display:grid;grid-template-columns:1.4fr 1fr auto;gap:8px;margin-top:8px;align-items:end">
            <div>
              <label style="font-size:11px;color:#6f685c;display:block">GPU Server URL</label>
              <input id="ppServerUrl" type="url" value="${_esc(cfg.url)}" class="ik-in" style="padding:6px 8px;font-size:12.5px">
            </div>
            <div>
              <label style="font-size:11px;color:#6f685c;display:block">API Key (tuỳ chọn)</label>
              <input id="ppApiKey" type="password" value="${_esc(cfg.apiKey)}" placeholder="${_esc(apiKeyPh)}" class="ik-in" style="padding:6px 8px;font-size:12.5px">
            </div>
            <button type="button" id="ppTestConnBtn" class="pp-btn">${_esc(_t('pp_test_btn', 'Kiểm tra kết nối'))}</button>
          </div>
        </details>
        <div id="ppHealthStatus" style="font-size:11.5px;margin-top:6px;padding:0 2px"></div>

        <div id="ppGuardMsg" class="pp-flag"></div>
        <div id="ppProgBox" class="pp-prog">
          <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:600">
            <span id="ppProgText">${_esc(_t('pp_uploading', 'Đang tải ảnh lên...'))}</span>
            <span id="ppProgPct">0%</span>
          </div>
          <div class="pp-pbar"><i id="ppProgBar"></i></div>
        </div>

        <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:14px">
          <button type="button" id="ppCancelBtn" class="pp-btn">${_esc(_t('pp_cancel', 'Đóng'))}</button>
          <button type="button" id="ppSubmitBtn" class="pp-btn pri" disabled>${_esc(_t('pp_submit', 'Gửi dựng 3D & đo 16 vòng'))}</button>
        </div>
      </div>
    </div>`;
  }

  function _bindModalEvents() {
    const closeBtn = document.getElementById('ppCloseBtn');
    const cancelBtn = document.getElementById('ppCancelBtn');
    if (closeBtn) closeBtn.onclick = closeModal;
    if (cancelBtn) cancelBtn.onclick = closeModal;

    const batchBtn = document.getElementById('ppBatchBtn');
    if (batchBtn) {
      batchBtn.onclick = () => {
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = 'image/*';
        inp.multiple = true;
        inp.onchange = () => {
          const assigned = assignBatchFilesToSlots(inp.files || [], state.slots, state.extraPhotos);
          state.slots = assigned.slots;
          state.extraPhotos = assigned.extraPhotos;
          _renderSlots();
          _refreshGuardBanner();
        };
        inp.click();
      };
    }

    const faceInp = document.getElementById('ppFaceInput');
    if (faceInp) {
      faceInp.onchange = () => {
        state.faceFile = faceInp.files && faceInp.files[0] ? faceInp.files[0] : null;
      };
    }

    const selectReconstructionMode = (mode) => {
      if (!RECONSTRUCTION_MODES.includes(mode)) return;
      state.reconstructionMode = mode;
      document.querySelectorAll('#plusModal .pp-rmode').forEach((card) => {
        const match = card.dataset.rmode === mode;
        card.classList.toggle('on', match);
        const r = card.querySelector('input[type="radio"]');
        if (r) r.checked = match;
      });
      _refreshGuardBanner();
    };

    document.querySelectorAll('#plusModal .pp-rmode').forEach((card) => {
      card.onclick = () => selectReconstructionMode(card.dataset.rmode);
      const r = card.querySelector('input[type="radio"]');
      if (r) {
        r.onchange = () => {
          if (r.checked) selectReconstructionMode(r.value);
        };
      }
    });

    document.querySelectorAll('#plusModal .pp-priv').forEach((btn) => {
      btn.onclick = () => {
        state.privacyLevel = Number(btn.dataset.priv);
        document.querySelectorAll('#plusModal .pp-priv').forEach((x) => x.classList.toggle('on', x === btn));
        const faceRow = document.getElementById('ppFaceRow');
        if (faceRow) faceRow.style.display = state.privacyLevel === 3 ? 'block' : 'none';
        _refreshGuardBanner();
      };
    });

    const urlIn = document.getElementById('ppServerUrl');
    const keyIn = document.getElementById('ppApiKey');
    const syncCfg = () => {
      if (urlIn) saveCfg(urlIn.value, keyIn ? keyIn.value : '');
      _refreshGuardBanner();
    };
    if (urlIn) urlIn.addEventListener('input', syncCfg);
    if (keyIn) keyIn.addEventListener('input', syncCfg);

    const hIn = document.getElementById('ppHeight');
    if (hIn) hIn.addEventListener('input', _refreshGuardBanner);
    const consentCk = document.getElementById('ppConsentChk');
    if (consentCk) consentCk.addEventListener('change', _refreshGuardBanner);

    const testBtn = document.getElementById('ppTestConnBtn');
    if (testBtn) {
      testBtn.onclick = () => onHealthCheck(urlIn ? urlIn.value : undefined, keyIn ? keyIn.value : '');
    }

    const goBtn = document.getElementById('ppSubmitBtn');
    if (goBtn) goBtn.onclick = _submitScan;

    _renderSlots();
    _refreshGuardBanner();
  }

  function _setProgress(pct, label) {
    const box = document.getElementById('ppProgBox');
    const bar = document.getElementById('ppProgBar');
    const pctEl = document.getElementById('ppProgPct');
    const txtEl = document.getElementById('ppProgText');
    if (box) box.style.display = 'block';
    if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
    if (pctEl) pctEl.textContent = Math.round(pct) + '%';
    if (txtEl && label) txtEl.textContent = label;
  }

  function buildScanFormData(opts) {
    const o = opts || {};
    const fd = new FormData();
    const photos = o.photos || [];
    const roles = Array.isArray(o.slotRoles) && o.slotRoles.length === photos.length
      ? o.slotRoles.map((r) => String(r || 'extra'))
      : photos.map((f, idx) => {
          const inferred = inferCardinalRoleFromFilename(f && f.name);
          if (inferred !== null && ANGLE_SLOTS[inferred]) return ANGLE_SLOTS[inferred].key;
          return idx < ANGLE_SLOTS.length ? ANGLE_SLOTS[idx].key : 'extra';
        });
    photos.forEach((f, idx) => fd.append('photos', f, (f && f.name) || `angle_${idx}.jpg`));
    fd.append('slot_roles', JSON.stringify(roles));
    if (Number(o.privacyLevel) === 3 && o.faceFile) {
      fd.append('closeup_face', o.faceFile, o.faceFile.name || 'closeup_face.jpg');
    }
    fd.append('height_cm', String(o.heightCm));
    fd.append('sex', o.sex === 'male' ? 'male' : 'female');
    fd.append('weight_kg', String(o.weightKg || 0));
    fd.append('person_id', o.personId || 'guest_01');
    fd.append('privacy_level', String(o.privacyLevel));
    fd.append('reconstruction_mode', String(o.reconstructionMode || DEFAULT_RECONSTRUCTION_MODE));
    fd.append('consent_granted', 'true');
    return fd;
  }

  async function _submitScan() {
    const client = state.bridge ? state.bridge.getClient() : null;
    if (!client) return;
    const urlIn = document.getElementById('ppServerUrl');
    const keyIn = document.getElementById('ppApiKey');
    const cfg = saveCfg(urlIn ? urlIn.value : loadCfg().url, keyIn ? keyIn.value : loadCfg().apiKey);
    const hVal = Number((document.getElementById('ppHeight') || {}).value || client.height || 168);
    const wVal = Number((document.getElementById('ppWeight') || {}).value || client.weight || 0);
    const consentCk = document.getElementById('ppConsentChk');
    const { photos, slotRoles } = _collectedPhotosWithRoles();

    const check = validatePrivacyAndConnection({
      consentMode: state.bridge ? state.bridge.getConsentMode() : 'local',
      serverUrl: cfg.url,
      privacyLevel: state.privacyLevel,
      reconstructionMode: state.reconstructionMode,
      cardinalSlotsCount: _cardinalSlotsCount(),
      consentChecked: Boolean(consentCk && consentCk.checked),
      photoCount: photos.length,
      heightCm: hVal,
    });
    if (!check.ok) {
      _refreshGuardBanner();
      return;
    }

    client.height = hVal;
    if (wVal > 0) client.weight = wVal;
    const goBtn = document.getElementById('ppSubmitBtn');
    if (goBtn) goBtn.disabled = true;
    _setProgress(5, _t('pp_uploading', 'Đang tải 4 ảnh lên server...'));

    try {
      // Check health first cleanly (ignore failures gracefully so actual POST error shows if unreachable)
      try {
        const hr = await _performFetch(fetch, cfg.url + '/health', { method: 'GET', includeAuth: false });
        const hj = await hr.json().catch(() => null);
        const hInfo = classifyHealthPayload(hj, hr.status);
        if (hInfo.state !== 'error' && hj) {
          state.health = hj;
          state.healthStatus = hInfo.state;
          state.serverMode = hInfo.mode || state.serverMode;
          state.testOnly = hInfo.testOnly;
          if (hInfo.isLoading) {
            _setProgress(5, _t('pp_health_loading', 'Server đã kết nối — đang nạp mô hình AI 3D vào GPU, vui lòng đợi...'));
          }
        }
      } catch (_) {}

      const fd = buildScanFormData({
        photos: photos,
        slotRoles: slotRoles,
        privacyLevel: state.privacyLevel,
        faceFile: state.faceFile,
        heightCm: hVal,
        sex: client.sex === 'male' ? 'male' : 'female',
        weightKg: wVal || 0,
        personId: client.id || client.name || 'guest_01',
        reconstructionMode: state.reconstructionMode || DEFAULT_RECONSTRUCTION_MODE,
      });

      const postResp = await _performFetch(fetch, cfg.url + '/api/v1/bodyscan', {
        method: 'POST',
        includeAuth: true,
        body: fd,
      });
      if (!postResp.ok) {
        const errBody = await postResp.json().catch(() => ({}));
        throw new Error(errBody.detail || 'HTTP ' + postResp.status);
      }
      const queued = await postResp.json();
      state.jobId = queued.job_id;
      state.lastJobId = queued.job_id;
      state.lastBaseUrl = cfg.url;
      await _pollJobUntilDone(cfg.url, queued.job_id);
    } catch (err) {
      const warnEl = document.getElementById('ppGuardMsg');
      if (warnEl) {
        warnEl.className = 'pp-flag warn';
        warnEl.textContent = _t('pp_submit_err', 'Lỗi xử lý: ') + (err && err.message ? err.message : err);
      }
      if (goBtn) goBtn.disabled = false;
    }
  }

  async function _pollJobUntilDone(baseUrl, jobId, opts) {
    const o = opts || {};
    const fetchFn = o.fetchFn || rootScope.fetch;
    const pollIntervalMs = Number.isFinite(Number(o.pollIntervalMs)) ? Math.max(10, Number(o.pollIntervalMs)) : 850;
    const maxRetries = Number.isFinite(Number(o.maxRetries)) ? Math.max(0, Number(o.maxRetries)) : 6;
    const retryBaseMs = Number.isFinite(Number(o.retryDelayMs)) ? Math.max(10, Number(o.retryDelayMs)) : 1000;
    const stepLabels = {
      queued: _t('pp_step_queued', 'Đang xếp hàng trên GPU...'),
      running: _t('pp_step_running', 'Đang khởi chạy pipeline...'),
      inference: _t('pp_step_inf', 'Đang ước lượng hình thể 3D...'),
      segmentation: _t('pp_step_seg', 'Đang tách nền & vùng đầu/tóc...'),
      fitting: _t('pp_step_fit', 'Đang khớp dáng MHR theo 4 góc & cân nặng...'),
      fusion: _t('pp_step_fus', 'Đang gộp 4 góc nhìn A-pose...'),
      head_hair: _t('pp_step_head', 'Đang dựng chi tiết mặt, mũi & tóc 3D...'),
      outer_shell: _t('pp_step_outer', 'Đang dựng lớp vỏ ngoài sát thực tế...'),
      measurements: _t('pp_step_meas', 'Đang cắt đo 16 vòng cơ thể...'),
      texturing: _t('pp_step_tex', 'Đang dán ảnh thật & bảo mật mặt...'),
      export: _t('pp_step_exp', 'Đang xuất file 3D GLB...'),
      completed: _t('pp_step_done', 'Đang tải mô hình 3D về trình duyệt...'),
    };
    const deadline = Date.now() + (Number.isFinite(Number(o.timeoutMs)) ? Number(o.timeoutMs) : 10 * 60 * 1000);
    let consecutiveErrors = 0;
    while (Date.now() < deadline) {
      let r = null;
      try {
        r = await _performFetch(fetchFn, baseUrl + '/api/v1/bodyscan/' + encodeURIComponent(jobId), {
          method: 'GET',
          includeAuth: true,
        });
      } catch (netErr) {
        consecutiveErrors += 1;
        if (consecutiveErrors > maxRetries) {
          throw netErr;
        }
        const waitMs = Math.min(5000, Math.round(retryBaseMs * Math.pow(1.5, consecutiveErrors - 1)));
        await new Promise((res) => setTimeout(res, waitMs));
        continue;
      }
      if (!r.ok) {
        const st = Number(r.status);
        if (st === 401 || st === 403 || st === 404 || st === 410) {
          throw new Error('Poll HTTP ' + st);
        }
        consecutiveErrors += 1;
        if (consecutiveErrors > maxRetries) {
          throw new Error('Poll HTTP ' + st);
        }
        const waitMs = Math.min(5000, Math.round(retryBaseMs * Math.pow(1.5, consecutiveErrors - 1)));
        await new Promise((res) => setTimeout(res, waitMs));
        continue;
      }
      consecutiveErrors = 0;
      const data = await r.json();
      _setProgress(data.progress || 15, stepLabels[data.step] || data.step || '');

      if (data.status === 'failed') {
        throw new Error(data.error || 'Scan failed on server');
      }
      if (data.status === 'completed') {
        await _applyCompletedPayload(baseUrl, data, o);
        return data;
      }
      await new Promise((res) => setTimeout(res, pollIntervalMs));
    }
    throw new Error('Timeout waiting for scan result');
  }

  function _collectUniqueWarnings() {
    const seen = new Set();
    const out = [];
    for (let i = 0; i < arguments.length; i++) {
      const list = arguments[i];
      if (!Array.isArray(list)) continue;
      for (const item of list) {
        const s = String(item == null ? '' : item).trim();
        if (s && !seen.has(s)) {
          seen.add(s);
          out.push(s);
        }
      }
    }
    return out;
  }

  function extractAuditSummary(source) {
    const src = source || state;
    const bs = src.bodyScan || src.body_scan || null;
    const artifacts = src.artifacts || {};
    const audit = src.proportionAudit || src.proportion_audit || (bs && bs.proportion_audit) || null;
    const rawMode = src.serverMode || src.mode || (state.health && state.health.mode) || (src.testOnly === false || src.test_only === false ? 'gpu' : 'mock');
    const engineMode = String(rawMode).toLowerCase() === 'gpu' ? 'gpu' : 'mock';
    const testOnly = src.testOnly !== undefined ? Boolean(src.testOnly) : (src.test_only !== undefined ? Boolean(src.test_only) : engineMode !== 'gpu');
    const reconstructionMode = src.activeReconstructionMode || src.reconstruction_mode || (bs && bs.reconstruction_mode) || null;
    const effectiveReconstructionMode = src.effectiveReconstructionMode || src.effective_reconstruction_mode || (bs && bs.effective_reconstruction_mode) || reconstructionMode;
    const wasDowngraded = Boolean(reconstructionMode && effectiveReconstructionMode && reconstructionMode !== effectiveReconstructionMode);
    const warnings = _collectUniqueWarnings(src.warnings, bs && bs.warnings, audit && audit.warnings);
    const coreMeshUrl = artifacts.core_mesh_url || null;
    const meshUrl = artifacts.mesh_url || null;
    const jsonUrl = artifacts.json_url || null;
    const summaryUrl = artifacts.summary_url || null;
    const backendsUsed = src.backendsUsed || src.backends_used || (bs && bs.backends_used) || null;
    return {
      engineMode,
      testOnly,
      reconstructionMode,
      effectiveReconstructionMode,
      wasDowngraded,
      proportionAudit: audit,
      backendsUsed,
      warnings,
      hasCoreMesh: Boolean(coreMeshUrl),
      meshViewLayer: src.meshViewLayer === 'core' ? 'core' : 'outer',
      coreMeshUrl,
      meshUrl,
      jsonUrl,
      summaryUrl,
    };
  }

  function getActiveGlbDownloadTarget(source) {
    const src = source || state;
    const artifacts = (src && src.artifacts) || {};
    const layer = src && src.meshViewLayer === 'core' ? 'core' : 'outer';
    const coreUrl = artifacts.core_mesh_url || src.coreMeshUrl || null;
    const outerUrl = artifacts.mesh_url || src.meshUrl || null;
    if (layer === 'core' && coreUrl) {
      return {
        layer: 'core',
        artifactKey: 'core_mesh_url',
        url: coreUrl,
        filename: 'alpha_scan_core.glb',
      };
    }
    return {
      layer: 'outer',
      artifactKey: 'mesh_url',
      url: outerUrl || null,
      filename: 'alpha_scan.glb',
    };
  }

  const DETAIL_EN_FALLBACKS = {
    overview_title: 'Body Overview',
    sex_m: 'Male',
    sex_f: 'Female',
    pp_del_done: 'Server 3D mesh & images deleted (16 girths retained in left panel & PDF report).',
    pp_holo_back: 'Center stage returned to reference hologram.',
    pp_show_holo: 'Show Reference Hologram',
    pp_show_3d: 'Show Real 3D Model',
    pp_detail_title: 'Alpha Scan Plus/PRO 3D Model',
    pp_engine_gpu: 'Real GPU',
    pp_engine_mock: 'MOCK TEST',
    pp_privacy_used: 'Privacy level:',
    pp_lvl_short_1: 'Level 1 (Mesh only)',
    pp_lvl_short_2: 'Level 2 (Blurred face)',
    pp_lvl_short_3: 'Level 3 (Clear face)',
    pp_mode_used: '3D Mode:',
    pp_mode_downgraded: 'Auto-downgraded',
    pp_rmode_short_dual_layer: 'Dual-layer (dual_layer)',
    pp_rmode_short_hi_res_head: 'Hi-res head (hi_res_head)',
    pp_rmode_short_calibrated_mhr: 'Calibrated MHR (calibrated_mhr)',
    pp_layer_viewing: 'Viewing layer:',
    pp_layer_outer: 'Real outer surface',
    pp_layer_core: 'MHR measurement core',
    pp_backends_title: 'Active 3D Pipeline:',
    pp_srv_auto_badge: 'Auto-connected — no manual URL needed',
    pp_apikey_ph: 'Leave blank (only fill if you set a custom password on Colab)',
    pp_switch_server_consent_btn: '⚡ Enable Server GPU mode (allow sending photos to Colab)',
    pp_audit_title: 'Proportion & Silhouette Audit',
    pp_audit_status_matched: 'Matched',
    pp_audit_status_estimated_only: 'Estimated only',
    pp_audit_status_mismatch_warning: 'Mismatch warning',
    pp_audit_est_w: 'Estimated weight',
    pp_audit_decl_w: 'Declared weight',
    pp_audit_iou: 'MHR silhouette fit',
    pp_audit_outer_iou: 'Outer surface fit',
    pp_warnings_title: 'Notices & Warnings:',
    pp_raw_shredded: 'raw photos shredded on server right after 3D generation',
    pp_ctrl_tip: 'Auto-rotating 360° — drag with mouse to orbit, scroll to zoom, or use Front / Back / Rotate.',
    pp_show_core_mesh: 'View MHR Core Mesh',
    pp_show_outer_mesh: 'View Real Outer Surface',
    pp_dl_glb: 'Download .glb',
    pp_dl_json: 'Download JSON',
    pp_del_srv_btn: 'Delete 3D/Photos on Server',
    pp_dup_warn: 'Duplicate photo warning (same name & size): {files}. Uploading duplicate angles may cause the server to auto-downgrade to calibrated_mhr.',
    pp_health_loading: 'Server connected — loading 3D AI models onto GPU, please wait...',
    pp_err_410: 'Server 3D assets were deleted (410 Gone); 16 circumferences are retained.',
    pp_dl_err: 'Download failed: ',
  };

  function _extractJobIdFromArtifacts(artifacts) {
    if (!artifacts || typeof artifacts !== 'object') return null;
    for (const k of Object.keys(artifacts)) {
      const u = String(artifacts[k] || '');
      const m = u.match(/\/api\/v1\/bodyscan\/([^/?#]+)\/files\//);
      if (m && m[1]) {
        try { return decodeURIComponent(m[1]); } catch (_) { return m[1]; }
      }
    }
    return null;
  }

  function _applyArtifactsMap(baseUrl, rawArtifacts) {
    const cleanBase = String(baseUrl || state.lastBaseUrl || loadCfg().url || '').replace(/\/+$/, '');
    const next = {};
    Object.keys(rawArtifacts || {}).forEach((k) => {
      if (rawArtifacts[k]) {
        next[k] = cleanBase ? resolveServerUrl(cleanBase, rawArtifacts[k]) : rawArtifacts[k];
      }
    });
    state.artifacts = next;
    return next;
  }

  function _setDelStatus(msg) {
    if (typeof document === 'undefined') return;
    const st = document.getElementById('ppDelStatus');
    if (st) st.textContent = msg;
  }

  function _syncDownloadLinks() {
    if (typeof document === 'undefined') return;
    const glbEl = document.getElementById('ppDlGlb');
    if (glbEl) {
      const target = getActiveGlbDownloadTarget(state);
      glbEl.setAttribute('href', target.url || '#');
      glbEl.setAttribute('download', target.filename);
    }
    const jsonEl = document.getElementById('ppDlJson');
    if (jsonEl) {
      const jsonUrl = (state.artifacts && state.artifacts.json_url) || '#';
      jsonEl.setAttribute('href', jsonUrl);
      jsonEl.setAttribute('download', 'body_scan.json');
    }
  }

  function _markAssetsDeletedFromServer(retainedArtifacts, baseUrl) {
    if (state.meshGroup) {
      if (state.bridge && typeof state.bridge.detachRealGroup === 'function') {
        state.bridge.detachRealGroup(state.meshGroup);
      }
      disposeObject3D(state.meshGroup);
      state.meshGroup = null;
    }
    state.outerGlbBuffer = null;
    state.coreGlbBuffer = null;
    state.coreMeshLoading = false;
    state.meshViewLayer = 'outer';
    state.assetsDeleted = true;
    state.hasLoadedMesh = false;
    state.realActive = false;
    if (retainedArtifacts && typeof retainedArtifacts === 'object') {
      _applyArtifactsMap(baseUrl, retainedArtifacts);
    } else if (state.artifacts) {
      const kept = {};
      if (state.artifacts.json_url) kept.json_url = state.artifacts.json_url;
      if (state.artifacts.summary_url) kept.summary_url = state.artifacts.summary_url;
      state.artifacts = kept;
    }
    renderLeft16Box();
    if (state.bridge && typeof state.bridge.refreshAppearance === 'function') {
      state.bridge.refreshAppearance();
    }
    if (typeof document !== 'undefined') {
      const detailEl = document.getElementById('detail');
      if (detailEl) renderDetailCard(detailEl);
    }
  }

  async function refreshArtifactUrlsIfNeeded(force, opts) {
    let forceRefresh = Boolean(force);
    let o = opts || {};
    if (typeof force === 'object' && force !== null) {
      o = force;
      forceRefresh = Boolean(o.force);
    }
    if (state.assetsDeleted === true) {
      return {
        refreshed: false,
        assetsDeleted: true,
        artifacts: state.artifacts,
        meshUrl: null,
        coreMeshUrl: null,
        jsonUrl: (state.artifacts && state.artifacts.json_url) || null,
        summaryUrl: (state.artifacts && state.artifacts.summary_url) || null,
      };
    }
    const jobId = o.jobId || state.lastJobId || state.jobId || _extractJobIdFromArtifacts(state.artifacts);
    if (!jobId) {
      return {
        refreshed: false,
        reason: 'no_job',
        artifacts: state.artifacts,
        meshUrl: (state.artifacts && state.artifacts.mesh_url) || null,
        coreMeshUrl: (state.artifacts && state.artifacts.core_mesh_url) || null,
        jsonUrl: (state.artifacts && state.artifacts.json_url) || null,
        summaryUrl: (state.artifacts && state.artifacts.summary_url) || null,
      };
    }
    const nowMs = Number.isFinite(Number(o.now)) ? Number(o.now) : Date.now();
    const fetchedAt = Number(state.artifactUrlsFetchedAt || 0);
    if (!forceRefresh && fetchedAt > 0 && (nowMs - fetchedAt) < ARTIFACT_URL_REFRESH_MS) {
      return {
        refreshed: false,
        reason: 'fresh',
        artifacts: state.artifacts,
        meshUrl: (state.artifacts && state.artifacts.mesh_url) || null,
        coreMeshUrl: (state.artifacts && state.artifacts.core_mesh_url) || null,
        jsonUrl: (state.artifacts && state.artifacts.json_url) || null,
        summaryUrl: (state.artifacts && state.artifacts.summary_url) || null,
      };
    }

    const fetchFn = o.fetchFn || rootScope.fetch;
    if (typeof fetchFn !== 'function') throw new Error('fetch is not available');
    const baseUrl = String(o.baseUrl || state.lastBaseUrl || loadCfg().url || '').replace(/\/+$/, '');
    if (!baseUrl) {
      return { refreshed: false, reason: 'no_base_url', artifacts: state.artifacts };
    }
    const statusUrl = baseUrl + '/api/v1/bodyscan/' + encodeURIComponent(jobId);
    const resp = await _performFetch(fetchFn, statusUrl, { method: 'GET', includeAuth: true });
    if (resp.status === 410) {
      _markAssetsDeletedFromServer(null, baseUrl);
      _setDelStatus(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
      return { refreshed: false, assetsDeleted: true, status: 410, artifacts: state.artifacts };
    }
    if (!resp.ok) {
      throw new Error('Refresh HTTP ' + resp.status);
    }
    const data = await resp.json();
    if (data && (data.assets_deleted === true || data.status === 'deleted')) {
      _markAssetsDeletedFromServer(data.artifacts || null, baseUrl);
      _setDelStatus(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
      return { refreshed: false, assetsDeleted: true, status: 410, artifacts: state.artifacts };
    }

    state.jobId = jobId;
    state.lastJobId = jobId;
    state.lastBaseUrl = baseUrl;
    _applyArtifactsMap(baseUrl, (data && data.artifacts) || {});
    state.artifactUrlsFetchedAt = nowMs;
    _syncDownloadLinks();
    return {
      refreshed: true,
      artifacts: state.artifacts,
      meshUrl: (state.artifacts && state.artifacts.mesh_url) || null,
      coreMeshUrl: (state.artifacts && state.artifacts.core_mesh_url) || null,
      jsonUrl: (state.artifacts && state.artifacts.json_url) || null,
      summaryUrl: (state.artifacts && state.artifacts.summary_url) || null,
    };
  }

  async function _fetchArtifactWithRetry(artifactKey, initialUrl, opts) {
    const o = opts || {};
    const fetchFn = o.fetchFn || rootScope.fetch;
    if (typeof fetchFn !== 'function') throw new Error('fetch is not available');

    const isBinary = artifactKey !== 'json_url' && artifactKey !== 'summary_url';
    if (state.assetsDeleted === true && isBinary) {
      const delErr = new Error(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
      delErr.status = 410;
      delErr.assetsDeleted = true;
      throw delErr;
    }

    let url = initialUrl;
    let resp = await _performFetch(fetchFn, url, { method: 'GET', includeAuth: false });
    let refreshedResult = null;

    if (resp.status === 410) {
      _markAssetsDeletedFromServer(null, o.baseUrl || state.lastBaseUrl);
      _setDelStatus(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
      const err410 = new Error('HTTP 410 Gone');
      err410.status = 410;
      err410.assetsDeleted = true;
      throw err410;
    }

    if (resp.status === 403 && state.assetsDeleted !== true) {
      refreshedResult = await refreshArtifactUrlsIfNeeded(true, o);
      if (state.assetsDeleted === true || (refreshedResult && refreshedResult.assetsDeleted)) {
        _setDelStatus(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
        const err410 = new Error('HTTP 410 Gone');
        err410.status = 410;
        err410.assetsDeleted = true;
        throw err410;
      }
      const nextUrl = (state.artifacts && state.artifacts[artifactKey]) || url;
      resp = await _performFetch(fetchFn, nextUrl, { method: 'GET', includeAuth: false });
      if (resp.status === 410) {
        _markAssetsDeletedFromServer(null, o.baseUrl || state.lastBaseUrl);
        _setDelStatus(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
        const err410 = new Error('HTTP 410 Gone');
        err410.status = 410;
        err410.assetsDeleted = true;
        throw err410;
      }
    }

    if (!resp.ok) {
      const httpErr = new Error('Failed to load GLB: HTTP ' + resp.status);
      httpErr.status = resp.status;
      throw httpErr;
    }
    return { resp, refreshed: Boolean(refreshedResult && refreshedResult.refreshed) };
  }

  function _triggerBrowserDownload(blob, filename) {
    if (typeof document === 'undefined' || typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') {
      return false;
    }
    const objUrl = URL.createObjectURL(blob);
    try {
      const a = document.createElement('a');
      if (!a) return false;
      a.href = objUrl;
      a.download = filename;
      if (a.style) a.style.display = 'none';
      if (document.body && typeof document.body.appendChild === 'function') {
        document.body.appendChild(a);
      }
      if (typeof a.click === 'function') a.click();
      if (document.body && typeof document.body.removeChild === 'function' && a.parentNode === document.body) {
        document.body.removeChild(a);
      }
      return true;
    } finally {
      if (typeof setTimeout === 'function') {
        setTimeout(() => {
          try { URL.revokeObjectURL(objUrl); } catch (_) {}
        }, 1000);
      } else {
        try { URL.revokeObjectURL(objUrl); } catch (_) {}
      }
    }
  }

  async function downloadArtifact(kind, opts) {
    const o = opts || {};
    const isJson = kind === 'json';
    if (!isJson && state.assetsDeleted === true) {
      const err = new Error(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
      err.status = 410;
      err.assetsDeleted = true;
      _setDelStatus(err.message);
      throw err;
    }

    await refreshArtifactUrlsIfNeeded(false, o);

    if (!isJson && state.assetsDeleted === true) {
      const err = new Error(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
      err.status = 410;
      err.assetsDeleted = true;
      _setDelStatus(err.message);
      throw err;
    }

    const target = isJson
      ? {
          artifactKey: 'json_url',
          url: (state.artifacts && state.artifacts.json_url) || null,
          filename: 'body_scan.json',
          mimeType: 'application/json',
        }
      : Object.assign(getActiveGlbDownloadTarget(state), { mimeType: 'model/gltf-binary' });

    if (!target.url) {
      throw new Error('Missing artifact URL for ' + target.artifactKey);
    }
    _syncDownloadLinks();

    const { resp } = await _fetchArtifactWithRetry(target.artifactKey, target.url, o);
    const activeUrl = (state.artifacts && state.artifacts[target.artifactKey]) || target.url;
    _syncDownloadLinks();

    let blob = null;
    if (typeof resp.blob === 'function') {
      blob = await resp.blob();
    } else if (typeof resp.arrayBuffer === 'function') {
      const buf = await resp.arrayBuffer();
      blob = new Blob([buf], { type: target.mimeType });
    } else if (typeof resp.text === 'function') {
      const txt = await resp.text();
      blob = new Blob([txt], { type: target.mimeType });
    }

    if (typeof o.triggerDownload === 'function') {
      o.triggerDownload({ blob, filename: target.filename, url: activeUrl });
    } else if (blob) {
      _triggerBrowserDownload(blob, target.filename);
    }
    return {
      ok: true,
      filename: target.filename,
      url: activeUrl,
      artifactKey: target.artifactKey,
      blob,
    };
  }

  async function _applyCompletedPayload(baseUrl, data, opts) {
    const o = opts || {};
    const fetchFn = o.fetchFn || rootScope.fetch;
    if (typeof fetchFn !== 'function') throw new Error('fetch is not available');
    const cleanBase = String(baseUrl || '').replace(/\/+$/, '');
    const meshUrl = resolveServerUrl(cleanBase, data.artifacts && data.artifacts.mesh_url);
    if (!meshUrl) throw new Error('Missing mesh_url in completed payload');

    const inferredJobId = (data && data.job_id) || state.jobId || state.lastJobId || _extractJobIdFromArtifacts(data.artifacts);
    const nowMs = Number.isFinite(Number(o.now)) ? Number(o.now) : Date.now();

    const { resp: glbResp, refreshed } = await _fetchArtifactWithRetry('mesh_url', meshUrl, {
      ...o,
      baseUrl: cleanBase,
      jobId: inferredJobId,
    });
    const arrayBuf = await glbResp.arrayBuffer();
    const THREE = o.THREE || rootScope.THREE;
    if (!THREE) throw new Error('THREE r128 is not available');

    const parsed = await parseGlbBinary(arrayBuf, THREE, o.imageFactory);

    // Replace previous real mesh safely and dispose GPU memory
    if (state.meshGroup) {
      if (state.bridge && typeof state.bridge.detachRealGroup === 'function') {
        state.bridge.detachRealGroup(state.meshGroup);
      }
      disposeObject3D(state.meshGroup);
    }

    state.jobStatus = 'completed';
    if (inferredJobId) {
      state.jobId = inferredJobId;
      state.lastJobId = inferredJobId;
    }
    if (cleanBase) state.lastBaseUrl = cleanBase;
    state.bodyScan = data.body_scan;
    const bs = data.body_scan || null;
    if (data.mode) {
      state.serverMode = data.mode;
    } else if (!state.serverMode && state.health && state.health.mode) {
      state.serverMode = state.health.mode;
    }
    if (data.test_only !== undefined) {
      state.testOnly = Boolean(data.test_only);
    }
    if (data.privacy_level !== undefined) {
      state.privacyLevel = Number(data.privacy_level);
    }
    state.activeReconstructionMode = data.reconstruction_mode || (bs && bs.reconstruction_mode) || state.reconstructionMode || DEFAULT_RECONSTRUCTION_MODE;
    state.effectiveReconstructionMode = data.effective_reconstruction_mode || (bs && bs.effective_reconstruction_mode) || state.activeReconstructionMode;
    state.proportionAudit = data.proportion_audit || (bs && bs.proportion_audit) || null;
    state.backendsUsed = data.backends_used || (bs && bs.backends_used) || null;
    state.warnings = _collectUniqueWarnings(data.warnings, bs && bs.warnings, state.proportionAudit && state.proportionAudit.warnings);
    if (!refreshed) {
      _applyArtifactsMap(cleanBase, data.artifacts || {});
      state.artifactUrlsFetchedAt = nowMs;
    }
    state.deletionAudit = data.deletion_audit || null;
    state.assetsDeleted = false;
    state.realActive = true;
    state.hasLoadedMesh = true;
    state.meshViewLayer = 'outer';
    state.outerGlbBuffer = arrayBuf;
    state.coreGlbBuffer = null;
    state.coreMeshLoading = false;
    state.meshGroup = parsed.group;
    state.meshMetrics = parsed.metrics;
    state.duplicateWarning = null;

    // Raw uploaded files in slots no longer needed in browser memory after upload completes
    state.slots = [null, null, null, null];
    state.extraPhotos = [];
    state.faceFile = null;

    if (state.bridge) {
      const client = state.bridge.getClient();
      const isInBody = Boolean(state.bridge.isInBodyClient && state.bridge.isInBodyClient());
      applyScanResultToClient(client, data.body_scan, isInBody);
      if (typeof state.bridge.onScanCompleted === 'function') {
        state.bridge.onScanCompleted(parsed.group, parsed.metrics);
      }
    }

    closeModal();
  }

  async function toggleCoreMeshView(opts) {
    const o = opts || {};
    if (!hasMesh() || !state.artifacts || !state.artifacts.core_mesh_url || state.coreMeshLoading) {
      return null;
    }
    const targetLayer = (o.targetLayer === 'core' || o.targetLayer === 'outer')
      ? o.targetLayer
      : (state.meshViewLayer === 'outer' ? 'core' : 'outer');
    if (targetLayer === state.meshViewLayer) return state.meshViewLayer;

    const THREE = o.THREE || rootScope.THREE;
    if (!THREE) throw new Error('THREE r128 is not available');

    const artifactKey = targetLayer === 'core' ? 'core_mesh_url' : 'mesh_url';
    if (!state.artifacts[artifactKey]) throw new Error('Missing mesh URL for layer: ' + targetLayer);

    state.coreMeshLoading = true;
    const coreBtn = typeof document !== 'undefined' ? document.getElementById('ppToggleCoreBtn') : null;
    if (coreBtn) coreBtn.disabled = true;

    try {
      let arrayBuf = targetLayer === 'core' ? state.coreGlbBuffer : state.outerGlbBuffer;
      if (!arrayBuf) {
        await refreshArtifactUrlsIfNeeded(false, o);
        if (state.assetsDeleted === true) {
          const delErr = new Error(_t('pp_err_410', 'Tệp 3D trên server đã bị xoá (410 Gone); 16 vòng đo vẫn được giữ.'));
          delErr.status = 410;
          delErr.assetsDeleted = true;
          throw delErr;
        }
        const targetUrl = state.artifacts && state.artifacts[artifactKey];
        if (!targetUrl) throw new Error('Missing mesh URL for layer: ' + targetLayer);
        const { resp } = await _fetchArtifactWithRetry(artifactKey, targetUrl, o);
        arrayBuf = await resp.arrayBuffer();
        if (targetLayer === 'core') state.coreGlbBuffer = arrayBuf;
        else state.outerGlbBuffer = arrayBuf;
      }

      const parsed = await parseGlbBinary(arrayBuf, THREE, o.imageFactory);

      if (state.meshGroup) {
        if (state.bridge && typeof state.bridge.detachRealGroup === 'function') {
          state.bridge.detachRealGroup(state.meshGroup);
        }
        disposeObject3D(state.meshGroup);
        state.meshGroup = null;
      }

      state.meshGroup = parsed.group;
      state.meshMetrics = parsed.metrics;
      state.meshViewLayer = targetLayer;
      state.realActive = true;
      _syncDownloadLinks();

      if (state.bridge) {
        if (typeof state.bridge.attachRealGroup === 'function') {
          state.bridge.attachRealGroup(parsed.group, parsed.metrics, { preserveCamera: true });
        } else if (typeof state.bridge.onScanCompleted === 'function') {
          state.bridge.onScanCompleted(parsed.group, parsed.metrics);
        }
      }
      if (typeof document !== 'undefined') {
        const detailEl = document.getElementById('detail');
        if (detailEl) renderDetailCard(detailEl);
      }
      return targetLayer;
    } catch (err) {
      if (!(err && err.status === 410)) {
        _setDelStatus(_t('pp_core_err', 'Không chuyển được lớp lõi/vỏ: ') + (err && err.message ? err.message : err));
      }
      throw err;
    } finally {
      state.coreMeshLoading = false;
      const btnAfter = typeof document !== 'undefined' ? document.getElementById('ppToggleCoreBtn') : null;
      if (btnAfter) btnAfter.disabled = false;
    }
  }

  function openModal() {
    if (typeof document === 'undefined') return;
    if (state.bridge && typeof state.bridge.closeBodyEditModal === 'function') {
      state.bridge.closeBodyEditModal();
    }
    const el = _ensureModal();
    if (!el) return;
    el.innerHTML = _buildModalHtml();
    el.style.display = 'flex';
    _bindModalEvents();
    if (
      state._lastHealthOk === null &&
      typeof rootScope.fetch === 'function' &&
      rootScope.location &&
      rootScope.location.protocol
    ) {
      onHealthCheck().catch(() => {});
    }
  }

  function closeModal() {
    const el = document.getElementById('plusModal');
    if (el) {
      _clearSlotsObjectUrls();
      el.style.display = 'none';
      el.innerHTML = '';
    }
  }

  function resetSession() {
    if (state.pollTimer) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
    closeModal();
    if (state.meshGroup) {
      if (state.bridge && typeof state.bridge.detachRealGroup === 'function') {
        state.bridge.detachRealGroup(state.meshGroup);
      }
      disposeObject3D(state.meshGroup);
      state.meshGroup = null;
    }
    state.serverCfg = null;
    state._explicitlySavedCfg = false;
    state._lastHealthOk = null;
    state.serverMode = null;
    state.healthStatus = null;
    state.jobId = null;
    state.lastJobId = null;
    state.lastBaseUrl = null;
    state.artifactUrlsFetchedAt = 0;
    state.jobStatus = null;
    state.bodyScan = null;
    state.artifacts = null;
    state.deletionAudit = null;
    state.privacyLevel = null;
    state.reconstructionMode = DEFAULT_RECONSTRUCTION_MODE;
    state.activeReconstructionMode = null;
    state.effectiveReconstructionMode = null;
    state.proportionAudit = null;
    state.backendsUsed = null;
    state.warnings = [];
    state.duplicateWarning = null;
    state.meshViewLayer = 'outer';
    state.outerGlbBuffer = null;
    state.coreGlbBuffer = null;
    state.coreMeshLoading = false;
    state.assetsDeleted = false;
    state.realActive = false;
    state.hasLoadedMesh = false;
    state.meshMetrics = null;
    state.slots = [null, null, null, null];
    state.extraPhotos = [];
    state.faceFile = null;
    renderLeft16Box();
  }

  function hasMesh() {
    return Boolean(state.hasLoadedMesh && state.meshGroup && !state.assetsDeleted);
  }

  function isReal3DActive() {
    return Boolean(state.realActive && hasMesh());
  }

  function setRealActive(flag) {
    state.realActive = Boolean(flag) && hasMesh();
    if (state.bridge && typeof state.bridge.refreshAppearance === 'function') {
      state.bridge.refreshAppearance();
    }
  }

  function getResult() {
    return state.bodyScan;
  }

  function getDeletedFlag() {
    return Boolean(state.assetsDeleted);
  }

  function getStateSnapshot() {
    return {
      bodyScan: state.bodyScan,
      serverMode: state.serverMode,
      healthStatus: state.healthStatus,
      lastHealthOk: state._lastHealthOk,
      testOnly: state.testOnly,
      assetsDeleted: state.assetsDeleted,
      privacyLevel: state.privacyLevel,
      reconstructionMode: state.reconstructionMode,
      activeReconstructionMode: state.activeReconstructionMode,
      effectiveReconstructionMode: state.effectiveReconstructionMode,
      proportionAudit: state.proportionAudit,
      backendsUsed: state.backendsUsed,
      warnings: state.warnings.slice(),
      duplicateWarning: state.duplicateWarning,
      meshViewLayer: state.meshViewLayer,
      hasCoreMesh: Boolean(state.artifacts && state.artifacts.core_mesh_url),
      artifacts: state.artifacts ? Object.assign({}, state.artifacts) : null,
      artifactUrlsFetchedAt: state.artifactUrlsFetchedAt,
      metrics: state.meshMetrics,
      jobId: state.jobId,
      lastJobId: state.lastJobId,
    };
  }

  function renderLeft16Box() {
    if (typeof document === 'undefined') return;
    const box = document.getElementById('plus16Box');
    if (!box) return;
    if (!state.bodyScan) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    box.style.display = 'block';
    box.innerHTML = renderLeft16Html(
      { bodyScan: state.bodyScan, testOnly: state.testOnly, assetsDeleted: state.assetsDeleted },
      _lang(),
      _t
    );
  }

  async function deleteServerAssets(opts) {
    const o = opts || {};
    const jobId = state.jobId || state.lastJobId;
    if (!jobId || state.assetsDeleted) return;
    const cfg = loadCfg();
    const fetchFn = o.fetchFn || rootScope.fetch;
    const baseUrl = String(o.baseUrl || state.lastBaseUrl || cfg.url || '').replace(/\/+$/, '');
    try {
      const r = await _performFetch(fetchFn, baseUrl + '/api/v1/bodyscan/' + encodeURIComponent(jobId) + '/delete', {
        method: 'POST',
        includeAuth: true,
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      _markAssetsDeletedFromServer(null, baseUrl);
    } catch (err) {
      _setDelStatus(_t('pp_del_err', 'Không xoá được trên server: ') + (err && err.message ? err.message : err));
    }
  }

  function _formatAuditPct(raw) {
    const n = Number(raw);
    if (!Number.isFinite(n)) return null;
    return n <= 1.0 ? n * 100.0 : n;
  }

  function renderDetailHtml(scanState, lang, tFn) {
    const src = scanState || state;
    const bs = src.bodyScan || src.body_scan;
    if (!bs) return '';

    const t = tFn || ((key, fb) => (lang === 'en' && DETAIL_EN_FALLBACKS[key]) || fb);
    const summary = extractAuditSummary(src);
    const delAudit = src.deletionAudit || src.deletion_audit || {};
    const delCount = delAudit.deleted_count || 4;
    const privLevel = src.privacyLevel || src.privacy_level || 2;
    const enginePill = summary.engineMode === 'gpu'
      ? `<span class="pp-pill ok" data-pp-engine="gpu">${_esc(t('pp_engine_gpu', 'GPU thật'))}</span>`
      : `<span class="pp-pill warn" data-pp-engine="mock">${_esc(t('pp_engine_mock', 'MOCK TEST'))}</span>`;
    const testBadge = (summary.testOnly && summary.engineMode === 'gpu')
      ? ` <span class="pp-pill warn">TEST_ONLY</span>`
      : '';

    if (src.assetsDeleted) {
      const deletedJsonHref = summary.jsonUrl || (src.artifacts && src.artifacts.json_url) || '';
      const deletedJsonBtn = deletedJsonHref
        ? `<div class="pp-detail-actions" style="margin-top:8px"><a id="ppDlJson" href="${_esc(deletedJsonHref)}" download="body_scan.json" class="pp-btn" style="text-decoration:none">${_esc(t('pp_dl_json', 'Tải JSON'))}</a></div>`
        : '';
      return `<div class="region-h">${_esc(t('overview_title', 'Tổng quan cơ thể'))} ${enginePill}${testBadge}</div>
        <div class="pp-detail-card">
          <div style="color:#065f46;font-weight:700">✓ ${_esc(t('pp_del_done', 'Đã xoá hình 3D & ảnh trên server (vẫn giữ 16 số đo ở cột trái & báo cáo PDF).'))}</div>
          <div id="ppDelStatus" style="font-size:11px;color:#7a7266;margin-top:4px">${_esc(t('pp_holo_back', 'Khung trung tâm đã trả về hình hologram minh hoạ.'))}</div>
          ${deletedJsonBtn}
        </div>`;
    }

    const hasLoaded = src.hasLoadedMesh !== undefined ? Boolean(src.hasLoadedMesh) : hasMesh();
    if (!hasLoaded) return '';

    const realActive = src.realActive !== undefined ? Boolean(src.realActive) : Boolean(state.realActive);
    const toggleLabel = realActive
      ? t('pp_show_holo', 'Xem hologram mẫu')
      : t('pp_show_3d', 'Xem mô hình 3D thật');

    const modeLabel = (m) => {
      if (!m) return '';
      if (m === 'dual_layer') return t('pp_rmode_short_dual_layer', 'Toàn diện 2 lớp (dual_layer)');
      if (m === 'hi_res_head') return t('pp_rmode_short_hi_res_head', 'Chỉ đầu chi tiết (hi_res_head)');
      if (m === 'calibrated_mhr') return t('pp_rmode_short_calibrated_mhr', 'Lõi chuẩn đo nhanh (calibrated_mhr)');
      return m;
    };

    let modeHtml = '';
    if (summary.effectiveReconstructionMode || summary.reconstructionMode) {
      const eff = summary.effectiveReconstructionMode || summary.reconstructionMode;
      const req = summary.reconstructionMode || eff;
      const modeText = summary.wasDowngraded
        ? `${_esc(modeLabel(req))} → ${_esc(modeLabel(eff))} <span class="pp-pill warn">${_esc(t('pp_mode_downgraded', 'Tự hạ mức'))}</span>`
        : _esc(modeLabel(eff));
      const layerHtml = summary.hasCoreMesh
        ? ` · <span>${_esc(t('pp_layer_viewing', 'Đang xem:'))} <b>${_esc(summary.meshViewLayer === 'core' ? t('pp_layer_core', 'Lõi đo MHR') : t('pp_layer_outer', 'Lớp ngoài thật'))}</b></span>`
        : '';
      modeHtml = `<div style="margin-top:4px"><b>${_esc(t('pp_mode_used', 'Chế độ dựng 3D:'))}</b> ${modeText}${layerHtml}</div>`;
    }

    let backendsHtml = '';
    const bu = summary.backendsUsed;
    if (bu && typeof bu === 'object') {
      const chips = [];
      if (bu.segmentation) {
        const segCls = /grabcut|fallback/i.test(String(bu.segmentation)) ? ' warn' : ' ok';
        chips.push(`<span class="pp-pill${segCls}" data-pp-backend="segmentation">seg: ${_esc(bu.segmentation)}</span>`);
      }
      if (bu.texture_rasterizer) {
        const texCls = /cpu|zbuf|fallback/i.test(String(bu.texture_rasterizer)) ? ' warn' : ' ok';
        chips.push(`<span class="pp-pill${texCls}" data-pp-backend="texture_rasterizer">tex: ${_esc(bu.texture_rasterizer)}</span>`);
      }
      if (bu.moge2_depth_used !== undefined && bu.moge2_depth_used !== null) {
        const mogeOn = Boolean(bu.moge2_depth_used);
        const mogeVal = typeof bu.moge2_depth_used === 'boolean'
          ? (mogeOn ? 'moge2:on' : 'moge2:off')
          : `moge2:${bu.moge2_depth_used}`;
        const mogeCls = mogeOn ? ' ok' : ' warn';
        chips.push(`<span class="pp-pill${mogeCls}" data-pp-backend="moge2_depth_used">${_esc(mogeVal)}</span>`);
      }
      if (bu.fitting_silhouette) {
        const fitCls = /cpu|proj|fallback/i.test(String(bu.fitting_silhouette)) ? ' warn' : ' ok';
        chips.push(`<span class="pp-pill${fitCls}" data-pp-backend="fitting_silhouette">fit: ${_esc(bu.fitting_silhouette)}</span>`);
      }
      if (chips.length > 0) {
        backendsHtml = `<div class="pp-backends-row" style="margin-top:4px;display:flex;flex-wrap:wrap;gap:4px;align-items:center"><b>${_esc(t('pp_backends_title', 'Pipeline 3D:'))}</b> ${chips.join(' ')}</div>`;
      }
    }

    let auditHtml = '';
    const audit = summary.proportionAudit;
    if (audit && typeof audit === 'object') {
      const rawSt = String(audit.status || 'estimated_only');
      const st = (rawSt === 'matched' || rawSt === 'mismatch_warning' || rawSt === 'estimated_only')
        ? rawSt
        : 'estimated_only';
      const stCls = st === 'matched' ? 'ok' : (st === 'mismatch_warning' ? 'bad' : '');
      const stText = st === 'matched'
        ? t('pp_audit_status_matched', 'Khớp tỉ lệ')
        : (st === 'mismatch_warning'
          ? t('pp_audit_status_mismatch_warning', 'Cảnh báo lệch')
          : t('pp_audit_status_estimated_only', 'Tự ước lượng'));
      const rows = [];
      if (Number.isFinite(Number(audit.estimated_weight_kg)) && Number(audit.estimated_weight_kg) > 0) {
        rows.push(`<div class="pp-audit-row"><span>${_esc(t('pp_audit_est_w', 'Cân nặng ước lượng'))}</span><b>${Number(audit.estimated_weight_kg).toFixed(1)} kg</b></div>`);
      }
      if (Number.isFinite(Number(audit.declared_weight_kg)) && Number(audit.declared_weight_kg) > 0) {
        const diffPart = Number.isFinite(Number(audit.weight_diff_pct))
          ? ` (${Number(audit.weight_diff_pct).toFixed(1)}%)`
          : '';
        rows.push(`<div class="pp-audit-row"><span>${_esc(t('pp_audit_decl_w', 'Cân nặng khai báo'))}</span><b>${Number(audit.declared_weight_kg).toFixed(1)} kg${_esc(diffPart)}</b></div>`);
      }
      const mhrPct = _formatAuditPct(audit.mhr_silhouette_iou);
      if (mhrPct !== null) {
        rows.push(`<div class="pp-audit-row"><span>${_esc(t('pp_audit_iou', 'Độ khớp ảnh MHR'))}</span><b>${mhrPct.toFixed(1)}%</b></div>`);
      }
      const outerPct = _formatAuditPct(audit.outer_silhouette_iou);
      if (outerPct !== null) {
        rows.push(`<div class="pp-audit-row"><span>${_esc(t('pp_audit_outer_iou', 'Độ khớp lớp ngoài'))}</span><b>${outerPct.toFixed(1)}%</b></div>`);
      }
      auditHtml = `<div class="pp-audit-box" data-audit-status="${_esc(st)}">
        <div class="pp-audit-row" style="margin-bottom:4px">
          <b>${_esc(t('pp_audit_title', 'Đối chiếu tỉ lệ cơ thể'))}</b>
          <span class="pp-pill ${stCls}">${_esc(stText)}</span>
        </div>
        ${rows.join('')}
      </div>`;
    }

    let warnHtml = '';
    if (summary.warnings.length > 0) {
      const items = summary.warnings.map((w) => `<li>${_esc(w)}</li>`).join('');
      warnHtml = `<div class="pp-warn-list">
        <b>${_esc(t('pp_warnings_title', 'Lưu ý / Cảnh báo:'))}</b>
        <ul>${items}</ul>
      </div>`;
    }

    const coreBtnHtml = summary.hasCoreMesh
      ? `<button type="button" id="ppToggleCoreBtn" class="pp-btn">${_esc(summary.meshViewLayer === 'core' ? t('pp_show_outer_mesh', 'Xem lớp ngoài thật') : t('pp_show_core_mesh', 'Xem lõi đo MHR'))}</button>`
      : '';

    const glbTarget = getActiveGlbDownloadTarget(src);
    const glbHref = glbTarget.url || '#';
    const glbFilename = glbTarget.filename;
    const jsonHref = summary.jsonUrl || (src.artifacts && src.artifacts.json_url) || '#';

    return `<div class="region-h">${_esc(t('pp_detail_title', 'Mô hình 3D Alpha Scan Plus/PRO'))} ${enginePill}${testBadge}</div>
      <div class="pp-detail-card">
        <div><b>${_esc(t('pp_privacy_used', 'Mức bảo mật:'))}</b> ${_esc(t('pp_lvl_short_' + privLevel, 'Mức ' + privLevel))}</div>
        ${modeHtml}
        ${backendsHtml}
        <div style="color:#065f46;margin-top:4px">✓ ${delCount} ${_esc(t('pp_raw_shredded', 'ảnh gốc đã xoá trên server ngay sau khi dựng 3D'))}</div>
        ${auditHtml}
        ${warnHtml}
        <div style="color:#6f685c;margin-top:6px">${_esc(t('pp_ctrl_tip', 'Mô hình đang tự xoay 360° — dùng chuột kéo xoay / lăn thu phóng, hoặc bấm Trước / Sau / Xoay.'))}</div>
        <div class="pp-detail-actions">
          <button type="button" id="ppToggleModeBtn" class="pp-btn">${_esc(toggleLabel)}</button>
          ${coreBtnHtml}
          <a id="ppDlGlb" href="${_esc(glbHref)}" download="${_esc(glbFilename)}" class="pp-btn" style="text-decoration:none">${_esc(t('pp_dl_glb', 'Tải .glb'))}</a>
          <a id="ppDlJson" href="${_esc(jsonHref)}" download="body_scan.json" class="pp-btn" style="text-decoration:none">${_esc(t('pp_dl_json', 'Tải JSON'))}</a>
          <button type="button" id="ppDeleteSrvBtn" class="pp-btn danger">${_esc(t('pp_del_srv_btn', 'Xoá hình 3D/ảnh trên server'))}</button>
        </div>
        <div id="ppDelStatus" style="font-size:11px;color:#991b1b;margin-top:4px"></div>
      </div>`;
  }

  function renderDetailCard(detailEl) {
    if (!detailEl || !state.bodyScan) return false;
    if (!state.assetsDeleted && !hasMesh()) return false;
    const html = renderDetailHtml({
      bodyScan: state.bodyScan,
      serverMode: state.serverMode,
      testOnly: state.testOnly,
      assetsDeleted: state.assetsDeleted,
      privacyLevel: state.privacyLevel,
      activeReconstructionMode: state.activeReconstructionMode,
      effectiveReconstructionMode: state.effectiveReconstructionMode,
      proportionAudit: state.proportionAudit,
      backendsUsed: state.backendsUsed,
      warnings: state.warnings,
      artifacts: state.artifacts,
      deletionAudit: state.deletionAudit,
      realActive: state.realActive,
      hasLoadedMesh: hasMesh(),
      meshViewLayer: state.meshViewLayer,
    }, _lang(), _t);
    if (!html) return false;
    detailEl.innerHTML = html;

    const tBtn = document.getElementById('ppToggleModeBtn');
    if (tBtn) tBtn.onclick = () => setRealActive(!state.realActive);
    const coreBtn = document.getElementById('ppToggleCoreBtn');
    if (coreBtn) coreBtn.onclick = () => { toggleCoreMeshView().catch(() => {}); };
    const dlGlbBtn = document.getElementById('ppDlGlb');
    if (dlGlbBtn) {
      dlGlbBtn.onclick = (ev) => {
        if (ev && typeof ev.preventDefault === 'function') ev.preventDefault();
        return downloadArtifact('glb').catch((err) => {
          if (!(err && err.status === 410)) {
            _setDelStatus(_t('pp_dl_err', 'Không tải được tệp: ') + (err && err.message ? err.message : err));
          }
        });
      };
    }
    const dlJsonBtn = document.getElementById('ppDlJson');
    if (dlJsonBtn) {
      dlJsonBtn.onclick = (ev) => {
        if (ev && typeof ev.preventDefault === 'function') ev.preventDefault();
        return downloadArtifact('json').catch((err) => {
          _setDelStatus(_t('pp_dl_err', 'Không tải được tệp: ') + (err && err.message ? err.message : err));
        });
      };
    }
    const dBtn = document.getElementById('ppDeleteSrvBtn');
    if (dBtn) dBtn.onclick = () => deleteServerAssets();
    _syncDownloadLinks();
    return true;
  }

  function init(bridge) {
    state.bridge = bridge || null;
    _injectStyles();
  }

  const api = {
    ANATOMICAL_ORDER,
    LABELS_VI,
    LABELS_EN,
    RECONSTRUCTION_MODES,
    DEFAULT_RECONSTRUCTION_MODE,
    ARTIFACT_URL_REFRESH_MS,
    isLoopbackUrl,
    isNgrokUrl,
    buildRequestHeaders,
    inferCardinalRoleFromFilename,
    enrichFileContentFingerprints,
    findDuplicateFileSignatures,
    assignBatchFilesToSlots,
    classifyHealthPayload,
    validatePrivacyAndConnection,
    parseGlbBinary,
    disposeObject3D,
    extract16GirthRows,
    applyScanResultToClient,
    extractAuditSummary,
    getActiveGlbDownloadTarget,
    buildScanFormData,
    renderLeft16Html,
    renderReport16Html,
    renderDetailHtml,
    init,
    loadCfg,
    saveCfg,
    openModal,
    closeModal,
    buildModalHtml: _buildModalHtml,
    applyCompletedPayload: _applyCompletedPayload,
    pollJobUntilDone: _pollJobUntilDone,
    refreshArtifactUrlsIfNeeded,
    downloadArtifact,
    toggleCoreMeshView,
    resetSession,
    hasMesh,
    isReal3DActive,
    setRealActive,
    getResult,
    getDeletedFlag,
    getStateSnapshot,
    renderLeft16Box,
    renderDetailCard,
    onHealthCheck,
    switchToServerConsent: _switchToServerConsent,
    formatHealthBackendsTag: _formatHealthBackendsTag,
    deleteServerAssets,
    getLang: _lang,
    t: _t,
  };

  rootScope.AlphaPlusPro = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
