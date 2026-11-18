import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../frontend_lab/index.html', import.meta.url), 'utf8');

test('InBody UI calls the Alpha Scan multipart endpoint and does not retain source base64', () => {
  assert.match(html, /AlphaPlusPro\.loadCfg\(\)/);
  assert.match(html, /new FormData\(\)/);
  assert.match(html, /\/api\/v1\/inbody\/extract/);
  assert.match(html, /form\.append\('consent_granted','true'\)/);
  assert.match(html, /sourceFile:null/);
  assert.doesNotMatch(html, /\/functions\/v1\/parse_inbody/);
});

test('inline script containing ocrInBody has valid JavaScript syntax', () => {
  const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)].map((m) => m[1]);
  const ocrScript = scripts.find((source) => source.includes('async function ocrInBody'));
  assert.ok(ocrScript, 'ocrInBody script not found');
  assert.doesNotThrow(() => new Function(ocrScript));
});
