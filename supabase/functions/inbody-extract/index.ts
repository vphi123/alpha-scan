// Supabase Edge Function: inbody-extract
// Port của bodyscan_pipeline/inbody_ocr.py + route POST /api/v1/inbody/extract trong api_server.py.
// Giữ nguyên contract request/response để frontend_lab không phải sửa.
//
// Secrets cần đặt (Supabase Dashboard → Edge Functions → Secrets):
//   OPENAI_API_KEY        bắt buộc
//   INBODY_API_KEY        bắt buộc — khoá mà frontend gửi qua "Authorization: Bearer ..."
//   OPENAI_INBODY_MODEL   tuỳ chọn (mặc định gpt-5.6-luna)
//   INBODY_OCR_TIMEOUT_SECONDS tuỳ chọn (mặc định 60)
//
// URL dùng trong ⚙ Kết nối của Alpha Scan:
//   https://<project-ref>.supabase.co/functions/v1/inbody-extract
// Frontend tự nối thêm /api/v1/inbody/extract và /health.

const PROMPT = `You extract measurements from exactly one InBody body-composition report.

Return only the requested structured response. Read values from the uploaded
document; never infer a measurement from a reference range or chart scale.
Use null when a value is absent or unreadable. Numbers must be JSON numbers.

Important rules:

- \`document_status\` is \`single_inbody_report\`, \`not_inbody_report\`, or
  \`ambiguous_multiple_reports\`.
- Convert the printed test time to \`YYYY-MM-DDTHH:mm\`. Vietnamese numeric dates
  use day-month-year. For example \`07.08.2026 06:18\` is
  \`2026-08-07T06:18\`, never August 27.
- Also copy the shortest exact printed date/time into an evidence item with
  field \`measurement.test_datetime\`; deterministic application code verifies
  the conversion from that raw text.
- Identify anatomical left/right from the report labels, never from the
  viewer's perspective.
- Do not confuse Body Fat Mass (kg) with Percent Body Fat (%), Skeletal Muscle
  Mass with lean mass, or visceral fat level with visceral fat area (cm2).
- Segmental Lean Analysis supplies \`lean_kg\` and \`lean_percent\`; Segmental Fat
  Analysis supplies \`fat_kg\` and \`fat_percent\`.
- Preserve signs on weight/fat/muscle control values.
- Preserve printed IDs as strings so leading zeroes survive.

Return an evidence item for every non-null extracted leaf field. Each evidence
item must contain the complete dotted field path, the shortest exact visible
text supporting it, one-based page number, and report region. Do not create
evidence for null values.`;

const MAX_BYTES = 20 * 1024 * 1024;

// ---------- JSON schema (strict structured output) ----------
const num = { type: ["number", "null"] };
const int = { type: ["integer", "null"] };
const str = { type: ["string", "null"] };
const obj = (props: Record<string, unknown>) => ({
  type: "object",
  additionalProperties: false,
  properties: props,
  required: Object.keys(props),
});
const segment = obj({ lean_kg: num, lean_percent: num, fat_kg: num, fat_percent: num });

export const SCHEMA = obj({
  data: obj({
    document_status: {
      type: "string",
      enum: ["single_inbody_report", "not_inbody_report", "ambiguous_multiple_reports"],
    },
    identity: obj({ inbody_sheet_id: str }),
    measurement: obj({
      test_datetime: str,
      device_model: str,
      height_cm: num,
      age: int,
      gender: { type: ["string", "null"], enum: ["male", "female", null] },
    }),
    body_composition: obj({
      weight_kg: num,
      total_body_water_l: num,
      protein_kg: num,
      minerals_kg: num,
      body_fat_mass_kg: num,
    }),
    muscle_fat_obesity: obj({ smm_kg: num, pbf_percent: num, bmi: num }),
    segmental: obj({
      left_arm: segment,
      right_arm: segment,
      trunk: segment,
      left_leg: segment,
      right_leg: segment,
    }),
    weight_control: obj({
      target_weight_kg: num,
      weight_control_kg: num,
      fat_control_kg: num,
      muscle_control_kg: num,
    }),
    other_metrics: obj({
      inbody_score: num,
      visceral_fat_level: int,
      visceral_fat_area_cm2: num,
      bmr_kcal: num,
      fat_free_mass_kg: num,
      smi_kg_m2: num,
      obesity_degree_percent: num,
      waist_hip_ratio: num,
      recommended_calorie_intake_kcal: num,
    }),
  }),
  evidence: {
    type: "array",
    items: obj({ field: { type: "string" }, raw_text: str, page: int, region: str }),
  },
});

// ---------- helpers ----------
class InBodyError extends Error {
  constructor(public detail: string, public status = 502) {
    super(detail);
  }
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
    "authorization, x-api-key, content-type, apikey, x-client-info, ngrok-skip-browser-warning",
  "Access-Control-Max-Age": "86400",
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json; charset=utf-8" },
  });

const pad = (n: number) => String(n).padStart(2, "0");
const iso = (y: number, mo: number, d: number, h: number, mi: number): string | null => {
  const dt = new Date(Date.UTC(y, mo - 1, d, h, mi));
  if (
    dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d ||
    h > 23 || mi > 59
  ) return null;
  return `${y}-${pad(mo)}-${pad(d)}T${pad(h)}:${pad(mi)}`;
};

/** Chuẩn hoá chuỗi ngày giờ in trên phiếu (ngày-tháng-năm kiểu VN) → YYYY-MM-DDTHH:mm. */
export function normalizeDatetime(value: unknown): string | null {
  if (typeof value !== "string") return null;
  let s = value.trim().replace(/\s*([./:-])\s*/g, "$1").replace(/\s+/g, " ");
  let m = s.match(/^(\d{1,2})([./-])(\d{1,2})\2(\d{4})[ T](\d{1,2}):(\d{2})$/);
  if (m) return iso(+m[4], +m[3], +m[1], +m[5], +m[6]); // d/m/Y, d.m.Y, d-m-Y
  m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})$/);
  if (m) return iso(+m[1], +m[2], +m[3], +m[4], +m[5]);
  m = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4}) (\d{1,2}):(\d{2}) ?([AaPp][Mm])$/);
  if (m) {
    let h = +m[4] % 12;
    if (/p/i.test(m[6])) h += 12;
    return iso(+m[3], +m[1], +m[2], h, +m[5]);
  }
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value) ? value : null;
}

function sniffMime(raw: Uint8Array): string {
  const b = raw;
  if (b[0] === 0x25 && b[1] === 0x50 && b[2] === 0x44 && b[3] === 0x46 && b[4] === 0x2d) {
    return "application/pdf";
  }
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "image/jpeg";
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return "image/png";
  throw new InBodyError("Chỉ hỗ trợ PDF, JPG, JPEG và PNG.", 415);
}

function toBase64(raw: Uint8Array): string {
  let bin = "";
  const CH = 0x8000;
  for (let i = 0; i < raw.length; i += CH) {
    bin += String.fromCharCode(...raw.subarray(i, i + CH));
  }
  return btoa(bin);
}

// deno-lint-ignore no-explicit-any
type Extraction = any;

/** Chuyển kết quả model sang contract của modal InBody (giống adapt_for_alpha). */
export function adaptForAlpha(ex: Extraction, model: string) {
  const d = ex.data;
  if (d.document_status === "not_inbody_report") {
    throw new InBodyError("Tệp tải lên không phải phiếu InBody.", 422);
  }
  if (d.document_status === "ambiguous_multiple_reports") {
    throw new InBodyError("Tệp chứa nhiều phiếu InBody; hãy tải từng phiếu riêng.", 422);
  }
  const flags: string[] = [];

  // Ngày giờ: tin chuỗi gốc trên phiếu hơn là model.
  const modelDt = d.measurement.test_datetime;
  const ev = (ex.evidence || []).find((e: Extraction) => e.field === "measurement.test_datetime");
  const evDt = ev ? normalizeDatetime(ev.raw_text) : null;
  if (evDt) {
    if (modelDt && evDt !== modelDt) {
      flags.push("Ngày giờ đã được hiệu chỉnh theo chuỗi gốc trên phiếu; vui lòng đối chiếu lại.");
    }
    d.measurement.test_datetime = evDt;
  } else if (modelDt) {
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(modelDt)) d.measurement.test_datetime = null;
    flags.push("Không có bằng chứng ngày giờ gốc; vui lòng đối chiếu ngày đo.");
  }

  const bc = d.body_composition, mfo = d.muscle_fat_obesity;
  const required: Record<string, unknown> = {
    "measurement.test_datetime": d.measurement.test_datetime,
    "body_composition.weight_kg": bc.weight_kg,
    "body_composition.body_fat_mass_kg": bc.body_fat_mass_kg,
    "muscle_fat_obesity.smm_kg": mfo.smm_kg,
    "muscle_fat_obesity.pbf_percent": mfo.pbf_percent,
  };
  const missing = Object.keys(required).filter((k) => required[k] == null);
  const failed: string[] = [];
  if (bc.weight_kg && bc.body_fat_mass_kg != null && mfo.pbf_percent != null) {
    if (Math.abs((bc.body_fat_mass_kg / bc.weight_kg) * 100 - mfo.pbf_percent) > 0.5) {
      failed.push("body_fat_percentage");
    }
  }
  if (bc.weight_kg && d.measurement.height_cm && mfo.bmi != null) {
    const calc = bc.weight_kg / (d.measurement.height_cm / 100) ** 2;
    if (Math.abs(calc - mfo.bmi) > 0.3) failed.push("bmi");
  }
  if (missing.length) flags.push("Thiếu trường cần kiểm tra: " + missing.join(", "));
  if (failed.length) flags.push("Số liệu chưa khớp phép đối chiếu: " + failed.join(", "));

  const seg = (s: Extraction) => ({ lean_kg: s?.lean_kg ?? null, fat_kg: s?.fat_kg ?? null });
  const dt = d.measurement.test_datetime;
  return {
    ok: true,
    data: {
      measured_datetime: dt,
      measured_date: dt ? dt.slice(0, 10) : null,
      device_model: d.measurement.device_model,
      weight_kg: bc.weight_kg,
      pbf_percent: mfo.pbf_percent,
      smm_kg: mfo.smm_kg,
      fat_mass_kg: bc.body_fat_mass_kg,
      visceral_fat_level: d.other_metrics.visceral_fat_level,
      inbody_score: d.other_metrics.inbody_score,
      segmental: {
        arm_left: seg(d.segmental.left_arm),
        arm_right: seg(d.segmental.right_arm),
        trunk: seg(d.segmental.trunk),
        leg_left: seg(d.segmental.left_leg),
        leg_right: seg(d.segmental.right_leg),
      },
    },
    validation: { passed: !missing.length && !failed.length, missing_required_fields: missing, failed_checks: failed },
    flags,
    provider: "openai",
    model,
  };
}

async function callOpenAI(
  raw: Uint8Array,
  mime: string,
  filename: string,
  apiKey: string,
  model: string,
  timeoutSec: number,
): Promise<Extraction> {
  const b64 = toBase64(raw);
  const content = mime === "application/pdf"
    ? { type: "input_file", filename: filename || "inbody.pdf", file_data: `data:application/pdf;base64,${b64}` }
    : { type: "input_image", image_url: `data:${mime};base64,${b64}`, detail: "original" };

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutSec * 1000);
  let r: Response;
  try {
    r = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      signal: ctrl.signal,
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        instructions: PROMPT,
        input: [{
          role: "user",
          content: [
            { type: "input_text", text: "Extract this single InBody report using the required schema." },
            content,
          ],
        }],
        reasoning: { effort: "none" },
        text: { format: { type: "json_schema", name: "inbody_extraction", strict: true, schema: SCHEMA } },
        store: false,
      }),
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") {
      throw new InBodyError(`OCR vượt quá giới hạn ${timeoutSec} giây.`, 504);
    }
    throw new InBodyError("OpenAI tạm thời không khả dụng.", 503);
  } finally {
    clearTimeout(timer);
  }

  const body = await r.json().catch(() => null);
  if (!r.ok) {
    const msg = body?.error?.message || `HTTP ${r.status}`;
    if (r.status === 429) throw new InBodyError("OpenAI hết quota hoặc đang bị giới hạn tốc độ.", 429);
    if (r.status === 401) throw new InBodyError("OPENAI_API_KEY trên server không hợp lệ.", 503);
    if ([408, 409, 500, 502, 503, 504].includes(r.status)) {
      throw new InBodyError("OpenAI tạm thời không khả dụng.", 503);
    }
    throw new InBodyError(`OpenAI xử lý thất bại: ${msg}`, 502);
  }

  let text: string | null = null;
  for (const item of body?.output || []) {
    if (item.type !== "message") continue;
    for (const c of item.content || []) {
      if (c.type === "refusal") throw new InBodyError("OpenAI từ chối xử lý tài liệu này.", 422);
      if (c.type === "output_text") text = c.text;
    }
  }
  if (!text) throw new InBodyError("OpenAI không trả về structured output.", 502);
  try {
    return JSON.parse(text);
  } catch {
    throw new InBodyError("OpenAI trả về JSON không hợp lệ.", 502);
  }
}

function checkAuth(req: Request, expected: string): boolean {
  const h = req.headers.get("authorization") || "";
  const token = h.toLowerCase().startsWith("bearer ")
    ? h.slice(7).trim()
    : (req.headers.get("x-api-key") || "").trim();
  if (!token || token.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < token.length; i++) diff |= token.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}

export async function handle(req: Request): Promise<Response> {
  const env = (k: string) => (Deno.env.get(k) || "").trim();
  const openaiKey = env("OPENAI_API_KEY");
  const serviceKey = env("INBODY_API_KEY");
  const model = env("OPENAI_INBODY_MODEL") || "gpt-5.6-luna";
  const timeoutSec = Number(env("INBODY_OCR_TIMEOUT_SECONDS")) || 60;
  const path = new URL(req.url).pathname.replace(/\/+$/, "");

  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });

  if (req.method === "GET" && (path.endsWith("/health") || path.endsWith("/inbody-extract"))) {
    return json({
      ok: true,
      status: "ready",
      model_ready: true,
      mode: "inbody_only",
      auth_required: true,
      inbody_ocr: { enabled: Boolean(openaiKey), provider: "openai", model, timeout_seconds: timeoutSec },
    });
  }

  if (req.method === "POST" && path.endsWith("/inbody/extract")) {
    try {
      if (!serviceKey) throw new InBodyError("Server chưa cấu hình INBODY_API_KEY.", 503);
      if (!checkAuth(req, serviceKey)) throw new InBodyError("Sai hoặc thiếu API key.", 401);
      if (!openaiKey) throw new InBodyError("OPENAI_API_KEY chưa được cấu hình trên server.", 503);

      let form: FormData;
      try {
        form = await req.formData();
      } catch {
        throw new InBodyError("Request phải là multipart/form-data có trường file.", 400);
      }
      const consent = String(form.get("consent_granted") ?? "false").trim().toLowerCase();
      if (!["true", "1", "yes", "on"].includes(consent)) {
        throw new InBodyError("Explicit client consent (consent_granted=true) is required.", 403);
      }
      const file = form.get("file");
      if (!(file instanceof File)) throw new InBodyError("Thiếu trường file.", 400);
      if (file.size === 0) throw new InBodyError("Tệp tải lên rỗng.", 400);
      if (file.size > MAX_BYTES) throw new InBodyError("Tệp vượt quá giới hạn 20 MB.", 413);

      const raw = new Uint8Array(await file.arrayBuffer());
      const mime = sniffMime(raw);
      const extraction = await callOpenAI(raw, mime, file.name, openaiKey, model, timeoutSec);
      return json(adaptForAlpha(extraction, model));
    } catch (e) {
      if (e instanceof InBodyError) return json({ ok: false, detail: e.detail }, e.status);
      console.error("inbody-extract unexpected error", e);
      return json({ ok: false, detail: "Lỗi không xác định trên server." }, 500);
    }
  }

  return json({ ok: false, detail: "Not Found — function này chỉ phục vụ trích InBody." }, 404);
}

if (typeof Deno !== "undefined" && typeof Deno.serve === "function") {
  Deno.serve(handle);
}
