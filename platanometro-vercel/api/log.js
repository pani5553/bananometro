// api/log.js
// Guarda feedbacks en memoria (MVP) + logs en consola
// POST /api/log  -> registra evento
// GET  /api/log?stats=1 -> devuelve estadísticas (best-effort en Vercel)

let store = globalThis.__PLAT_STORE__;
if (!store) {
  store = globalThis.__PLAT_STORE__ = {
    connected: 0,
    feedback: [],
    perfect: 0
  };
}

function safeBody(req) {
  try { return req.body || {}; } catch { return {}; }
}

export default async function handler(req, res) {
  const now = new Date().toISOString();

  // --- GET stats ---
  if (req.method === "GET" && req.query?.stats) {
    const byLabel = {};
    for (const f of store.feedback) {
      const k = f.confirmedLabel || "unknown";
      byLabel[k] = (byLabel[k] || 0) + 1;
    }
    return res.status(200).json({
      ok: true,
      now,
      connected: store.connected,
      detected_perfect: store.perfect,
      feedback_total: store.feedback.length,
      feedback_by_label: byLabel,
      note: "MVP en memoria: en Vercel puede reiniciarse entre instancias."
    });
  }

  // --- POST event ---
  const body = safeBody(req);
  const event = body.event || "unknown";

  if (event === "connected") store.connected += 1;
  if (event === "detected_perfect") store.perfect += 1;

  if (event === "user_feedback") {
    // Guardamos solo lo útil (sin snapshot base64 en memoria)
    store.feedback.push({
      ts: body.ts || now,
      userId: body.userId || null,
      confirmedLabel: body.confirmedLabel || null,
      predictedLabel: body.predictedLabel || null,
      predictedConf: body.predictedConf ?? null,
      bananaScore: body.bananaScore ?? null,
      hasBanana: !!body.hasBanana
    });

    // limitamos tamaño para no crecer infinito
    if (store.feedback.length > 200) store.feedback.shift();
  }

  // Evitar logs gigantes si viene snapshot
  const safe = { ...body };
  if (typeof safe.snapshotJpeg === "string") {
    safe.snapshotJpeg = `[jpeg dataurl len=${safe.snapshotJpeg.length}]`;
  }

  console.log("[LOG]", now, safe);
  res.status(200).json({ ok: true, now });
}
