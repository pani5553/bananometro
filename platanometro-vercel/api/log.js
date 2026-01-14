// api/log.js
// Endpoint: POST /api/log
// Logs mínimos en consola (Vercel -> Functions Logs)

export default async function handler(req, res) {
  const now = new Date().toISOString();
  const body = req?.method === "POST" ? (req.body || {}) : {};

  console.log("[LOG]", now, body);

  res.status(200).json({ ok: true, now });
}
