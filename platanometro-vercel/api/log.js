// api/log.js
// Endpoint: POST /api/log
// Logs mínimos en consola (Vercel -> Functions Logs)

module.exports = async function handler(req, res) {
  const now = new Date().toISOString();

  // Vercel parsea JSON automáticamente si llega Content-Type: application/json
  const body = (req && req.method === "POST") ? (req.body || {}) : {};

  console.log("[LOG]", now, body);

  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify({ ok: true, now }));
};
