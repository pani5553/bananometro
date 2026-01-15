// public/app.js
// @ts-nocheck

// ===== ONNX =====
const IMG = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

const MIN_CONF = 0.45;
const PERFECT_CLASS = "perfecto";

// ===== Detector =====
const DET_INTERVAL_MS = 500;     // sube si va lento (700–1200)
const BANANA_MIN_SCORE = 0.30;   // sube si hay falsos (0.35–0.50)
const BBOX_MARGIN = 0.35;        // margen extra para clasificador
const HOLD_BBOX_MS = 1500;       // mantiene bbox aunque se pierda un momento
const SMOOTH_ALPHA = 0.55;       // suaviza bbox (0..1)

// ===== Elements =====
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const userBoxEl = document.getElementById("userbox");
const feedbackMsgEl = document.getElementById("feedback_msg");

// Botones feedback
const btnVerde = document.getElementById("btn_verde");
const btnPerfecto = document.getElementById("btn_perfecto");
const btnPasado = document.getElementById("btn_pasado");
const btnPodrido = document.getElementById("btn_podrido");
const btnNo = document.getElementById("btn_no");

// ===== UI helpers =====
function setStatus(t) { statusEl.textContent = t; }
function setError(e) { errorEl.textContent = e ? String(e) : ""; }
function updateFeedbackText(msg) { if (feedbackMsgEl) feedbackMsgEl.textContent = msg; }

// ===== Math =====
function softmax(arr) {
  let max = -Infinity;
  for (const v of arr) max = Math.max(max, v);
  let sum = 0;
  const exps = arr.map(v => Math.exp(v - max));
  for (const e of exps) sum += e;
  return exps.map(e => e / sum);
}

// ===== User ID =====
function getOrCreateUserId() {
  const key = "platanometro_user_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = "u_" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem(key, id);
  }
  return id;
}
const userId = getOrCreateUserId();

// ===== Logging mínimo =====
async function logUserEvent(event, extra = {}) {
  try {
    await fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, userId, ...extra, ts: new Date().toISOString() })
    });
  } catch (_) {}
}

// ===== Setup =====
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
}

async function loadLabels() {
  const res = await fetch("/models/labels.json", { cache: "no-store" });
  if (!res.ok) throw new Error("No se pudo cargar /models/labels.json");
  const labels = await res.json();
  if (!Array.isArray(labels) || labels.length === 0) throw new Error("labels.json inválido");
  return labels;
}

async function loadOnnxModel() {
  if (ort?.env?.wasm) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  }
  return await ort.InferenceSession.create("/models/banana.onnx");
}

async function loadDetector() {
  if (typeof tf !== "undefined" && tf?.ready) await tf.ready();
  return await cocoSsd.load();
}

// ===== Preprocess / Verdict =====
function preprocess(imgData) {
  const input = new Float32Array(1 * 3 * IMG * IMG);
  let pR = 0, pG = IMG * IMG, pB = 2 * IMG * IMG;

  for (let i = 0; i < IMG * IMG; i++) {
    const r = imgData[i * 4 + 0] / 255;
    const g = imgData[i * 4 + 1] / 255;
    const b = imgData[i * 4 + 2] / 255;

    input[pR++] = (r - MEAN[0]) / STD[0];
    input[pG++] = (g - MEAN[1]) / STD[1];
    input[pB++] = (b - MEAN[2]) / STD[2];
  }
  return input;
}

function verdictFrom(label, conf) {
  if (conf < MIN_CONF) return "NO SEGURO";
  return (label === PERFECT_CLASS) ? "SI" : "NO";
}

// ===== Overlay =====
function syncOverlayToVideo() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  if (overlay.width !== vw || overlay.height !== vh) {
    overlay.width = vw;
    overlay.height = vh;
  }
}

function drawOverlay(bbox, titleLine, subtitleLine) {
  const ctx = overlay.getContext("2d");
  syncOverlayToVideo();
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!bbox) return;

  const [x, y, w, h] = bbox;

  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(0,255,0,0.9)";
  ctx.strokeRect(x, y, w, h);

  const label = subtitleLine ? (titleLine + " • " + subtitleLine) : titleLine;
  if (!label) return;

  ctx.font = "18px system-ui";
  const pad = 6;
  const textW = ctx.measureText(label).width;
  const boxH = 26;
  const bx = x;
  const by = Math.max(0, y - boxH);

  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(bx, by, textW + pad * 2, boxH);

  ctx.fillStyle = "white";
  ctx.fillText(label, bx + pad, by + 18);
}

// ===== BBox smoothing & crop =====
function smoothBbox(prev, next) {
  if (!prev) return next;
  const a = SMOOTH_ALPHA;
  return [
    prev[0] + a * (next[0] - prev[0]),
    prev[1] + a * (next[1] - prev[1]),
    prev[2] + a * (next[2] - prev[2]),
    prev[3] + a * (next[3] - prev[3]),
  ];
}

function drawBboxCropToCanvas(bbox) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  let [x, y, w, h] = bbox;

  // margen
  const mx = w * BBOX_MARGIN;
  const my = h * BBOX_MARGIN;
  x -= mx; y -= my; w += 2 * mx; h += 2 * my;

  // cuadrado
  const side = Math.max(w, h);
  const cx = x + w / 2;
  const cy = y + h / 2;

  let sx = Math.floor(cx - side / 2);
  let sy = Math.floor(cy - side / 2);
  let sw = Math.floor(side);
  let sh = Math.floor(side);

  // clamp
  if (sx < 0) sx = 0;
  if (sy < 0) sy = 0;
  if (sx + sw > vw) sw = vw - sx;
  if (sy + sh > vh) sh = vh - sy;

  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, IMG, IMG);
  return ctx.getImageData(0, 0, IMG, IMG).data;
}

// ===== Live state (para feedback) =====
const liveState = {
  hasBanana: false,
  bananaScore: 0,
  bbox: null,
  clsLabel: null,
  clsConf: 0,
  verdict: "",
};

// ===== Feedback buttons =====
function bindFeedbackButtons() {
  function send(label) {
    logUserEvent("user_feedback", {
      confirmedLabel: label,
      predictedLabel: liveState.clsLabel,
      predictedConf: Number((liveState.clsConf || 0).toFixed(4)),
      bananaScore: Number((liveState.bananaScore || 0).toFixed(4)),
      hasBanana: !!liveState.hasBanana
    });

    updateFeedbackText("Feedback enviado: " + label + " (gracias)");
    setTimeout(
      () => updateFeedbackText("Si la predicción falla, pulsa el estado correcto. (Solo se registra un evento.)"),
      2000
    );
  }

  btnVerde?.addEventListener("click", () => send("verde"));
  btnPerfecto?.addEventListener("click", () => send("perfecto"));
  btnPasado?.addEventListener("click", () => send("pasado"));
  btnPodrido?.addEventListener("click", () => send("podrido"));
  btnNo?.addEventListener("click", () => send("no_banana"));
}

// ===== Main loop =====
(async function main() {
  try {
    setError("");
    bindFeedbackButtons();

    // info usuario (mínimo)
    const countKey = "platanometro_connected_count";
    const prev = Number(localStorage.getItem(countKey) || "0");
    const nowCount = prev + 1;
    localStorage.setItem(countKey, String(nowCount));
    if (userBoxEl) userBoxEl.textContent = "Usuario: " + userId + " | Conexiones (este dispositivo): " + nowCount;

    setStatus("Cargando labels…");
    const labels = await loadLabels();

    setStatus("Cargando modelo ONNX…");
    const session = await loadOnnxModel();

    setStatus("Cargando detector (banana)…");
    const detector = await loadDetector();

    setStatus("Activando cámara…");
    await setupCamera();

    setStatus("Listo. Buscando plátano…");
    await logUserEvent("connected", { ua: navigator.userAgent, localCount: nowCount });

    let busy = false;

    // detección estable
    let lastDetTs = 0;
    let lastSeenTs = 0;
    let smooth = null;
    let bananaScore = 0;

    // log perfecto con cooldown
    let lastPerfectLog = 0;

    setInterval(async () => {
      if (busy) return;
      if (!video.videoWidth) return;

      busy = true;
      try {
        const now = Date.now();

        // --- DETECCIÓN ---
        if (now - lastDetTs > DET_INTERVAL_MS) {
          lastDetTs = now;

          const preds = await detector.detect(video);

          let best = null;
          for (const p of preds) {
            if (p.class === "banana" && p.score >= BANANA_MIN_SCORE) {
              if (!best || p.score > best.score) best = p;
            }
          }

          if (best) {
            bananaScore = best.score;
            lastSeenTs = now;
            smooth = smoothBbox(smooth, best.bbox);
          } else {
            // mantenemos bbox un rato para evitar parpadeo
            if (now - lastSeenTs > HOLD_BBOX_MS) {
              smooth = null;
              bananaScore = 0;
            }
          }
        }

        const hasBanana = !!smooth;
        liveState.hasBanana = hasBanana;
        liveState.bananaScore = bananaScore;
        liveState.bbox = smooth;

        if (!hasBanana) {
          drawOverlay(null, "", "");
          setStatus("No veo un plátano. Acércalo / céntralo / mejor luz.");
          setError("");
          return;
        }

        // --- CLASIFICACIÓN ---
        const imgData = drawBboxCropToCanvas(smooth);
        const input = preprocess(imgData);

        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = new ort.Tensor("float32", input, [1, 3, IMG, IMG]);

        const out = await session.run(feeds);
        const outputName = session.outputNames[0];
        const logits = out[outputName].data;

        const probs = softmax(Array.from(logits));
        let bestI = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestI]) bestI = i;

        const label = labels[bestI] || ("class_" + bestI);
        const conf = probs[bestI];
        const verdict = verdictFrom(label, conf);

        liveState.clsLabel = label;
        liveState.clsConf = conf;
        liveState.verdict = verdict;

        drawOverlay(smooth, "BANANA " + bananaScore.toFixed(2), label + " " + conf.toFixed(2));

        setStatus(
          "Detectado: banana " + bananaScore.toFixed(2) +
          " | Estado: " + label +
          " | Conf: " + conf.toFixed(2) +
          " | Perfecto: " + verdict
        );

        // --- LOG PERFECTO ---
        if (label === PERFECT_CLASS && conf >= 0.75 && (now - lastPerfectLog) > 10000) {
          lastPerfectLog = now;
          await logUserEvent("detected_perfect", {
            conf: Number(conf.toFixed(4)),
            bananaScore: Number(bananaScore.toFixed(4))
          });
        }

        setError("");
      } catch (e) {
        setError(e);
      } finally {
        busy = false;
      }
    }, 300);

  } catch (err) {
    setStatus("Error");
    setError(err);
  }
})();
