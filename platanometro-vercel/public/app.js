// public/app.js
// @ts-nocheck
// Pipeline:
// 1) COCO-SSD detecta "banana" => bbox
// 2) Recortamos bbox (con margen) => 224x224 => ONNX clasifica estado

// Config ONNX
const IMG = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

const MIN_CONF = 0.50;
const PERFECT_CLASS = "perfecto";

// Config detector
const DET_INTERVAL_MS = 800;     // cada cuánto detectamos (sube si va lento)
const BANANA_MIN_SCORE = 0.30;   // umbral detector (bájalo si no detecta)
const BBOX_MARGIN = 0.30;        // margen alrededor del bbox (25%)

// Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");     // 224x224 hidden
const overlay = document.getElementById("overlay");   // dibujo bbox
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const userBoxEl = document.getElementById("userbox");

// Utils
function softmax(arr) {
  let max = -Infinity;
  for (const v of arr) max = Math.max(max, v);
  let sum = 0;
  const exps = arr.map(v => Math.exp(v - max));
  for (const e of exps) sum += e;
  return exps.map(e => e / sum);
}

function setStatus(text) { statusEl.textContent = text; }
function setError(msg) { errorEl.textContent = msg ? String(msg) : ""; }

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

async function logUserEvent(event, extra = {}) {
  try {
    await fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, userId, ...extra, ts: new Date().toISOString() })
    });
  } catch (_) {}
}

async function setupCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Este navegador no soporta getUserMedia (cámara).");
  }

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
  if (!Array.isArray(labels) || labels.length === 0) {
    throw new Error('labels.json debe ser un array JSON, p.ej. ["pasado","perfecto","podrido","verde"]');
  }
  return labels;
}

async function loadOnnxModel() {
  if (ort?.env?.wasm) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  }
  return await ort.InferenceSession.create("/models/banana.onnx");
}

async function loadDetector() {
  // TFJS inicializa backend
  if (typeof tf !== "undefined" && tf?.ready) {
    await tf.ready();
    // opcional: forzar webgl si existe
    // try { await tf.setBackend("webgl"); } catch(_) {}
  }
  // cocoSsd global (por script)
  return await cocoSsd.load();
}

// Preprocess para ONNX (NCHW float32)
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

// Dibuja bbox en overlay
function drawOverlay(bbox, labelText) {
  const ctx = overlay.getContext("2d");
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;

  // sincroniza tamaño real del canvas con el vídeo (para que escale bien)
  // y luego CSS lo ajusta a 360px
  if (overlay.width !== vw || overlay.height !== vh) {
    overlay.width = vw;
    overlay.height = vh;
  }

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!bbox) return;

  const [x, y, w, h] = bbox;

  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(0,255,0,0.9)";
  ctx.strokeRect(x, y, w, h);

  if (labelText) {
    ctx.font = "20px system-ui";
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    const pad = 6;
    const textW = ctx.measureText(labelText).width;
    ctx.fillRect(x, Math.max(0, y - 28), textW + pad * 2, 28);

    ctx.fillStyle = "white";
    ctx.fillText(labelText, x + pad, Math.max(20, y - 8));
  }
}

// Convierte bbox a crop cuadrado con margen y lo dibuja a 224x224
function drawBboxCropToCanvas(bbox) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  let [x, y, w, h] = bbox;

  // margen
  const mx = w * BBOX_MARGIN;
  const my = h * BBOX_MARGIN;
  x = x - mx; y = y - my; w = w + 2 * mx; h = h + 2 * my;

  // a cuadrado
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

(async function main() {
  try {
    setError("");

    // info usuario (mínimo)
    const countKey = "platanometro_connected_count";
    const prev = Number(localStorage.getItem(countKey) || "0");
    const nowCount = prev + 1;
    localStorage.setItem(countKey, String(nowCount));
    if (userBoxEl) {
      userBoxEl.textContent = "Usuario: " + userId + " | Conexiones (este dispositivo): " + nowCount;
    }

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

    let lastDetTs = 0;
    let lastBbox = null;       // bbox en coords del vídeo
    let lastBananaScore = 0;

    let lastPerfectLog = 0;

    setInterval(async () => {
      if (busy) return;
      if (!video.videoWidth) return;

      busy = true;
      try {
        const now = Date.now();

        // 1) Detección cada DET_INTERVAL_MS
        if (now - lastDetTs > DET_INTERVAL_MS) {
          lastDetTs = now;

          const preds = await detector.detect(video);
          // busca la mejor predicción "banana"
          let best = null;
          for (const p of preds) {
            if (p.class === "banana" && p.score >= BANANA_MIN_SCORE) {
              if (!best || p.score > best.score) best = p;
            }
          }

          if (best) {
            lastBbox = best.bbox;       // [x,y,w,h]
            lastBananaScore = best.score;
          } else {
            lastBbox = null;
            lastBananaScore = 0;
          }
        }

        // 2) Overlay + mensaje
        if (!lastBbox) {
          drawOverlay(null, "");
          setStatus("No veo un plátano. Acércalo / céntralo / mejor luz.");
          setError("");
          return;
        } else {
          drawOverlay(lastBbox, "BANANA " + lastBananaScore.toFixed(2));
        }

        // 3) Clasificación ONNX sobre el crop del plátano
        const imgData = drawBboxCropToCanvas(lastBbox);
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

        setStatus(
          "Detectado: banana " + lastBananaScore.toFixed(2) +
          " | Estado: " + label +
          " | Conf: " + conf.toFixed(2) +
          " | Perfecto: " + verdict
        );

        // Log cuando detecta "perfecto" con alta confianza (cooldown 10s)
        if (label === PERFECT_CLASS && conf >= 0.75 && (now - lastPerfectLog) > 10000) {
          lastPerfectLog = now;
          await logUserEvent("detected_perfect", {
            conf: Number(conf.toFixed(4)),
            bananaScore: Number(lastBananaScore.toFixed(4))
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
