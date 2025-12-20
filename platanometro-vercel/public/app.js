// public/app.js
// @ts-nocheck
// Platanómetro (estático) - ONNX Runtime Web
// Requiere en producción:
//   /models/banana.onnx
//   /models/labels.json  -> ["pasado","perfecto","podrido","verde"]

// Config
const IMG = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

const MIN_CONF = 0.50;            // bájalo si te sale mucho "NO SEGURO"
const PERFECT_CLASS = "perfecto";

// Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");

// Softmax
function softmax(arr) {
  let max = -Infinity;
  for (const v of arr) max = Math.max(max, v);
  let sum = 0;
  const exps = arr.map(v => Math.exp(v - max));
  for (const e of exps) sum += e;
  return exps.map(e => e / sum);
}

function setStatus(text) {
  // Texto plano para evitar líos de comillas/backticks y warnings del editor
  statusEl.textContent = text;
}

function setError(msg) {
  errorEl.textContent = msg ? String(msg) : "";
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
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

async function loadModel() {
  // Para que ORT encuentre los WASM del CDN
  if (typeof ort !== "undefined" && ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  }
  return await ort.InferenceSession.create("/models/banana.onnx");
}

// Recorte central: toma un cuadrado del centro y lo escala a 224x224
function drawCenterCropToCanvas() {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const side = Math.min(vw, vh);

  const sx = Math.floor((vw - side) / 2);
  const sy = Math.floor((vh - side) / 2);

  ctx.drawImage(video, sx, sy, side, side, 0, 0, IMG, IMG);
  return ctx.getImageData(0, 0, IMG, IMG).data;
}

function preprocess(imgData) {
  // NCHW float32: [1, 3, 224, 224]
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

// Logging mínimo (opcional)
async function logUserEvent(event, extra = {}) {
  try {
    await fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, ...extra, ts: new Date().toISOString() })
    });
  } catch (_) {
    // silencioso
  }
}

(async function main() {
  try {
    setError("");

    setStatus("Cargando labels...");
    const labels = await loadLabels();

    setStatus("Cargando modelo...");
    const session = await loadModel();

    setStatus("Activando cámara...");
    await setupCamera();

    setStatus("Listo. Detectando...");
    await logUserEvent("connected", { ua: navigator.userAgent });

    let busy = false;
    let lastPerfectLog = 0;

    setInterval(async () => {
      if (busy) return;
      if (!video.videoWidth) return;

      busy = true;
      try {
        const imgData = drawCenterCropToCanvas();
        const input = preprocess(imgData);

        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = new ort.Tensor("float32", input, [1, 3, IMG, IMG]);

        const out = await session.run(feeds);

        const outputName = session.outputNames[0];
        const logits = out[outputName].data;

        const probs = softmax(Array.from(logits));

        let bestI = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[bestI]) bestI = i;
        }

        const label = labels[bestI] || ("class_" + bestI);
        const conf = probs[bestI];

        const verdict = verdictFrom(label, conf);

        setStatus(
          "Clase: " + label +
          " | Conf: " + conf.toFixed(2) +
          " | Perfecto: " + verdict
        );

        // Log cuando detecta "perfecto" con alta confianza (cooldown 10s)
        const now = Date.now();
        if (label === PERFECT_CLASS && conf >= 0.75 && (now - lastPerfectLog) > 10000) {
          lastPerfectLog = now;
          await logUserEvent("detected_perfect", { conf: Number(conf.toFixed(4)) });
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
