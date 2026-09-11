import { createScene, drawScene } from './viewer.js';

const MAX_DIMENSION = 640;
const THUMB_WIDTH = 52;
const THUMB_HEIGHT = 39;
const MIN_IMAGES = 3;
const ZOOM_STEP = 1.12;
const ZOOM_RANGE = [0.3, 6];
const dom = {};
for (const id of ['drop', 'files', 'thumbs', 'run', 'reset', 'count', 'progress-section',
  'stage', 'stage-count', 'track', 'fill', 'summary-section', 'summary', 'warning',
  'camera-section', 'cameras', 'skipped', 'scene', 'caption', 'legend', 'fast', 'focal']) {
  dom[id.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = document.getElementById(id);
}

const state = {
  images: [],
  names: [],
  scene: null,
  view: { yaw: 0.6, pitch: -0.35, zoom: 1 },
  worker: null,
};

function fail(message) {
  dom.progressSection.hidden = true;
  dom.summarySection.hidden = false;
  dom.summary.innerHTML = '';
  dom.warning.innerHTML = `<div class="note bad">${message}</div>`;
  dom.run.disabled = state.images.length < MIN_IMAGES;
}

function makeCanvas(width, height) {
  if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function loadBitmap(file) {
  if (typeof createImageBitmap === 'function') return createImageBitmap(file);
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const element = new Image();
    element.onload = () => {
      URL.revokeObjectURL(url);
      resolve(element);
    };
    element.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`cannot read ${file.name}`));
    };
    element.src = url;
  });
}

async function toGrayscale(file) {
  const bitmap = await loadBitmap(file);
  const sourceWidth = bitmap.width ?? bitmap.naturalWidth;
  const sourceHeight = bitmap.height ?? bitmap.naturalHeight;
  const scale = Math.min(1, MAX_DIMENSION / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = makeCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  const grey = new Float32Array(width * height);
  for (let i = 0; i < grey.length; i += 1) {
    const p = i * 4;
    grey[i] = (0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]) / 255;
  }
  bitmap.close?.();
  return { width, height, data: grey };
}

function drawThumb(image) {
  const canvas = document.createElement('canvas');
  canvas.width = THUMB_WIDTH;
  canvas.height = THUMB_HEIGHT;
  const context = canvas.getContext('2d');
  const target = context.createImageData(THUMB_WIDTH, THUMB_HEIGHT);
  for (let y = 0; y < THUMB_HEIGHT; y += 1) {
    for (let x = 0; x < THUMB_WIDTH; x += 1) {
      const sx = Math.floor((x / THUMB_WIDTH) * image.width);
      const sy = Math.floor((y / THUMB_HEIGHT) * image.height);
      const value = Math.round(image.data[sy * image.width + sx] * 255);
      const p = (y * THUMB_WIDTH + x) * 4;
      target.data[p] = value;
      target.data[p + 1] = value;
      target.data[p + 2] = value;
      target.data[p + 3] = 255;
    }
  }
  context.putImageData(target, 0, 0);
  dom.thumbs.append(canvas);
}

async function addFiles(fileList) {
  const files = [...fileList].filter((file) => file.type.startsWith('image/'));
  if (files.length === 0) return;
  dom.count.textContent = 'reading...';
  try {
    for (const file of files) {
      const image = await toGrayscale(file);
      state.images.push(image);
      state.names.push(file.name);
      drawThumb(image);
      dom.count.textContent = `${state.images.length} photos`;
    }
  } catch (error) {
    fail(error.message);
    return;
  }
  const { width, height } = state.images[0];
  dom.count.textContent = `${state.images.length} photos, ${width}x${height}`;
  dom.run.disabled = state.images.length < MIN_IMAGES;
  dom.caption.textContent = state.images.length < MIN_IMAGES
    ? `${MIN_IMAGES - state.images.length} more needed`
    : '';
}

function resizeCanvas() {
  const rect = dom.scene.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  dom.scene.width = Math.max(1, Math.round(rect.width * ratio));
  dom.scene.height = Math.max(1, Math.round(rect.height * ratio));
  drawScene(dom.scene, state.scene, state.view);
}

function showProgress(message) {
  dom.progressSection.hidden = false;
  dom.track.classList.toggle('busy', message.stage === 'reconstruct');
  if (message.stage === 'detect') {
    dom.stage.textContent = 'Detecting features';
    dom.stageCount.textContent = `${message.done}/${message.total} photos`;
    dom.fill.style.width = `${(message.done / message.total) * 100}%`;
  } else if (message.stage === 'match') {
    dom.stage.textContent = 'Matching features';
    dom.stageCount.textContent = `${message.done}/${message.total} pairs, ${message.kept} kept`;
    dom.fill.style.width = `${(message.done / message.total) * 100}%`;
  } else if (message.stage === 'seed') {
    dom.stage.textContent = 'Choosing a seed pair';
    dom.stageCount.textContent = 'testing each candidate';
  } else {
    dom.stage.textContent = 'Reconstructing';
    dom.stageCount.textContent = `from photo ${message.seed[0] + 1} and ${message.seed[1] + 1}`;
  }
}

const row = (label, value) =>
  `<div class="stat"><span>${label}</span><span>${value}</span></div>`;

function showResult(message) {
  dom.progressSection.hidden = true;
  dom.summarySection.hidden = false;
  dom.cameraSection.hidden = false;
  dom.legend.hidden = false;

  const counts = message.featureCounts;
  dom.summary.innerHTML = [
    row('Cameras placed', `${message.poses.length} / ${state.images.length}`),
    row('Points', message.points.length),
    row('Reprojection error', `${message.rms.toFixed(3)} px`),
    row('Features per photo', `${Math.min(...counts)}–${Math.max(...counts)}`),
    row('Usable pairs', `${message.pairsKept} / ${message.pairsTotal}`),
    row('Time', `${message.seconds.toFixed(1)} s`),
    row('Focal length', `${message.focal.toFixed(0)} px${message.focalGuessed ? ' (guessed)' : ''}`),
  ].join('');

  dom.warning.innerHTML = message.isDegenerate
    ? `<div class="note bad">Underconstrained: ${message.nullSpaceDimension} directions are
       undetermined instead of 1. The numbers below cannot be trusted.</div>`
    : '';

  const sorted = [...message.poses].sort((a, b) => a.camera - b.camera);
  dom.cameras.innerHTML = sorted.map((pose) => {
    const name = state.names[pose.camera] ?? `photo ${pose.camera + 1}`;
    if (pose.camera === message.anchor) {
      return `<tr class="anchor"><td>${name}</td><td>anchor</td><td>anchor</td></tr>`;
    }
    return `<tr><td>${name}</td><td>${pose.angleSigma.toFixed(3)}°</td>`
      + `<td>${pose.centreSigma.toFixed(4)}</td></tr>`;
  }).join('');

  dom.skipped.innerHTML = message.rejected.length
    ? `<div>Skipped ${message.rejected.length}:</div>`
      + message.rejected.slice(0, 5).map((item) => {
        const name = state.names[item.camera] ?? `photo ${item.camera + 1}`;
        return `<div>${name}, ${item.reason}</div>`;
      }).join('')
    : '';

  state.scene = createScene(message.points, sorted);
  state.view = { yaw: 0.6, pitch: -0.35, zoom: 1 };
  drawScene(dom.scene, state.scene, state.view);
  dom.caption.textContent = '';
  dom.run.disabled = false;
}

function start() {
  dom.run.disabled = true;
  dom.summarySection.hidden = true;
  dom.cameraSection.hidden = true;
  dom.progressSection.hidden = false;
  dom.stage.textContent = 'Starting';
  dom.stageCount.textContent = '';
  dom.fill.style.width = '0%';

  state.worker?.terminate();
  state.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  state.worker.onerror = (event) => fail(`worker error: ${event.message}`);
  state.worker.onmessage = (event) => {
    const message = event.data;
    if (message.type === 'progress') showProgress(message);
    else if (message.type === 'done') showResult(message);
    else fail(`Reconstruction failed: ${message.message}`);
  };
  const focal = Number(dom.focal.value);
  state.worker.postMessage({
    images: state.images,
    fast: dom.fast.checked,
    focal: Number.isFinite(focal) && focal > 0 ? focal : null,
  });
}

dom.files.addEventListener('change', (event) => {
  addFiles(event.target.files);
});
dom.drop.addEventListener('dragover', (event) => {
  event.preventDefault();
  dom.drop.classList.add('over');
});
dom.drop.addEventListener('dragleave', () => dom.drop.classList.remove('over'));
dom.drop.addEventListener('drop', (event) => {
  event.preventDefault();
  dom.drop.classList.remove('over');
  addFiles(event.dataTransfer.files);
});
dom.run.addEventListener('click', start);
dom.reset.addEventListener('click', () => {
  state.worker?.terminate();
  state.images = [];
  state.names = [];
  state.scene = null;
  dom.thumbs.innerHTML = '';
  dom.files.value = '';
  dom.count.textContent = '';
  dom.run.disabled = true;
  dom.progressSection.hidden = true;
  dom.summarySection.hidden = true;
  dom.cameraSection.hidden = true;
  dom.legend.hidden = true;
  dom.caption.textContent = '';
  drawScene(dom.scene, null, state.view);
});

let dragging = null;
dom.scene.addEventListener('pointerdown', (event) => {
  dragging = { x: event.clientX, y: event.clientY };
  dom.scene.setPointerCapture(event.pointerId);
});
dom.scene.addEventListener('pointermove', (event) => {
  if (!dragging) return;
  state.view.yaw += (event.clientX - dragging.x) * 0.008;
  state.view.pitch += (event.clientY - dragging.y) * 0.008;
  dragging = { x: event.clientX, y: event.clientY };
  drawScene(dom.scene, state.scene, state.view);
});
dom.scene.addEventListener('pointerup', () => { dragging = null; });
dom.scene.addEventListener('wheel', (event) => {
  event.preventDefault();
  const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
  state.view.zoom = Math.min(ZOOM_RANGE[1], Math.max(ZOOM_RANGE[0], state.view.zoom * factor));
  drawScene(dom.scene, state.scene, state.view);
}, { passive: false });

window.addEventListener('resize', resizeCanvas);
window.addEventListener('error', (event) => fail(`Error: ${event.message}`));
resizeCanvas();