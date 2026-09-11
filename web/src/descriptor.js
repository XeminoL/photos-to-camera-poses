import { SCALES_PER_OCTAVE } from './pyramid.js';

export const ORIENTATION_BINS = 36;
export const ORIENTATION_RADIUS_FACTOR = 3;
export const ORIENTATION_SIGMA_FACTOR = 1.5;
export const ORIENTATION_PEAK_RATIO = 0.8;
export const SMOOTHING_PASSES = 6;

export const DESCRIPTOR_CELLS = 4;
export const DESCRIPTOR_BINS = 8;
export const DESCRIPTOR_LENGTH = DESCRIPTOR_CELLS * DESCRIPTOR_CELLS * DESCRIPTOR_BINS;
export const DESCRIPTOR_SCALE_FACTOR = 3;
export const DESCRIPTOR_CLIP = 0.2;
export const DESCRIPTOR_OUTPUT_SCALE = 512;
export const DESCRIPTOR_MAX = 255;
export const NORM_FLOOR = 1.1920929e-7;

const TWO_PI = Math.PI * 2;
const DEGREES = 180 / Math.PI;

function gradientAt(layer, x, y) {
  const { width, height, data } = layer;
  if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return null;
  const dx = data[y * width + x + 1] - data[y * width + x - 1];
  const dy = data[(y - 1) * width + x] - data[(y + 1) * width + x];
  return { magnitude: Math.hypot(dx, dy), angle: Math.atan2(dy, dx) };
}

function smoothCircular(histogram, passes) {
  const size = histogram.length;
  let current = histogram;
  for (let pass = 0; pass < passes; pass += 1) {
    const next = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      next[i] = (current[(i - 1 + size) % size] + current[i] + current[(i + 1) % size]) / 3;
    }
    current = next;
  }
  return current;
}

export function computeOrientations(layer, x, y, scale) {
  const histogram = new Float64Array(ORIENTATION_BINS);
  const radius = Math.round(ORIENTATION_RADIUS_FACTOR * ORIENTATION_SIGMA_FACTOR * scale);
  const weightSigma = ORIENTATION_SIGMA_FACTOR * scale;
  const denominator = 2 * weightSigma * weightSigma;
  const cx = Math.round(x);
  const cy = Math.round(y);

  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      const sample = gradientAt(layer, cx + dx, cy + dy);
      if (!sample) continue;
      const weight = Math.exp(-(dx * dx + dy * dy) / denominator);
      let bin = Math.round((ORIENTATION_BINS * sample.angle) / TWO_PI);
      bin = ((bin % ORIENTATION_BINS) + ORIENTATION_BINS) % ORIENTATION_BINS;
      histogram[bin] += weight * sample.magnitude;
    }
  }

  const smoothed = smoothCircular(histogram, SMOOTHING_PASSES);
  const peak = Math.max(...smoothed);
  if (peak <= 0) return [];

  const angles = [];
  for (let i = 0; i < ORIENTATION_BINS; i += 1) {
    const left = smoothed[(i - 1 + ORIENTATION_BINS) % ORIENTATION_BINS];
    const right = smoothed[(i + 1) % ORIENTATION_BINS];
    if (smoothed[i] <= left || smoothed[i] < right) continue;
    if (smoothed[i] < ORIENTATION_PEAK_RATIO * peak) continue;
    const denom = left - 2 * smoothed[i] + right;
    const interpolated = denom === 0 ? i : i + (0.5 * (left - right)) / denom;
    const wrapped = ((interpolated % ORIENTATION_BINS) + ORIENTATION_BINS) % ORIENTATION_BINS;
    angles.push((360 - (wrapped * 360) / ORIENTATION_BINS) % 360);
  }
  return angles;
}

export function computeDescriptor(layer, x, y, scale, orientation) {
  const d = DESCRIPTOR_CELLS;
  const n = DESCRIPTOR_BINS;
  const angle = 360 - orientation;
  const radians = angle / DEGREES;
  const histWidth = DESCRIPTOR_SCALE_FACTOR * scale;
  const radius = Math.min(
    Math.round(histWidth * Math.SQRT2 * (d + 1) * 0.5),
    Math.round(Math.hypot(layer.width, layer.height)),
  );
  const cos = Math.cos(radians) / histWidth;
  const sin = Math.sin(radians) / histWidth;
  const binsPerDegree = n / 360;
  const expScale = -1 / (d * d * 0.5);

  const stride = n + 2;
  const rowStride = (d + 2) * stride;
  const histogram = new Float64Array((d + 2) * rowStride);
  const cx = Math.round(x);
  const cy = Math.round(y);

  for (let i = -radius; i <= radius; i += 1) {
    for (let j = -radius; j <= radius; j += 1) {
      const cRot = j * cos - i * sin;
      const rRot = j * sin + i * cos;
      const rbin = rRot + Math.floor(d / 2) - 0.5;
      const cbin = cRot + Math.floor(d / 2) - 0.5;
      if (rbin <= -1 || rbin >= d || cbin <= -1 || cbin >= d) continue;

      const sample = gradientAt(layer, cx + j, cy + i);
      if (!sample) continue;

      let degrees = sample.angle * DEGREES;
      if (degrees < 0) degrees += 360;
      const weight = Math.exp((cRot * cRot + rRot * rRot) * expScale);
      const value = sample.magnitude * weight;

      let obin = (degrees - angle) * binsPerDegree;
      const r0 = Math.floor(rbin);
      const c0 = Math.floor(cbin);
      let o0 = Math.floor(obin);
      const rf = rbin - r0;
      const cf = cbin - c0;
      const of = obin - o0;
      o0 = ((o0 % n) + n) % n;

      const v1 = value * rf;
      const v0 = value - v1;
      const v11 = v1 * cf;
      const v10 = v1 - v11;
      const v01 = v0 * cf;
      const v00 = v0 - v01;
      const v111 = v11 * of;
      const v110 = v11 - v111;
      const v101 = v10 * of;
      const v100 = v10 - v101;
      const v011 = v01 * of;
      const v010 = v01 - v011;
      const v001 = v00 * of;
      const v000 = v00 - v001;

      const base = (r0 + 1) * rowStride + (c0 + 1) * stride + o0;
      histogram[base] += v000;
      histogram[base + 1] += v001;
      histogram[base + stride] += v010;
      histogram[base + stride + 1] += v011;
      histogram[base + rowStride] += v100;
      histogram[base + rowStride + 1] += v101;
      histogram[base + rowStride + stride] += v110;
      histogram[base + rowStride + stride + 1] += v111;
    }
  }

  const raw = new Float64Array(DESCRIPTOR_LENGTH);
  for (let i = 0; i < d; i += 1) {
    for (let j = 0; j < d; j += 1) {
      const base = (i + 1) * rowStride + (j + 1) * stride;
      histogram[base] += histogram[base + n];
      histogram[base + 1] += histogram[base + n + 1];
      for (let k = 0; k < n; k += 1) {
        raw[(i * d + j) * n + k] = histogram[base + k];
      }
    }
  }

  let squared = raw.reduce((sum, value) => sum + value * value, 0);
  const threshold = Math.sqrt(squared) * DESCRIPTOR_CLIP;
  squared = 0;
  for (let i = 0; i < raw.length; i += 1) {
    raw[i] = Math.min(raw[i], threshold);
    squared += raw[i] * raw[i];
  }
  const factor = DESCRIPTOR_OUTPUT_SCALE / Math.max(Math.sqrt(squared), NORM_FLOOR);

  const descriptor = new Float32Array(DESCRIPTOR_LENGTH);
  for (let i = 0; i < raw.length; i += 1) {
    descriptor[i] = Math.max(0, Math.min(DESCRIPTOR_MAX, Math.round(raw[i] * factor)));
  }
  return descriptor;
}

export function describeKeypoints(pyramid, keypoints, scales = SCALES_PER_OCTAVE) {
  const described = [];
  for (const keypoint of keypoints) {
    const layers = pyramid[keypoint.octave];
    if (!layers) continue;
    const index = Math.max(0, Math.min(layers.length - 1, Math.round(keypoint.layer)));
    const layer = layers[index];
    const step = 2 ** (keypoint.octave - 1);
    const localX = keypoint.x / step;
    const localY = keypoint.y / step;
    const localScale = keypoint.sigma / step;

    for (const angle of computeOrientations(layer, localX, localY, localScale)) {
      described.push({
        ...keypoint,
        angle,
        descriptor: computeDescriptor(layer, localX, localY, localScale, angle),
      });
    }
  }
  return described;
}