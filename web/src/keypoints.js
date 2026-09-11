import { SCALES_PER_OCTAVE, INITIAL_SIGMA } from './pyramid.js';

export const CONTRAST_THRESHOLD = 0.04;
export const EDGE_THRESHOLD = 10;
export const BORDER = 5;
export const MAX_REFINEMENT_STEPS = 5;
export const OFFSET_LIMIT = 0.5;

const at = (layer, x, y) => layer.data[y * layer.width + x];

function isExtremum(below, current, above, x, y, threshold) {
  const value = at(current, x, y);
  if (Math.abs(value) <= threshold) return false;
  const positive = value > 0;
  for (const layer of [below, current, above]) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (layer === current && dx === 0 && dy === 0) continue;
        const neighbour = at(layer, x + dx, y + dy);
        if (positive ? neighbour >= value : neighbour <= value) return false;
      }
    }
  }
  return true;
}

function gradient(below, current, above, x, y) {
  return [
    (at(current, x + 1, y) - at(current, x - 1, y)) / 2,
    (at(current, x, y + 1) - at(current, x, y - 1)) / 2,
    (at(above, x, y) - at(below, x, y)) / 2,
  ];
}

function hessian(below, current, above, x, y) {
  const centre = at(current, x, y);
  const dxx = at(current, x + 1, y) + at(current, x - 1, y) - 2 * centre;
  const dyy = at(current, x, y + 1) + at(current, x, y - 1) - 2 * centre;
  const dss = at(above, x, y) + at(below, x, y) - 2 * centre;
  const dxy = (at(current, x + 1, y + 1) - at(current, x - 1, y + 1)
    - at(current, x + 1, y - 1) + at(current, x - 1, y - 1)) / 4;
  const dxs = (at(above, x + 1, y) - at(above, x - 1, y)
    - at(below, x + 1, y) + at(below, x - 1, y)) / 4;
  const dys = (at(above, x, y + 1) - at(above, x, y - 1)
    - at(below, x, y + 1) + at(below, x, y - 1)) / 4;
  return [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];
}

function solve3(a, b) {
  const [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = a;
  const det = a00 * (a11 * a22 - a12 * a21)
    - a01 * (a10 * a22 - a12 * a20)
    + a02 * (a10 * a21 - a11 * a20);
  if (Math.abs(det) < 1e-16) return null;
  const inv = [
    [(a11 * a22 - a12 * a21) / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det],
    [(a12 * a20 - a10 * a22) / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det],
    [(a10 * a21 - a11 * a20) / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det],
  ];
  return inv.map((row) => -(row[0] * b[0] + row[1] * b[1] + row[2] * b[2]));
}

function onEdge(h, threshold) {
  const trace = h[0][0] + h[1][1];
  const determinant = h[0][0] * h[1][1] - h[0][1] * h[0][1];
  if (determinant <= 0) return true;
  const limit = ((threshold + 1) ** 2) / threshold;
  return (trace * trace) / determinant >= limit;
}

function refine(octaveLayers, layerIndex, startX, startY, scales) {
  let x = startX;
  let y = startY;
  let index = layerIndex;
  let offset = null;
  let h = null;

  for (let step = 0; step < MAX_REFINEMENT_STEPS; step += 1) {
    const current = octaveLayers[index];
    const below = octaveLayers[index - 1];
    const above = octaveLayers[index + 1];
    const g = gradient(below, current, above, x, y);
    h = hessian(below, current, above, x, y);
    offset = solve3(h, g);
    if (!offset) return null;
    if (Math.abs(offset[0]) < OFFSET_LIMIT
      && Math.abs(offset[1]) < OFFSET_LIMIT
      && Math.abs(offset[2]) < OFFSET_LIMIT) {
      const value = at(current, x, y) + 0.5 * (g[0] * offset[0] + g[1] * offset[1] + g[2] * offset[2]);
      if (Math.abs(value) * scales < CONTRAST_THRESHOLD) return null;
      if (onEdge(h, EDGE_THRESHOLD)) return null;
      return { x: x + offset[0], y: y + offset[1], layer: index + offset[2], value };
    }
    x += Math.round(offset[0]);
    y += Math.round(offset[1]);
    index += Math.round(offset[2]);
    if (index < 1 || index > octaveLayers.length - 2) return null;
    const layer = octaveLayers[index];
    if (x < BORDER || y < BORDER || x >= layer.width - BORDER || y >= layer.height - BORDER) {
      return null;
    }
  }
  return null;
}

export function findKeypoints(dog, scales = SCALES_PER_OCTAVE) {
  const threshold = 0.5 * CONTRAST_THRESHOLD / scales;
  const keypoints = [];

  dog.forEach((layers, octave) => {
    for (let index = 1; index < layers.length - 1; index += 1) {
      const current = layers[index];
      const below = layers[index - 1];
      const above = layers[index + 1];
      for (let y = BORDER; y < current.height - BORDER; y += 1) {
        for (let x = BORDER; x < current.width - BORDER; x += 1) {
          if (!isExtremum(below, current, above, x, y, threshold)) continue;
          const refined = refine(layers, index, x, y, scales);
          if (!refined) continue;
          const step = 2 ** (octave - 1);
          keypoints.push({
            x: refined.x * step,
            y: refined.y * step,
            octave,
            layer: refined.layer,
            sigma: INITIAL_SIGMA * 2 ** (refined.layer / scales) * step,
            response: Math.abs(refined.value),
          });
        }
      }
    }
  });
  return keypoints;
}