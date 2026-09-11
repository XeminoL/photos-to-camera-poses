import { Matrix, solve } from '../vendor/ml-matrix.js';

export const CAMERA_PARAMETERS = 6;
export const POINT_PARAMETERS = 3;
export const BLOCK_FLOOR = 1e-12;

function invert3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(det) || Math.abs(det) < BLOCK_FLOOR) return null;
  return [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
}

export function buildNormalEquations(observations, jacobians, residuals, cameraIndex, pointCount) {
  const cameras = cameraIndex.size;
  const width = cameras * CAMERA_PARAMETERS;
  const cameraBlock = Matrix.zeros(width, width);
  const cameraGradient = new Float64Array(width);
  const pointBlocks = Array.from({ length: pointCount }, () => new Float64Array(9));
  const pointGradients = Array.from({ length: pointCount }, () => new Float64Array(3));
  const crossBlocks = new Map();

  observations.forEach((obs, n) => {
    const { dCamera, dPoint } = jacobians[n];
    const residual = residuals[n];
    const slot = cameraIndex.get(obs.camera);
    const base = slot === undefined ? -1 : slot * CAMERA_PARAMETERS;

    for (let a = 0; a < POINT_PARAMETERS; a += 1) {
      for (let b = 0; b < POINT_PARAMETERS; b += 1) {
        pointBlocks[obs.point][a * 3 + b] +=
          dPoint[0][a] * dPoint[0][b] + dPoint[1][a] * dPoint[1][b];
      }
      pointGradients[obs.point][a] += dPoint[0][a] * residual[0] + dPoint[1][a] * residual[1];
    }
    if (base < 0) return;

    for (let a = 0; a < CAMERA_PARAMETERS; a += 1) {
      for (let b = 0; b < CAMERA_PARAMETERS; b += 1) {
        cameraBlock.set(base + a, base + b, cameraBlock.get(base + a, base + b)
          + dCamera[0][a] * dCamera[0][b] + dCamera[1][a] * dCamera[1][b]);
      }
      cameraGradient[base + a] += dCamera[0][a] * residual[0] + dCamera[1][a] * residual[1];
    }

    const key = `${slot}:${obs.point}`;
    let cross = crossBlocks.get(key);
    if (!cross) {
      cross = { slot, point: obs.point, values: new Float64Array(CAMERA_PARAMETERS * 3) };
      crossBlocks.set(key, cross);
    }
    for (let a = 0; a < CAMERA_PARAMETERS; a += 1) {
      for (let b = 0; b < POINT_PARAMETERS; b += 1) {
        cross.values[a * 3 + b] += dCamera[0][a] * dPoint[0][b] + dCamera[1][a] * dPoint[1][b];
      }
    }
  });

  return { cameraBlock, cameraGradient, pointBlocks, pointGradients, crossBlocks };
}

export function solveSchur(system, lambda) {
  const { cameraBlock, cameraGradient, pointBlocks, pointGradients, crossBlocks } = system;
  const width = cameraBlock.rows;
  const reduced = cameraBlock.clone();
  const reducedGradient = Float64Array.from(cameraGradient);

  for (let i = 0; i < width; i += 1) {
    reduced.set(i, i, reduced.get(i, i) * (1 + lambda) + lambda * BLOCK_FLOOR);
  }

  const byPoint = new Map();
  for (const cross of crossBlocks.values()) {
    if (!byPoint.has(cross.point)) byPoint.set(cross.point, []);
    byPoint.get(cross.point).push(cross);
  }

  const inverses = new Map();
  for (const [point, group] of byPoint) {
    const block = Float64Array.from(pointBlocks[point]);
    for (let i = 0; i < 3; i += 1) {
      block[i * 3 + i] = block[i * 3 + i] * (1 + lambda) + lambda * BLOCK_FLOOR;
    }
    const inverse = invert3(block);
    if (!inverse) continue;
    inverses.set(point, inverse);

    const weighted = pointGradients[point];
    const scaled = [
      inverse[0] * weighted[0] + inverse[1] * weighted[1] + inverse[2] * weighted[2],
      inverse[3] * weighted[0] + inverse[4] * weighted[1] + inverse[5] * weighted[2],
      inverse[6] * weighted[0] + inverse[7] * weighted[1] + inverse[8] * weighted[2],
    ];

    for (const cross of group) {
      const base = cross.slot * CAMERA_PARAMETERS;
      for (let a = 0; a < CAMERA_PARAMETERS; a += 1) {
        reducedGradient[base + a] -= cross.values[a * 3] * scaled[0]
          + cross.values[a * 3 + 1] * scaled[1]
          + cross.values[a * 3 + 2] * scaled[2];
      }
      for (const other of group) {
        const otherBase = other.slot * CAMERA_PARAMETERS;
        for (let a = 0; a < CAMERA_PARAMETERS; a += 1) {
          for (let b = 0; b < CAMERA_PARAMETERS; b += 1) {
            let sum = 0;
            for (let p = 0; p < 3; p += 1) {
              for (let q = 0; q < 3; q += 1) {
                sum += cross.values[a * 3 + p] * inverse[p * 3 + q] * other.values[b * 3 + q];
              }
            }
            reduced.set(base + a, otherBase + b, reduced.get(base + a, otherBase + b) - sum);
          }
        }
      }
    }
  }

  let cameraDelta;
  try {
    cameraDelta = solve(reduced, Matrix.columnVector([...reducedGradient]).mul(-1)).getColumn(0);
  } catch {
    return null;
  }
  if (cameraDelta.some((value) => !Number.isFinite(value))) return null;

  const pointDelta = new Float64Array(pointBlocks.length * 3);
  for (const [point, inverse] of inverses) {
    const rhs = [-pointGradients[point][0], -pointGradients[point][1], -pointGradients[point][2]];
    for (const cross of byPoint.get(point)) {
      const base = cross.slot * CAMERA_PARAMETERS;
      for (let b = 0; b < 3; b += 1) {
        let sum = 0;
        for (let a = 0; a < CAMERA_PARAMETERS; a += 1) {
          sum += cross.values[a * 3 + b] * cameraDelta[base + a];
        }
        rhs[b] -= sum;
      }
    }
    for (let i = 0; i < 3; i += 1) {
      pointDelta[point * 3 + i] =
        inverse[i * 3] * rhs[0] + inverse[i * 3 + 1] * rhs[1] + inverse[i * 3 + 2] * rhs[2];
    }
  }

  return { cameraDelta, pointDelta };
}