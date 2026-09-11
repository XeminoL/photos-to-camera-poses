import { EigenvalueDecomposition, Matrix } from '../vendor/ml-matrix.js';

import { projectionJacobian, rodrigues, matmul } from './geometry.js';
import { buildNormalEquations, solveSchur } from './schur.js';

export const SCALE_FREEDOM = 1;
export const RANK_TOLERANCE = 1e-8;
export const LAMBDA_INITIAL = 1e-3;
export const LAMBDA_GROWTH = 10;
export const LAMBDA_DECAY = 3;
export const LAMBDA_CEILING = 1e12;
export const LAMBDA_FLOOR = 1e-12;
export const BACKTRACK_LIMIT = 12;
export const DIAGONAL_FLOOR = 1e-12;

const cameraIndex = (poses, anchor) => {
  const index = new Map();
  [...poses.keys()].filter((c) => c !== anchor).sort((a, b) => a - b)
    .forEach((camera, position) => index.set(camera, position));
  return index;
};

function residual(recon, k) {
  const out = new Float64Array(2 * recon.observations.length);
  recon.observations.forEach((obs, n) => {
    const { r, t } = recon.poses.get(obs.camera);
    const point = recon.points[obs.point];
    const y = [
      r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + t[0],
      r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + t[1],
      r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + t[2],
    ];
    out[2 * n] = (k[0][0] * y[0] + k[0][1] * y[1] + k[0][2] * y[2]) / y[2] - obs.pixel[0];
    out[2 * n + 1] = (k[1][1] * y[1] + k[1][2] * y[2]) / y[2] - obs.pixel[1];
  });
  return out;
}

function jacobianBlocks(recon, k) {
  return recon.observations.map((obs) => {
    const { r, t } = recon.poses.get(obs.camera);
    const { dAngle, dTranslation, dPoint } = projectionJacobian(k, r, t, recon.points[obs.point]);
    return {
      dCamera: [
        [...dAngle[0], ...dTranslation[0]],
        [...dAngle[1], ...dTranslation[1]],
      ],
      dPoint,
    };
  });
}

function jacobian(recon, k, index) {
  const columns = index.size * 6 + 3 * recon.points.length;
  const rows = 2 * recon.observations.length;
  const j = Matrix.zeros(rows, columns);
  recon.observations.forEach((obs, n) => {
    const { r, t } = recon.poses.get(obs.camera);
    const { dAngle, dTranslation, dPoint } = projectionJacobian(k, r, t, recon.points[obs.point]);
    const pointBase = index.size * 6 + 3 * obs.point;
    const cameraBase = index.has(obs.camera) ? index.get(obs.camera) * 6 : -1;
    for (let row = 0; row < 2; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        if (cameraBase >= 0) {
          j.set(2 * n + row, cameraBase + col, dAngle[row][col]);
          j.set(2 * n + row, cameraBase + 3 + col, dTranslation[row][col]);
        }
        j.set(2 * n + row, pointBase + col, dPoint[row][col]);
      }
    }
  });
  return j;
}

function applyDelta(recon, index, delta) {
  const poses = new Map();
  for (const [camera, pose] of recon.poses) {
    if (!index.has(camera)) {
      poses.set(camera, pose);
      continue;
    }
    const base = index.get(camera) * 6;
    poses.set(camera, {
      r: matmul(rodrigues([delta[base], delta[base + 1], delta[base + 2]]), pose.r),
      t: pose.t.map((value, i) => value + delta[base + 3 + i]),
    });
  }
  const offset = index.size * 6;
  const points = recon.points.map((point, i) =>
    point.map((value, j) => value + delta[offset + 3 * i + j]));
  return { poses, points, observations: recon.observations, anchor: recon.anchor };
}

function nullSpaceDimension(j) {
  const scaled = j.clone();
  for (let col = 0; col < scaled.columns; col += 1) {
    const norm = Math.max(Math.hypot(...scaled.getColumn(col)), DIAGONAL_FLOOR);
    scaled.mulColumn(col, 1 / norm);
  }
  const eig = new EigenvalueDecomposition(scaled.transpose().mmul(scaled), { assumeSymmetric: true });
  const values = [...eig.realEigenvalues].sort((a, b) => a - b);
  const largest = values[values.length - 1];
  return Math.max(values.filter((v) => v < RANK_TOLERANCE * largest).length, SCALE_FREEDOM);
}

function covarianceDiagonal(h, sumSquares, nullDim, residualCount) {
  const eig = new EigenvalueDecomposition(h, { assumeSymmetric: true });
  const order = eig.realEigenvalues
    .map((value, i) => ({ value, i }))
    .sort((a, b) => a.value - b.value);
  const dof = Math.max(residualCount - (h.rows - nullDim), 1);
  const sigmaSquared = sumSquares / dof;

  const diagonal = new Float64Array(h.rows);
  order.forEach((entry, rank) => {
    if (rank < nullDim) return;
    const column = eig.eigenvectorMatrix.getColumn(entry.i);
    const weight = sigmaSquared / entry.value;
    for (let row = 0; row < diagonal.length; row += 1) {
      diagonal[row] += weight * column[row] * column[row];
    }
  });
  return diagonal;
}

export function optimize(recon, k, iterations) {
  const index = cameraIndex(recon.poses, recon.anchor);
  let current = recon;
  let r = residual(current, k);
  let sumSquares = r.reduce((sum, v) => sum + v * v, 0);
  let lambda = LAMBDA_INITIAL;

  for (let step = 0; step < iterations; step += 1) {
    const blocks = jacobianBlocks(current, k);
    const paired = current.observations.map((_, n) => [r[2 * n], r[2 * n + 1]]);
    const system = buildNormalEquations(
      current.observations, blocks, paired, index, current.points.length,
    );
    let improved = false;

    for (let attempt = 0; attempt < BACKTRACK_LIMIT; attempt += 1) {
      const update = solveSchur(system, lambda);
      if (!update) {
        lambda *= LAMBDA_GROWTH;
        continue;
      }
      const delta = new Float64Array(index.size * 6 + current.points.length * 3);
      delta.set(update.cameraDelta, 0);
      delta.set(update.pointDelta, index.size * 6);
      const candidate = applyDelta(current, index, delta);
      const trial = residual(candidate, k);
      const trialSum = trial.reduce((sum, v) => sum + v * v, 0);
      if (trialSum < sumSquares) {
        current = candidate;
        r = trial;
        sumSquares = trialSum;
        lambda = Math.max(lambda / LAMBDA_DECAY, LAMBDA_FLOOR);
        improved = true;
        break;
      }
      lambda *= LAMBDA_GROWTH;
    }
    if (!improved || lambda > LAMBDA_CEILING) break;
  }

  const j = jacobian(current, k, index);
  const h = j.transpose().mmul(j);
  const nullDim = nullSpaceDimension(j);
  const diagonal = covarianceDiagonal(h, sumSquares, nullDim, r.length);

  const angleSigma = new Map();
  const centerSigma = new Map();
  for (const [camera, position] of index) {
    const base = position * 6;
    const angle = diagonal[base] + diagonal[base + 1] + diagonal[base + 2];
    const center = diagonal[base + 3] + diagonal[base + 4] + diagonal[base + 5];
    angleSigma.set(camera, (Math.sqrt(Math.max(angle, 0)) * 180) / Math.PI);
    centerSigma.set(camera, Math.sqrt(Math.max(center, 0)));
  }

  return {
    reconstruction: current,
    report: {
      rms: Math.sqrt(sumSquares / current.observations.length),
      nullSpaceDimension: nullDim,
      excessNullDimensions: nullDim - SCALE_FREEDOM,
      isDegenerate: nullDim > SCALE_FREEDOM,
      angleSigma,
      centerSigma,
    },
  };
}