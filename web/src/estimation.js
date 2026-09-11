import { Matrix, SingularValueDecomposition, determinant, inverse } from '../vendor/ml-matrix.js';

import { apply, identity, matmul, normalizePixel, skew, transpose } from './geometry.js';

export const SAMPSON_FLOOR = 1e-30;
export const MIN_POINTS_FUNDAMENTAL = 8;
export const MIN_POINTS_PNP = 6;
export const PNP_SAMPLE_SIZE = 10;

const nullVector = (rows) => {
  const width = rows[0].length;
  const padded = rows.length >= width
    ? rows
    : [...rows, ...Array.from({ length: width - rows.length }, () => new Array(width).fill(0))];
  const svd = new SingularValueDecomposition(new Matrix(padded));
  return svd.rightSingularVectors.getColumn(width - 1);
};

function isotropicNormalizer(points, targetScale) {
  const dim = points[0].length;
  const center = points[0].map((_, j) => points.reduce((sum, p) => sum + p[j], 0) / points.length);
  const spread = points.reduce((sum, p) =>
    sum + Math.hypot(...p.map((v, j) => v - center[j])), 0) / points.length;
  const s = targetScale / spread;
  const t = Array.from({ length: dim + 1 }, (_, i) =>
    Array.from({ length: dim + 1 }, (_, j) => (i === j ? 1 : 0)));
  for (let i = 0; i < dim; i += 1) {
    t[i][i] = s;
    t[i][dim] = -s * center[i];
  }
  return t;
}

const homogeneous = (p) => [...p, 1];

export function fundamentalFromPoints(left, right) {
  if (left.length < MIN_POINTS_FUNDAMENTAL) {
    throw new Error(`need ${MIN_POINTS_FUNDAMENTAL} points, got ${left.length}`);
  }
  const t1 = isotropicNormalizer(left, Math.SQRT2);
  const t2 = isotropicNormalizer(right, Math.SQRT2);
  const rows = left.map((point, i) => {
    const a = apply(t1, homogeneous(point));
    const b = apply(t2, homogeneous(right[i]));
    return [b[0] * a[0], b[0] * a[1], b[0], b[1] * a[0], b[1] * a[1], b[1], a[0], a[1], 1];
  });
  const f = nullVector(rows);
  const svd = new SingularValueDecomposition(new Matrix([f.slice(0, 3), f.slice(3, 6), f.slice(6, 9)]));
  const rank2 = svd.leftSingularVectors
    .mmul(Matrix.diag([svd.diagonal[0], svd.diagonal[1], 0]))
    .mmul(svd.rightSingularVectors.transpose())
    .to2DArray();
  return matmul(matmul(transpose(t2), rank2), t1);
}

export function sampsonDistance(f, left, right) {
  return left.map((point, i) => {
    const a = homogeneous(point);
    const b = homogeneous(right[i]);
    const fa = apply(f, a);
    const ftb = apply(transpose(f), b);
    const numerator = (b[0] * fa[0] + b[1] * fa[1] + b[2] * fa[2]) ** 2;
    const denominator = fa[0] ** 2 + fa[1] ** 2 + ftb[0] ** 2 + ftb[1] ** 2;
    return Math.sqrt(numerator / Math.max(denominator, SAMPSON_FLOOR));
  });
}

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function sampleIndices(count, size, random) {
  const pool = Array.from({ length: count }, (_, i) => i);
  for (let i = 0; i < size; i += 1) {
    const j = i + Math.floor(random() * (count - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, size);
}

export function ransacFundamental(left, right, threshold, iterations, seed) {
  const random = makeRandom(seed);
  let bestCount = -1;
  let bestF = null;
  for (let i = 0; i < iterations; i += 1) {
    const idx = sampleIndices(left.length, MIN_POINTS_FUNDAMENTAL, random);
    let candidate;
    try {
      candidate = fundamentalFromPoints(idx.map((n) => left[n]), idx.map((n) => right[n]));
    } catch {
      continue;
    }
    const count = sampsonDistance(candidate, left, right).filter((d) => d < threshold).length;
    if (count > bestCount) {
      bestCount = count;
      bestF = candidate;
    }
  }
  if (!bestF) throw new Error('ransac found no candidate');
  let inliers = sampsonDistance(bestF, left, right).map((d) => d < threshold);
  if (inliers.filter(Boolean).length < MIN_POINTS_FUNDAMENTAL) return { f: bestF, inliers };
  const refined = fundamentalFromPoints(
    left.filter((_, i) => inliers[i]),
    right.filter((_, i) => inliers[i]),
  );
  inliers = sampsonDistance(refined, left, right).map((d) => d < threshold);
  return { f: refined, inliers };
}

export function projectToEssential(e) {
  const svd = new SingularValueDecomposition(new Matrix(e));
  const mean = (svd.diagonal[0] + svd.diagonal[1]) / 2;
  return svd.leftSingularVectors
    .mmul(Matrix.diag([mean, mean, 0]))
    .mmul(svd.rightSingularVectors.transpose())
    .to2DArray();
}

export function triangulatePoint(p1, p2, ray1, ray2) {
  const rows = [
    p1[2].map((v, i) => ray1[0] * v - p1[0][i]),
    p1[2].map((v, i) => ray1[1] * v - p1[1][i]),
    p2[2].map((v, i) => ray2[0] * v - p2[0][i]),
    p2[2].map((v, i) => ray2[1] * v - p2[1][i]),
  ];
  const x = nullVector(rows);
  return [x[0] / x[3], x[1] / x[3], x[2] / x[3]];
}

const asProjection = (r, t) => r.map((row, i) => [...row, t[i]]);

export function decomposeEssential(e, rays1, rays2) {
  const svd = new SingularValueDecomposition(new Matrix(projectToEssential(e)));
  const u = svd.leftSingularVectors;
  const v = svd.rightSingularVectors;
  if (determinant(u) < 0) u.mulColumn(2, -1);
  if (determinant(v) < 0) v.mulColumn(2, -1);
  const uArray = u.to2DArray();
  const vt = v.transpose().to2DArray();
  const w = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
  const translation = uArray.map((row) => row[2]);
  const p1 = asProjection(identity(), [0, 0, 0]);

  let best = { count: -1 };
  for (const r of [matmul(matmul(uArray, w), vt), matmul(matmul(uArray, transpose(w)), vt)]) {
    for (const sign of [1, -1]) {
      const t = translation.map((value) => value * sign);
      const p2 = asProjection(r, t);
      const mask = rays1.map((ray, i) => {
        const point = triangulatePoint(p1, p2, ray, rays2[i]);
        const inSecond = apply(r, point).map((value, j) => value + t[j]);
        return point[2] > 0 && inSecond[2] > 0;
      });
      const count = mask.filter(Boolean).length;
      if (count > best.count) best = { count, r, t, mask };
    }
  }
  return best;
}

export function solvePnp(points, pixels, k, kInverse) {
  if (points.length < MIN_POINTS_PNP) {
    throw new Error(`need ${MIN_POINTS_PNP} points, got ${points.length}`);
  }
  const u = isotropicNormalizer(points, Math.sqrt(3));
  const t = isotropicNormalizer(pixels, Math.SQRT2);
  const rows = [];
  points.forEach((point, i) => {
    const x = apply(u, homogeneous(point));
    const uv = apply(t, homogeneous(pixels[i]));
    rows.push([...x, 0, 0, 0, 0, ...x.map((v) => -uv[0] * v)]);
    rows.push([0, 0, 0, 0, ...x, ...x.map((v) => -uv[1] * v)]);
  });
  const solution = nullVector(rows);
  const normalized = new Matrix([solution.slice(0, 4), solution.slice(4, 8), solution.slice(8, 12)]);
  const p = inverse(new Matrix(t)).mmul(normalized).mmul(new Matrix(u)).to2DArray();

  let m = matmul(kInverse, p.map((row) => row.slice(0, 3)));
  let tv = apply(kInverse, p.map((row) => row[3]));
  if (determinant(new Matrix(m)) < 0) {
    m = m.map((row) => row.map((v) => -v));
    tv = tv.map((v) => -v);
  }
  const svd = new SingularValueDecomposition(new Matrix(m));
  let r = svd.leftSingularVectors.mmul(svd.rightSingularVectors.transpose()).to2DArray();
  if (determinant(new Matrix(r)) < 0) {
    r = r.map((row) => row.map((v) => -v));
    tv = tv.map((v) => -v);
  }
  const scale = svd.diagonal.reduce((sum, v) => sum + v, 0) / svd.diagonal.length;
  return { r, t: tv.map((v) => v / scale) };
}