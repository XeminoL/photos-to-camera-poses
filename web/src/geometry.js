import { EigenvalueDecomposition, Matrix } from '../vendor/ml-matrix.js';

export const ROTATION_EPS = 1e-12;

export const skew = (v) => [
  [0, -v[2], v[1]],
  [v[2], 0, -v[0]],
  [-v[1], v[0], 0],
];

export const identity = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

export const matmul = (a, b) => a.map((row) =>
  b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));

export const transpose = (a) => a[0].map((_, j) => a.map((row) => row[j]));

export const apply = (a, v) => a.map((row) => row.reduce((sum, value, j) => sum + value * v[j], 0));

export const scale = (a, factor) => a.map((row) => row.map((value) => value * factor));

export const add = (a, b) => a.map((row, i) => row.map((value, j) => value + b[i][j]));

export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

export const norm = (v) => Math.hypot(...v);

export const subtract = (a, b) => a.map((value, i) => value - b[i]);

export const trace = (a) => a.reduce((sum, row, i) => sum + row[i], 0);

export function rodrigues(w) {
  const theta = norm(w);
  if (theta < ROTATION_EPS) return identity();
  const k = skew(w.map((value) => value / theta));
  const kk = matmul(k, k);
  return add(add(identity(), scale(k, Math.sin(theta))), scale(kk, 1 - Math.cos(theta)));
}

const clamp = (value) => Math.min(1, Math.max(-1, value));

export const rotationAngle = (r) => (Math.acos(clamp((trace(r) - 1) / 2)) * 180) / Math.PI;

export function rotationAngleFromPhase(r) {
  const eig = new EigenvalueDecomposition(new Matrix(r));
  let best = 0;
  let angle = 0;
  eig.realEigenvalues.forEach((re, i) => {
    const im = eig.imaginaryEigenvalues[i];
    if (Math.abs(im) > best) {
      best = Math.abs(im);
      angle = Math.abs(Math.atan2(im, re));
    }
  });
  return (angle * 180) / Math.PI;
}

export function rotationAxis(r) {
  const eig = new EigenvalueDecomposition(new Matrix(r));
  let index = 0;
  let closest = Infinity;
  eig.realEigenvalues.forEach((re, i) => {
    const distance = Math.hypot(re - 1, eig.imaginaryEigenvalues[i]);
    if (distance < closest) {
      closest = distance;
      index = i;
    }
  });
  const column = eig.eigenvectorMatrix.getColumn(index);
  const length = norm(column);
  return column.map((value) => value / length);
}

export const angleBetweenRotations = (a, b) => rotationAngle(matmul(transpose(a), b));

export const angleBetweenDirections = (a, b) =>
  (Math.acos(clamp(Math.abs(dot(a, b) / (norm(a) * norm(b))))) * 180) / Math.PI;

export const cameraCenter = (r, t) => apply(transpose(r), t).map((value) => -value);

export function relativePose(rFrom, tFrom, rTo, tTo) {
  const r = matmul(rTo, transpose(rFrom));
  return { r, t: subtract(tTo, apply(r, tFrom)) };
}

export const normalizePixel = (pixel, kInverse) => apply(kInverse, [pixel[0], pixel[1], 1]);

export function project(k, r, t, point) {
  const y = apply(r, point).map((value, i) => value + t[i]);
  const p = apply(k, y);
  return [p[0] / p[2], p[1] / p[2]];
}

export function reprojectionError(k, r, t, point, observed) {
  const [x, y] = project(k, r, t, point);
  return Math.hypot(x - observed[0], y - observed[1]);
}

export function projectionJacobian(k, r, t, point) {
  const y = apply(r, point).map((value, i) => value + t[i]);
  const z = y[2];
  const [fx, s] = [k[0][0], k[0][1]];
  const fy = k[1][1];
  const dPi = [
    [fx / z, s / z, -(fx * y[0] + s * y[1]) / (z * z)],
    [0, fy / z, -(fy * y[1]) / (z * z)],
  ];
  const rotated = apply(r, point);
  return {
    dAngle: matmul(dPi, scale(skew(rotated), -1)),
    dTranslation: dPi,
    dPoint: matmul(dPi, r),
  };
}