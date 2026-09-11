import { inverse, Matrix } from '../vendor/ml-matrix.js';

import { optimize } from './bundle.js';
import { describeKeypoints } from './descriptor.js';
import {
  MIN_POINTS_PNP,
  PNP_SAMPLE_SIZE,
  decomposeEssential,
  ransacFundamental,
  solvePnp,
  triangulatePoint,
} from './estimation.js';
import { identity, normalizePixel, reprojectionError, rotationAngle } from './geometry.js';
import { findKeypoints } from './keypoints.js';
import { buildDifferenceOfGaussian, buildGaussianPyramid, octaveCount } from './pyramid.js';

export const FUNDAMENTAL_THRESHOLD = 0.3;
export const FUNDAMENTAL_ITERATIONS = 1000;
export const LOWE_RATIO = 0.75;
export const PNP_THRESHOLD = 20;
export const PNP_ITERATIONS = 800;
export const BUNDLE_ITERATIONS = 12;
export const MIN_PAIR_MATCHES = 30;
export const MIN_PNP_MATCHES = 8;
export const MIN_SEED_ANGLE = 3;
export const MAX_SEED_ANGLE = 60;
export const SEED_CANDIDATES = 8;
export const DEFAULT_SEED = 7;

export function detectImage(image, { doubleFirst = true } = {}) {
  const octaves = octaveCount(image.width, image.height) - (doubleFirst ? 0 : 1);
  const pyramid = buildGaussianPyramid(image, Math.max(1, octaves), undefined, doubleFirst);
  const keypoints = findKeypoints(buildDifferenceOfGaussian(pyramid));
  return describeKeypoints(pyramid, keypoints).map((point) => ({
    ...point,
    x: doubleFirst ? point.x : point.x * 2,
    y: doubleFirst ? point.y : point.y * 2,
  }));
}

export function packDescriptors(features) {
  const width = features[0]?.descriptor.length ?? 0;
  const packed = new Float32Array(features.length * width);
  features.forEach((feature, i) => packed.set(feature.descriptor, i * width));
  return { packed, width, count: features.length };
}

export function matchFeatures(left, right, ratio = LOWE_RATIO) {
  const a = left.packed ? left : packDescriptors(left);
  const b = right.packed ? right : packDescriptors(right);
  const { width } = a;
  const limit = ratio * ratio;
  const pairs = [];

  for (let i = 0; i < a.count; i += 1) {
    const base = i * width;
    let best = Infinity;
    let second = Infinity;
    let bestIndex = -1;
    for (let j = 0; j < b.count; j += 1) {
      const other = j * width;
      let sum = 0;
      for (let n = 0; n < width; n += 1) {
        const difference = a.packed[base + n] - b.packed[other + n];
        sum += difference * difference;
        if (sum >= second) break;
      }
      if (sum < best) {
        second = best;
        best = sum;
        bestIndex = j;
      } else if (sum < second) {
        second = sum;
      }
    }
    if (best < limit * second) pairs.push([i, bestIndex]);
  }
  return pairs;
}

const key = (left, right) => (left < right ? `${left}:${right}` : `${right}:${left}`);

function samplePnp(points, pixels, k, kInverse, threshold, iterations, seed) {
  let state = seed >>> 0;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const size = Math.min(Math.max(PNP_SAMPLE_SIZE, MIN_POINTS_PNP), points.length);
  let bestCount = -1;
  let best = null;
  for (let round = 0; round < iterations; round += 1) {
    const pool = points.map((_, i) => i);
    for (let i = 0; i < size; i += 1) {
      const j = i + Math.floor(random() * (pool.length - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const chosen = pool.slice(0, size);
    let pose;
    try {
      pose = solvePnp(chosen.map((i) => points[i]), chosen.map((i) => pixels[i]), k, kInverse);
    } catch {
      continue;
    }
    const count = points.filter((point, i) =>
      reprojectionError(k, pose.r, pose.t, point, pixels[i]) < threshold).length;
    if (count > bestCount) {
      bestCount = count;
      best = pose;
    }
  }
  if (!best) throw new Error('pnp found no candidate');
  const inliers = points.map((point, i) =>
    reprojectionError(k, best.r, best.t, point, pixels[i]) < threshold);
  if (inliers.filter(Boolean).length < MIN_POINTS_PNP) return { ...best, inliers };
  const keptPoints = points.filter((_, i) => inliers[i]);
  const keptPixels = pixels.filter((_, i) => inliers[i]);
  try {
    const refined = solvePnp(keptPoints, keptPixels, k, kInverse);
    const after = points.map((point, i) =>
      reprojectionError(k, refined.r, refined.t, point, pixels[i]) < threshold);
    if (after.filter(Boolean).length < inliers.filter(Boolean).length) {
      return { ...best, inliers };
    }
    return { ...refined, inliers: after };
  } catch {
    return { ...best, inliers };
  }
}

export class Session {
  constructor(features, matches, k, seed = DEFAULT_SEED) {
    this.features = features;
    this.matches = matches;
    this.k = k;
    this.kInverse = inverse(new Matrix(k)).to2DArray();
    this.seed = seed;
    this.tracks = new Map();
  }

  pixel(camera, index) {
    const point = this.features.get(camera)[index];
    return [point.x, point.y];
  }

  pairsFor(left, right) {
    const stored = this.matches.get(key(left, right));
    if (!stored) return null;
    return left < right ? stored : stored.map(([a, b]) => [b, a]);
  }

  initialize(left, right) {
    const pairs = this.pairsFor(left, right);
    if (!pairs) throw new Error(`no matches between ${left} and ${right}`);
    const x1 = pairs.map(([a]) => this.pixel(left, a));
    const x2 = pairs.map(([, b]) => this.pixel(right, b));
    const { f, inliers } = ransacFundamental(
      x1, x2, FUNDAMENTAL_THRESHOLD, FUNDAMENTAL_ITERATIONS, this.seed,
    );
    const keptPairs = pairs.filter((_, i) => inliers[i]);
    const rays1 = x1.filter((_, i) => inliers[i]).map((p) => normalizePixel(p, this.kInverse));
    const rays2 = x2.filter((_, i) => inliers[i]).map((p) => normalizePixel(p, this.kInverse));

    const kt = this.k[0].map((_, j) => this.k.map((row) => row[j]));
    const essential = kt.map((row) => row.map((_, j) =>
      row.reduce((sum, value, n) => sum + value * f[n][j], 0)));
    const e = essential.map((row) => row.map((_, j) =>
      row.reduce((sum, value, n) => sum + value * this.k[n][j], 0)));
    const best = decomposeEssential(e, rays1, rays2);

    const p1 = identity().map((row) => [...row, 0]);
    const p2 = best.r.map((row, i) => [...row, best.t[i]]);
    const points = [];
    const observations = [];
    this.tracks = new Map();
    rays1.forEach((ray, row) => {
      if (!best.mask[row]) return;
      const point = triangulatePoint(p1, p2, ray, rays2[row]);
      const index = points.length;
      points.push(point);
      const [a, b] = keptPairs[row];
      this.tracks.set(`${left}:${a}`, index);
      this.tracks.set(`${right}:${b}`, index);
      observations.push({ point: index, camera: left, pixel: this.pixel(left, a) });
      observations.push({ point: index, camera: right, pixel: this.pixel(right, b) });
    });

    return {
      poses: new Map([[left, { r: identity(), t: [0, 0, 0] }], [right, { r: best.r, t: best.t }]]),
      points,
      observations,
      anchor: left,
    };
  }

  correspondences(reconstruction, camera) {
    const found = new Map();
    for (const seen of reconstruction.poses.keys()) {
      const pairs = this.pairsFor(seen, camera);
      if (!pairs) continue;
      for (const [a, b] of pairs) {
        const point = this.tracks.get(`${seen}:${a}`);
        if (point !== undefined) found.set(point, b);
      }
    }
    return found;
  }

  tryRegister(reconstruction, camera, known) {
    const found = known ?? this.correspondences(reconstruction, camera);
    if (found.size < MIN_PNP_MATCHES) {
      return { rejected: { camera, reason: 'too few correspondences', matches: found.size } };
    }
    const indices = [...found.keys()];
    const points = indices.map((i) => reconstruction.points[i]);
    const pixels = indices.map((i) => this.pixel(camera, found.get(i)));

    let pose;
    try {
      pose = samplePnp(points, pixels, this.k, this.kInverse,
        PNP_THRESHOLD, PNP_ITERATIONS, this.seed);
    } catch {
      return { rejected: { camera, reason: 'pnp failed', matches: found.size } };
    }
    const inlierCount = pose.inliers.filter(Boolean).length;
    if (inlierCount < MIN_POINTS_PNP) {
      return {
        rejected: {
          camera, reason: 'too few pnp inliers', matches: found.size, inliers: inlierCount,
        },
      };
    }

    const poses = new Map(reconstruction.poses);
    poses.set(camera, { r: pose.r, t: pose.t });
    const observations = [...reconstruction.observations];
    const added = [];
    indices.forEach((index, n) => {
      if (reprojectionError(this.k, pose.r, pose.t, points[n], pixels[n]) >= PNP_THRESHOLD) return;
      observations.push({ point: index, camera, pixel: pixels[n] });
      added.push(`${camera}:${found.get(index)}`);
    });

    const candidate = {
      poses, points: reconstruction.points.map((p) => [...p]), observations,
      anchor: reconstruction.anchor,
    };
    const result = optimize(candidate, this.k, BUNDLE_ITERATIONS);
    if (result.report.isDegenerate) {
      return {
        rejected: {
          camera,
          reason: 'would make pose underdetermined',
          matches: found.size,
          inliers: inlierCount,
          excessNullDimensions: result.report.excessNullDimensions,
        },
      };
    }
    for (const track of added) this.tracks.set(track, this.tracks.get(track) ?? -1);
    added.forEach((track, n) => this.tracks.set(track, indices[n]));
    return { reconstruction: result.reconstruction, report: result.report };
  }

  scoreSeed(left, right) {
    let reconstruction;
    try {
      reconstruction = this.initialize(left, right);
    } catch {
      return null;
    }
    if (reconstruction.points.length < MIN_PAIR_MATCHES) return null;

    const pose = reconstruction.poses.get(right);
    const angle = rotationAngle(pose.r);
    if (!Number.isFinite(angle)) return null;
    if (angle < MIN_SEED_ANGLE || angle > MAX_SEED_ANGLE) return null;

    const { report } = optimize(reconstruction, this.k, 0);
    if (report.isDegenerate) return null;
    return { pair: [left, right], points: reconstruction.points.length, angle, rms: report.rms };
  }

  chooseSeed() {
    const ranked = [...this.matches.entries()]
      .map(([pairKey, pairs]) => ({ pair: pairKey.split(':').map(Number), count: pairs.length }))
      .sort((a, b) => b.count - a.count)
      .slice(0, SEED_CANDIDATES);
    const scored = [];
    for (const { pair } of ranked) {
      const result = this.scoreSeed(pair[0], pair[1]);
      if (result) scored.push(result);
    }
    if (scored.length === 0) return null;
    scored.sort((a, b) => b.points - a.points);
    return scored[0];
  }

  run(seedPair) {
    let reconstruction = this.initialize(seedPair[0], seedPair[1]);
    let { reconstruction: refined, report } = optimize(reconstruction, this.k, BUNDLE_ITERATIONS);
    reconstruction = refined;

    const rejected = [];
    const history = [{
      camera: seedPair[1], rms: report.rms,
      cameras: reconstruction.poses.size, points: reconstruction.points.length,
    }];
    const blocked = new Set();

    for (;;) {
      const remaining = [...this.features.keys()]
        .filter((c) => !reconstruction.poses.has(c) && !blocked.has(c));
      if (remaining.length === 0) break;
      const shared = new Map(remaining.map((camera) =>
        [camera, this.correspondences(reconstruction, camera)]));
      remaining.sort((a, b) => shared.get(b).size - shared.get(a).size);

      let progressed = false;
      for (const camera of remaining) {
        const outcome = this.tryRegister(reconstruction, camera, shared.get(camera));
        if (outcome.rejected) {
          rejected.push(outcome.rejected);
          blocked.add(camera);
          continue;
        }
        reconstruction = outcome.reconstruction;
        report = outcome.report;
        history.push({
          camera, rms: report.rms,
          cameras: reconstruction.poses.size, points: reconstruction.points.length,
        });
        progressed = true;
        break;
      }
      if (!progressed) break;
    }
    return { reconstruction, report, rejected, history };
  }
}