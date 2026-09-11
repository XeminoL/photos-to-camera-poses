import { readFileSync } from 'node:fs';

import {
  MIN_PAIR_MATCHES,
  Session,
  detectImage,
  matchFeatures,
} from '../src/pipeline.js';
import {
  angleBetweenRotations,
  cameraCenter,
  matmul,
  norm,
  relativePose,
  subtract,
  transpose,
} from '../src/geometry.js';

const DATASET = './tests/temple.json';
const SEED_PAIR = [11, 12];

function loadDataset() {
  const data = JSON.parse(readFileSync(DATASET, 'utf8'));
  const images = data.images.map((pixels) => ({
    width: data.width,
    height: data.height,
    data: Float32Array.from(pixels),
  }));
  return { cameras: data.cameras, images };
}

function buildSession({ cameras, images }) {
  const started = Date.now();
  const features = new Map();
  images.forEach((image, index) => {
    features.set(index, detectImage(image));
  });
  const detectMs = Date.now() - started;

  const matchStart = Date.now();
  const matches = new Map();
  for (let i = 0; i < images.length; i += 1) {
    for (let j = i + 1; j < images.length; j += 1) {
      const pairs = matchFeatures(features.get(i), features.get(j));
      if (pairs.length >= MIN_PAIR_MATCHES) matches.set(`${i}:${j}`, pairs);
    }
  }
  const matchMs = Date.now() - matchStart;

  const counts = [...features.values()].map((list) => list.length);
  console.log(`detected ${Math.min(...counts)}-${Math.max(...counts)} features per image`
    + ` in ${(detectMs / 1000).toFixed(1)} s`);
  console.log(`matched ${matches.size} of ${(images.length * (images.length - 1)) / 2} pairs`
    + ` in ${(matchMs / 1000).toFixed(1)} s`);
  return new Session(features, matches, cameras[0].k);
}

function report(result, cameras, anchor) {
  const { reconstruction, report: summary } = result;
  console.log(`\n${reconstruction.poses.size} cameras, ${reconstruction.points.length} points,`
    + ` rms ${summary.rms.toFixed(4)} px, null space ${summary.nullSpaceDimension}`);
  console.log('camera | angle error | predicted | centre error');

  let scale = null;
  for (const camera of [...reconstruction.poses.keys()].sort((a, b) => a - b)) {
    const pose = reconstruction.poses.get(camera);
    const truth = relativePose(cameras[anchor].r, cameras[anchor].t,
      cameras[camera].r, cameras[camera].t);
    if (scale === null && norm(truth.t) > 1e-9) {
      scale = norm(truth.t) / Math.max(norm(pose.t), 1e-12);
    }
    const angle = angleBetweenRotations(pose.r, truth.r);
    const estimated = cameraCenter(pose.r, pose.t).map((v) => v * (scale ?? 1));
    const centre = norm(subtract(estimated, cameraCenter(truth.r, truth.t)));
    const predicted = summary.angleSigma.get(camera) ?? 0;
    console.log(`${String(camera + 1).padStart(6)} | ${angle.toFixed(4).padStart(11)}`
      + ` | ${predicted.toFixed(4).padStart(9)} | ${centre.toFixed(6).padStart(12)}`);
  }
  for (const item of result.rejected) {
    const extra = item.excessNullDimensions ? `, excess null ${item.excessNullDimensions}` : '';
    console.log(`  rejected ${item.camera + 1}: ${item.reason}`
      + ` (matches ${item.matches ?? 0}, inliers ${item.inliers ?? 0}${extra})`);
  }
}

const dataset = loadDataset();
const session = buildSession(dataset);
const started = Date.now();
const result = session.run(SEED_PAIR);
console.log(`reconstructed in ${((Date.now() - started) / 1000).toFixed(1)} s`);
report(result, dataset.cameras, SEED_PAIR[0]);