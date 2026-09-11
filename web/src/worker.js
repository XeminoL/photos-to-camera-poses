import { MIN_PAIR_MATCHES, Session, detectImage, matchFeatures, packDescriptors } from './pipeline.js';

const DEFAULT_FOCAL_RATIO = 1.2;

function intrinsicsFor(width, height, focal) {
  const f = focal ?? DEFAULT_FOCAL_RATIO * Math.max(width, height);
  return [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]];
}

function post(stage, detail) {
  self.postMessage({ type: 'progress', stage, ...detail });
}

self.onmessage = async (event) => {
  const { images, focal, fast } = event.data;
  try {
    const started = Date.now();
    const features = new Map();
    images.forEach((image, index) => {
      features.set(index, detectImage(image, { doubleFirst: !fast }));
      post('detect', { done: index + 1, total: images.length });
    });

    const packed = new Map();
    for (const [index, list] of features) packed.set(index, packDescriptors(list));

    const matches = new Map();
    const totalPairs = (images.length * (images.length - 1)) / 2;
    let inspected = 0;
    for (let i = 0; i < images.length; i += 1) {
      for (let j = i + 1; j < images.length; j += 1) {
        const pairs = matchFeatures(packed.get(i), packed.get(j));
        if (pairs.length >= MIN_PAIR_MATCHES) matches.set(`${i}:${j}`, pairs);
        inspected += 1;
        if (inspected % 5 === 0 || inspected === totalPairs) {
          post('match', { done: inspected, total: totalPairs, kept: matches.size });
        }
      }
    }

    if (matches.size === 0) throw new Error('no image pair shares enough features');

    const k = intrinsicsFor(images[0].width, images[0].height, focal);
    const focalUsed = k[0][0];
    const session = new Session(features, matches, k);

    post('seed', {});
    const seed = session.chooseSeed();
    if (!seed) throw new Error('no image pair gives a usable starting geometry');

    post('reconstruct', { seed: seed.pair });
    const result = session.run(seed.pair);

    self.postMessage({
      type: 'done',
      seconds: (Date.now() - started) / 1000,
      seedPair: seed.pair,
      focal: focalUsed,
      focalGuessed: !focal,
      poses: [...result.reconstruction.poses].map(([camera, pose]) => ({
        camera,
        r: pose.r,
        t: pose.t,
        angleSigma: result.report.angleSigma.get(camera) ?? 0,
        centreSigma: result.report.centerSigma.get(camera) ?? 0,
      })),
      points: result.reconstruction.points,
      anchor: result.reconstruction.anchor,
      rms: result.report.rms,
      nullSpaceDimension: result.report.nullSpaceDimension,
      isDegenerate: result.report.isDegenerate,
      rejected: result.rejected,
      featureCounts: [...features.values()].map((list) => list.length),
      pairsKept: matches.size,
      pairsTotal: totalPairs,
    });
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message });
  }
};