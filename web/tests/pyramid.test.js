import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  INITIAL_SIGMA,
  SCALES_PER_OCTAVE,
  blur,
  buildDifferenceOfGaussian,
  buildGaussianPyramid,
  gaussianKernel,
  halve,
  octaveCount,
  upsample,
} from '../src/pyramid.js';

const reference = JSON.parse(
  readFileSync(fileURLToPath(new URL('./blur-reference.json', import.meta.url)), 'utf8'),
);

const source = {
  width: reference.width,
  height: reference.height,
  data: Float32Array.from(reference.image),
};

const sum = (values) => values.reduce((total, value) => total + value, 0);

test('kernel matches opencv and sums to one', () => {
  for (const [sigma, expected] of Object.entries(reference.kernels)) {
    const { kernel } = gaussianKernel(Number(sigma));
    assert.equal(kernel.length, expected.length, `length differs at sigma ${sigma}`);
    assert.ok(Math.abs(sum([...kernel]) - 1) < 1e-6);
    const worst = Math.max(...expected.map((v, i) => Math.abs(v - kernel[i])));
    assert.ok(worst < 1e-6, `sigma ${sigma} kernel differs by ${worst}`);
  }
});

test('blur matches opencv on a real image', () => {
  for (const [sigma, expected] of Object.entries(reference.blurred)) {
    const got = blur(source, Number(sigma));
    const worst = Math.max(...expected.map((v, i) => Math.abs(v - got.data[i])));
    assert.ok(worst < 1e-5, `sigma ${sigma} differs by ${worst}`);
  }
});

test('blur preserves total brightness', () => {
  const before = sum([...source.data]);
  for (const sigma of [0.8, 1.6, 3.2]) {
    const after = sum([...blur(source, sigma).data]);
    assert.ok(Math.abs(after - before) / before < 0.01);
  }
});

test('blurring a flat image changes nothing', () => {
  const flat = { width: 20, height: 16, data: new Float32Array(320).fill(0.37) };
  const got = blur(flat, 2.5);
  assert.ok(Math.max(...got.data.map((v) => Math.abs(v - 0.37))) < 1e-6);
});

test('two small blurs equal one larger blur', () => {
  const first = 1.2;
  const second = 1.6;
  const combined = Math.sqrt(first * first + second * second);
  const twice = blur(blur(source, first), second);
  const once = blur(source, combined);
  const worst = Math.max(...[...once.data].map((v, i) => Math.abs(v - twice.data[i])));
  assert.ok(worst < 5e-3, `semigroup property broken by ${worst}`);
});

test('upsample then halve returns the original size', () => {
  const back = halve(upsample(source));
  assert.equal(back.width, source.width);
  assert.equal(back.height, source.height);
});

test('octave count shrinks with image size', () => {
  assert.ok(octaveCount(640, 480) > octaveCount(160, 120));
  assert.ok(octaveCount(32, 32) >= 1);
});

test('pyramid has the expected shape', () => {
  const octaves = octaveCount(source.width, source.height);
  const pyramid = buildGaussianPyramid(source, octaves);
  assert.ok(pyramid.length >= 1);
  for (const layers of pyramid) {
    assert.equal(layers.length, SCALES_PER_OCTAVE + 3);
    for (const layer of layers) {
      assert.equal(layer.data.length, layer.width * layer.height);
    }
  }
  for (let i = 1; i < pyramid.length; i += 1) {
    assert.ok(pyramid[i][0].width < pyramid[i - 1][0].width);
  }
});

test('each layer is blurrier than the one before', () => {
  const pyramid = buildGaussianPyramid(source, 1);
  const variance = (layer) => {
    let total = 0;
    for (let y = 1; y < layer.height - 1; y += 1) {
      for (let x = 1; x < layer.width - 1; x += 1) {
        const dx = layer.data[y * layer.width + x + 1] - layer.data[y * layer.width + x - 1];
        const dy = layer.data[(y + 1) * layer.width + x] - layer.data[(y - 1) * layer.width + x];
        total += dx * dx + dy * dy;
      }
    }
    return total / (layer.width * layer.height);
  };
  const energies = pyramid[0].map(variance);
  for (let i = 1; i < energies.length; i += 1) {
    assert.ok(energies[i] < energies[i - 1], `layer ${i} is not smoother`);
  }
});

test('difference of gaussian has one fewer layer and sums near zero', () => {
  const pyramid = buildGaussianPyramid(source, 2);
  const dog = buildDifferenceOfGaussian(pyramid);
  assert.equal(dog.length, pyramid.length);
  for (let octave = 0; octave < dog.length; octave += 1) {
    assert.equal(dog[octave].length, pyramid[octave].length - 1);
    for (const layer of dog[octave]) {
      const mean = sum([...layer.data]) / layer.data.length;
      assert.ok(Math.abs(mean) < 0.01, `dog mean ${mean} is not near zero`);
    }
  }
});