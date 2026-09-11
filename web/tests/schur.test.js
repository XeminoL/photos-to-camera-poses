import assert from 'node:assert/strict';
import test from 'node:test';
import { Matrix, solve } from '../vendor/ml-matrix.js';

import {
  CAMERA_PARAMETERS,
  buildNormalEquations,
  solveSchur,
} from '../src/schur.js';

function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296 - 0.5;
  };
}

function makeProblem({ cameras = 3, points = 12, seed = 5 } = {}) {
  const random = seeded(seed);
  const cameraIndex = new Map();
  for (let c = 0; c < cameras; c += 1) cameraIndex.set(c + 1, c);

  const observations = [];
  const jacobians = [];
  const residuals = [];
  for (let point = 0; point < points; point += 1) {
    for (const camera of cameraIndex.keys()) {
      observations.push({ camera, point });
      jacobians.push({
        dCamera: [
          Array.from({ length: CAMERA_PARAMETERS }, () => random()),
          Array.from({ length: CAMERA_PARAMETERS }, () => random()),
        ],
        dPoint: [
          Array.from({ length: 3 }, () => random()),
          Array.from({ length: 3 }, () => random()),
        ],
      });
      residuals.push([random(), random()]);
    }
  }
  return { observations, jacobians, residuals, cameraIndex, points };
}

function solveDense(problem, lambda) {
  const { observations, jacobians, residuals, cameraIndex, points } = problem;
  const width = cameraIndex.size * CAMERA_PARAMETERS + points * 3;
  const j = Matrix.zeros(2 * observations.length, width);
  const r = new Float64Array(2 * observations.length);

  observations.forEach((obs, n) => {
    const base = cameraIndex.get(obs.camera) * CAMERA_PARAMETERS;
    const pointBase = cameraIndex.size * CAMERA_PARAMETERS + obs.point * 3;
    for (let row = 0; row < 2; row += 1) {
      for (let a = 0; a < CAMERA_PARAMETERS; a += 1) {
        j.set(2 * n + row, base + a, jacobians[n].dCamera[row][a]);
      }
      for (let a = 0; a < 3; a += 1) {
        j.set(2 * n + row, pointBase + a, jacobians[n].dPoint[row][a]);
      }
      r[2 * n + row] = residuals[n][row];
    }
  });

  const h = j.transpose().mmul(j);
  for (let i = 0; i < h.rows; i += 1) {
    h.set(i, i, h.get(i, i) * (1 + lambda) + lambda * 1e-12);
  }
  const gradient = j.transpose().mmul(Matrix.columnVector([...r]));
  return solve(h, gradient.mul(-1)).getColumn(0);
}

function runSchur(problem, lambda) {
  const system = buildNormalEquations(
    problem.observations, problem.jacobians, problem.residuals,
    problem.cameraIndex, problem.points,
  );
  return solveSchur(system, lambda);
}

test('schur matches the dense solution', () => {
  for (const lambda of [1e-3, 1e-2, 0.1]) {
    const problem = makeProblem();
    const dense = solveDense(problem, lambda);
    const sparse = runSchur(problem, lambda);
    assert.ok(sparse, 'schur returned nothing');

    const cameraWidth = problem.cameraIndex.size * CAMERA_PARAMETERS;
    let worst = 0;
    for (let i = 0; i < cameraWidth; i += 1) {
      worst = Math.max(worst, Math.abs(dense[i] - sparse.cameraDelta[i]));
    }
    for (let i = 0; i < problem.points * 3; i += 1) {
      worst = Math.max(worst, Math.abs(dense[cameraWidth + i] - sparse.pointDelta[i]));
    }
    assert.ok(worst < 1e-8, `lambda ${lambda}: differs by ${worst}`);
  }
});

test('schur handles a single camera', () => {
  const problem = makeProblem({ cameras: 1, points: 6, seed: 11 });
  const dense = solveDense(problem, 1e-3);
  const sparse = runSchur(problem, 1e-3);
  assert.ok(sparse);
  assert.ok(Math.abs(dense[0] - sparse.cameraDelta[0]) < 1e-8);
});

test('schur scales to many points', () => {
  const problem = makeProblem({ cameras: 4, points: 60, seed: 3 });
  const dense = solveDense(problem, 1e-2);
  const sparse = runSchur(problem, 1e-2);
  const cameraWidth = problem.cameraIndex.size * CAMERA_PARAMETERS;
  let worst = 0;
  for (let i = 0; i < cameraWidth; i += 1) {
    worst = Math.max(worst, Math.abs(dense[i] - sparse.cameraDelta[i]));
  }
  assert.ok(worst < 1e-7, `camera delta differs by ${worst}`);
});

test('point blocks stay independent of each other', () => {
  const problem = makeProblem({ cameras: 2, points: 20, seed: 7 });
  const system = buildNormalEquations(
    problem.observations, problem.jacobians, problem.residuals,
    problem.cameraIndex, problem.points,
  );
  assert.equal(system.pointBlocks.length, problem.points);
  for (const block of system.pointBlocks) {
    assert.equal(block.length, 9);
    assert.ok(Math.abs(block[1] - block[3]) < 1e-12, 'point block must be symmetric');
    assert.ok(Math.abs(block[2] - block[6]) < 1e-12);
    assert.ok(Math.abs(block[5] - block[7]) < 1e-12);
  }
});