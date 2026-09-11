import { Matrix, SingularValueDecomposition, EigenvalueDecomposition, solve, inverse, determinant } from '../vendor/ml-matrix.js';
const s = new Matrix([[220,-100,0,0,0,-100],[-100,220,-100,0,0,0],[0,-100,220,-100,0,0],
  [0,0,-100,220,-100,0],[0,0,0,-100,220,-100],[-100,0,0,0,-100,220]]);
const eig = new EigenvalueDecomposition(s, { assumeSymmetric: true });
const got = [...eig.realEigenvalues].sort((a,b)=>a-b).map(v=>Math.round(v*1e6)/1e6);
console.log('tri rieng circulant:', got.join(', '));
console.log('mong doi          : 20, 120, 120, 320, 320, 420');
console.log('khop:', got.join(',') === '20,120,120,320,320,420');
const e = new Matrix([[0,0,0],[0,0,-0.5],[0,0.5,0]]);
console.log('gia tri ky di [s,s,0]:', new SingularValueDecomposition(e).diagonal.map(v=>v.toFixed(6)).join(', '));
console.log('inverse, determinant, solve:', typeof inverse, typeof determinant, typeof solve);