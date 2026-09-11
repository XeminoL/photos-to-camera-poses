import { cameraCenter, matmul, rodrigues, subtract } from './geometry.js';

export const POINT_RADIUS = 1.6;
export const CAMERA_SIZE = 0.05;
export const MARGIN = 40;

function boundingSphere(points) {
  const centre = [0, 1, 2].map((axis) =>
    points.reduce((sum, point) => sum + point[axis], 0) / points.length);
  const radius = Math.max(...points.map((point) =>
    Math.hypot(...subtract(point, centre))), 1e-6);
  return { centre, radius };
}

export function createScene(points, poses) {
  const centres = poses.map((pose) => cameraCenter(pose.r, pose.t));
  const sphere = boundingSphere([...points, ...centres]);
  return { points, poses, centres, ...sphere };
}

function viewMatrix(yaw, pitch) {
  return matmul(rodrigues([0, yaw, 0]), rodrigues([pitch, 0, 0]));
}

export function drawScene(canvas, scene, view) {
  const context = canvas.getContext('2d');
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  if (!scene) return;

  const rotation = viewMatrix(view.yaw, view.pitch);
  const span = Math.min(width, height) - 2 * MARGIN;
  const scale = (span / (2 * scene.radius)) * view.zoom;

  const flatten = (point) => {
    const local = subtract(point, scene.centre);
    const rotated = rotation.map((row) =>
      row[0] * local[0] + row[1] * local[1] + row[2] * local[2]);
    return {
      x: width / 2 + rotated[0] * scale,
      y: height / 2 - rotated[1] * scale,
      depth: rotated[2],
    };
  };

  const drawn = scene.points.map(flatten).sort((a, b) => a.depth - b.depth);
  const style = getComputedStyle(canvas);
  const pointColour = style.getPropertyValue('--point').trim() || '#3b6ea5';
  const cameraColour = style.getPropertyValue('--camera').trim() || '#c25b3a';

  for (const point of drawn) {
    const fade = Math.max(0.25, Math.min(1, 0.5 + point.depth / (2 * scene.radius)));
    context.globalAlpha = fade;
    context.fillStyle = pointColour;
    context.beginPath();
    context.arc(point.x, point.y, POINT_RADIUS, 0, Math.PI * 2);
    context.fill();
  }

  context.globalAlpha = 1;
  context.strokeStyle = cameraColour;
  context.fillStyle = cameraColour;
  context.lineWidth = 1.5;
  scene.centres.forEach((centre, index) => {
    const origin = flatten(centre);
    const forward = scene.poses[index].r[2];
    const tip = flatten(centre.map((value, axis) =>
      value + forward[axis] * scene.radius * CAMERA_SIZE * 4));
    context.beginPath();
    context.arc(origin.x, origin.y, 4, 0, Math.PI * 2);
    context.fill();
    context.beginPath();
    context.moveTo(origin.x, origin.y);
    context.lineTo(tip.x, tip.y);
    context.stroke();
  });
}