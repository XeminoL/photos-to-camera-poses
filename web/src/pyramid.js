export const SCALES_PER_OCTAVE = 3;
export const INITIAL_SIGMA = 1.6;
export const CAMERA_SIGMA = 0.5;
export const KERNEL_WIDTH = 4;
export const MIN_OCTAVE_SIZE = 8;

export function gaussianKernel(sigma) {
  const radius = Math.max(1, Math.ceil(KERNEL_WIDTH * sigma));
  const kernel = new Float32Array(2 * radius + 1);
  const denominator = 2 * sigma * sigma;
  let sum = 0;
  for (let i = -radius; i <= radius; i += 1) {
    const value = Math.exp(-(i * i) / denominator);
    kernel[i + radius] = value;
    sum += value;
  }
  for (let i = 0; i < kernel.length; i += 1) kernel[i] /= sum;
  return { kernel, radius };
}

const reflect = (index, limit) => {
  if (index < 0) return -index;
  if (index >= limit) return 2 * limit - index - 2;
  return index;
};

export function blur(image, sigma) {
  const { kernel, radius } = gaussianKernel(sigma);
  const { width, height, data } = image;
  const horizontal = new Float32Array(width * height);
  const output = new Float32Array(width * height);

  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let i = -radius; i <= radius; i += 1) {
        sum += kernel[i + radius] * data[row + reflect(x + i, width)];
      }
      horizontal[row + x] = sum;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let i = -radius; i <= radius; i += 1) {
        sum += kernel[i + radius] * horizontal[reflect(y + i, height) * width + x];
      }
      output[y * width + x] = sum;
    }
  }
  return { width, height, data: output };
}

export function upsample(image) {
  const width = image.width * 2;
  const height = image.height * 2;
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const sy = (y - 0.5) / 2;
    const y0 = Math.max(0, Math.min(image.height - 1, Math.floor(sy)));
    const y1 = Math.min(image.height - 1, y0 + 1);
    const fy = Math.max(0, Math.min(1, sy - y0));
    for (let x = 0; x < width; x += 1) {
      const sx = (x - 0.5) / 2;
      const x0 = Math.max(0, Math.min(image.width - 1, Math.floor(sx)));
      const x1 = Math.min(image.width - 1, x0 + 1);
      const fx = Math.max(0, Math.min(1, sx - x0));
      data[y * width + x] =
        image.data[y0 * image.width + x0] * (1 - fx) * (1 - fy)
        + image.data[y0 * image.width + x1] * fx * (1 - fy)
        + image.data[y1 * image.width + x0] * (1 - fx) * fy
        + image.data[y1 * image.width + x1] * fx * fy;
    }
  }
  return { width, height, data };
}

export function halve(image) {
  const width = Math.floor(image.width / 2);
  const height = Math.floor(image.height / 2);
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data[y * width + x] = image.data[2 * y * image.width + 2 * x];
    }
  }
  return { width, height, data };
}

export function octaveCount(width, height) {
  return Math.max(1, Math.round(Math.log2(Math.min(width, height))) - 2);
}

export function buildGaussianPyramid(image, octaves, scales = SCALES_PER_OCTAVE, doubleFirst = true) {
  const source = doubleFirst ? upsample(image) : image;
  const priorSigma = doubleFirst ? 2 * CAMERA_SIGMA : CAMERA_SIGMA;
  const start = Math.sqrt(Math.max(INITIAL_SIGMA ** 2 - priorSigma ** 2, 0.01));
  let base = blur(source, start);

  const k = 2 ** (1 / scales);
  const steps = [];
  for (let i = 1; i < scales + 3; i += 1) {
    const previous = INITIAL_SIGMA * k ** (i - 1);
    steps.push(Math.sqrt((previous * k) ** 2 - previous ** 2));
  }

  const pyramid = [];
  for (let octave = 0; octave < octaves; octave += 1) {
    const layers = [base];
    for (const step of steps) layers.push(blur(layers[layers.length - 1], step));
    pyramid.push(layers);
    if (octave + 1 < octaves) {
      const next = halve(layers[scales]);
      if (next.width < MIN_OCTAVE_SIZE || next.height < MIN_OCTAVE_SIZE) break;
      base = next;
    }
  }
  return pyramid;
}

export function buildDifferenceOfGaussian(pyramid) {
  return pyramid.map((layers) => {
    const differences = [];
    for (let i = 1; i < layers.length; i += 1) {
      const data = new Float32Array(layers[i].data.length);
      for (let n = 0; n < data.length; n += 1) {
        data[n] = layers[i].data[n] - layers[i - 1].data[n];
      }
      differences.push({ width: layers[i].width, height: layers[i].height, data });
    }
    return differences;
  });
}