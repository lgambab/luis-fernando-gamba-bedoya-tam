import fs from 'node:fs'
import path from 'node:path'
import type { Tensor28 } from './preprocess.js'

interface CnnWeights {
  conv1Kernel: number[]
  conv1Bias: number[]
  conv2Kernel: number[]
  conv2Bias: number[]
  dense1Kernel: number[][]
  dense1Bias: number[]
  dense2Kernel: number[][]
  dense2Bias: number[]
  testAccuracy: number
}

const weightPath = path.resolve(process.cwd(), 'models/cnn_weights.json')
const weights: CnnWeights = JSON.parse(fs.readFileSync(weightPath, 'utf8'))

function convRelu(
  input: number[][][],
  kernels: number[][],
  biases: number[],
  outChannels: number,
): number[][][] {
  const height = input.length
  const width = input[0].length
  const inChannels = input[0][0].length
  const output: number[][][] = Array.from({ length: height }, () =>
    Array.from({ length: width }, () => Array<number>(outChannels).fill(0)),
  )
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < outChannels; c++) {
        let sum = biases[c]
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const iy = y + ky
            const ix = x + kx
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
              for (let ic = 0; ic < inChannels; ic++) {
                const kernelIndex =
                  ((ky + 1) * 3 + kx + 1) * inChannels * outChannels +
                  ic * outChannels +
                  c
                sum += input[iy][ix][ic] * kernels[kernelIndex]
              }
            }
          }
        }
        output[y][x][c] = Math.max(0, sum)
      }
    }
  }
  return output
}

function pool(input: number[][][]): number[][][] {
  const height = input.length
  const width = input[0].length
  const channels = input[0][0].length
  const outHeight = height >> 1
  const outWidth = width >> 1
  const output: number[][][] = Array.from({ length: outHeight }, () =>
    Array.from({ length: outWidth }, () => Array<number>(channels).fill(0)),
  )
  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      for (let c = 0; c < channels; c++) {
        output[y][x][c] = Math.max(
          input[y * 2][x * 2][c],
          input[y * 2 + 1][x * 2][c],
          input[y * 2][x * 2 + 1][c],
          input[y * 2 + 1][x * 2 + 1][c],
        )
      }
    }
  }
  return output
}

function dense(input: number[], kernel: number[][], bias: number[]): number[] {
  return bias.map((b, outputIndex) => {
    let sum = b
    for (let inputIndex = 0; inputIndex < input.length; inputIndex++) {
      sum += input[inputIndex] * kernel[inputIndex][outputIndex]
    }
    return sum
  })
}

function softmax(input: number[]): number[] {
  const maximum = Math.max(...input)
  const exponents = input.map((value) => Math.exp(value - maximum))
  const sum = exponents.reduce((total, value) => total + value, 0)
  return exponents.map((value) => value / sum)
}

function predictDigit(input: Tensor28): number[] {
  const spatial: number[][][] = input.map((row) => row.map((value) => [value]))
  const conv1 = convRelu(spatial, weights.conv1Kernel, weights.conv1Bias, 32)
  const pool1 = pool(conv1)
  const conv2 = convRelu(pool1, weights.conv2Kernel, weights.conv2Bias, 64)
  const pool2 = pool(conv2)
  const flattened: number[] = []
  for (let y = 0; y < pool2.length; y++) {
    for (let x = 0; x < pool2[0].length; x++) {
      flattened.push(...pool2[y][x])
    }
  }
  const hidden = dense(flattened, weights.dense1Kernel as unknown as number[][], weights.dense1Bias)
    .map((value) => Math.max(0, value))
  return softmax(dense(hidden, weights.dense2Kernel as unknown as number[][], weights.dense2Bias))
}

export function predictNumberCnn(inputs: Tensor28[]) {
  const probabilities = inputs.map(predictDigit)
  const digits = probabilities.map((values) => values.indexOf(Math.max(...values)))
  const confidences = probabilities.map((values) => Math.max(...values))
  const number = digits.length === 1 ? digits[0] : digits[0] * 10 + digits[1]
  return {
    number,
    confidence: confidences.reduce((sum, value) => sum + value, 0) / confidences.length,
    probabilities: probabilities.flatMap((values, digitIndex) =>
      values.map((probability, label) =>
        [`Dígito ${digitIndex + 1}: clase ${label}`, probability.toFixed(6)] as [string, string],
      ),
    ),
  }
}
