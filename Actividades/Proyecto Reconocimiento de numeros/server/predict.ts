import fs from 'node:fs'
import path from 'node:path'
import type { Tensor28 } from './preprocess.js'

interface DenseWeights {
  hidden1Kernel: number[][]
  hidden1Bias: number[]
  hidden2Kernel: number[][]
  hidden2Bias: number[]
  outputKernel: number[][]
  outputBias: number[]
}

const weightPath = path.resolve(process.cwd(), 'models/dense_weights.json')
const weights: DenseWeights = JSON.parse(fs.readFileSync(weightPath, 'utf8'))

function dense(input: number[], kernel: number[][], bias: number[], relu = false): number[] {
  return bias.map((value, outIndex) => {
    let sum = value
    for (let inIndex = 0; inIndex < input.length; inIndex++) {
      sum += input[inIndex] * kernel[inIndex][outIndex]
    }
    return relu ? Math.max(0, sum) : sum
  })
}

function softmax(input: number[]): number[] {
  const max = Math.max(...input)
  const exp = input.map((value) => Math.exp(value - max))
  const sum = exp.reduce((total, value) => total + value, 0)
  return exp.map((value) => value / sum)
}

export function predictNumberDense(inputs: Tensor28[]) {
  const probabilities = inputs.map((tensor) => {
    const flat = tensor.flat()
    const hidden1 = dense(flat, weights.hidden1Kernel, weights.hidden1Bias, true)
    const hidden2 = dense(hidden1, weights.hidden2Kernel, weights.hidden2Bias, true)
    return softmax(dense(hidden2, weights.outputKernel, weights.outputBias))
  })
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
