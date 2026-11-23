import express from 'express'
import cors from 'cors'
import { predictNumberDense } from './server/predict.js'
import { predictNumberCnn } from './server/predictCnn.js'
import {
  decodeImageDataUrl,
  preprocessingPreview,
  segmentImage,
  type RecognitionResponse,
} from './server/preprocess.js'

const app = express()
app.use(cors())
app.use(express.json({ limit: '12mb' }))

async function recognize(
  model: 'dense' | 'cnn',
  imageDataUrl: unknown,
): Promise<RecognitionResponse> {
  if (typeof imageDataUrl !== 'string' || !imageDataUrl.startsWith('data:image/')) {
    throw Object.assign(new Error('Formato de imagen inválido. Use PNG, JPG o JPEG.'), { status: 400 })
  }
  if (!/^data:image\/(png|jpeg|jpg);base64,/i.test(imageDataUrl)) {
    throw Object.assign(new Error('Formato de imagen inválido. Use PNG, JPG o JPEG.'), { status: 400 })
  }

  const image = await decodeImageDataUrl(imageDataUrl)
  const segment = await segmentImage(image)
  if (segment.inputs.length === 0) {
    throw Object.assign(new Error('La imagen no contiene trazos detectables.'), { status: 400 })
  }
  if (segment.inputs.length > 2) {
    throw Object.assign(new Error('La imagen contiene más de dos dígitos no segmentables.'), { status: 400 })
  }

  const result = model === 'cnn'
    ? predictNumberCnn(segment.inputs)
    : predictNumberDense(segment.inputs)
  const preview = await preprocessingPreview(image, segment.inputs)
  return {
    number: result.number,
    confidence: `${(result.confidence * 100).toFixed(2)}%`,
    grayscale: preview.grayscale,
    binary: preview.binary,
    cropped: preview.cropped,
    processed: preview.processed,
    probabilities: result.probabilities,
  }
}

app.post('/api/recognize-dense', async (req, res) => {
  try {
    res.json(await recognize('dense', req.body?.imageDataUrl))
  } catch (error) {
    const status = error instanceof Error && 'status' in error
      ? Number((error as Error & { status: number }).status)
      : 500
    res.status(status).json({
      error: error instanceof Error ? error.message : 'Error interno al reconocer la imagen.',
    })
  }
})

app.post('/api/recognize-cnn', async (req, res) => {
  try {
    res.json(await recognize('cnn', req.body?.imageDataUrl))
  } catch (error) {
    const status = error instanceof Error && 'status' in error
      ? Number((error as Error & { status: number }).status)
      : 500
    res.status(status).json({
      error: error instanceof Error ? error.message : 'Error interno al reconocer la imagen.',
    })
  }
})

app.listen(3000, () => {
  console.log('API listening on http://localhost:3000')
})
