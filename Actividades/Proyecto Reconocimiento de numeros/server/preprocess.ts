import sharp from 'sharp'

export type Tensor28 = number[][]
export interface RecognitionResponse {
  number: number
  confidence: string
  grayscale: string
  binary: string
  cropped: string
  processed: string
  probabilities: Array<[string, string]>
  architecture?: string
}

interface DecodedImage {
  width: number
  height: number
  pixels: Uint8Array
}

export async function decodeImageDataUrl(dataUrl: string): Promise<DecodedImage> {
  const comma = dataUrl.indexOf(',')
  if (comma < 0) throw Object.assign(new Error('Imagen base64 inválida.'), { status: 400 })
  const buffer = Buffer.from(dataUrl.slice(comma + 1), 'base64')
  if (!buffer.length) throw Object.assign(new Error('La imagen está vacía.'), { status: 400 })
  const { data, info } = await sharp(buffer)
    .flatten({ background: '#fff' })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, pixels: new Uint8Array(data) }
}

function foregroundMask(image: DecodedImage): Uint8Array {
  const { width, height, pixels } = image
  const border: number[] = []
  for (let x = 0; x < width; x++) {
    border.push(pixels[x], pixels[(height - 1) * width + x])
  }
  for (let y = 0; y < height; y++) {
    border.push(pixels[y * width], pixels[y * width + width - 1])
  }
  border.sort((a, b) => a - b)
  const background = border[Math.floor(border.length / 2)]
  const darkInk = background >= 128
  const mask = new Uint8Array(width * height)
  for (let i = 0; i < pixels.length; i++) {
    mask[i] = darkInk ? (pixels[i] < 170 ? 1 : 0) : (pixels[i] > 170 ? 1 : 0)
  }
  return mask
}

function cropMask(image: DecodedImage, mask: Uint8Array): { left: number; top: number; right: number; bottom: number } | null {
  let left = image.width
  let top = image.height
  let right = -1
  let bottom = -1
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      if (mask[y * image.width + x]) {
        left = Math.min(left, x)
        top = Math.min(top, y)
        right = Math.max(right, x)
        bottom = Math.max(bottom, y)
      }
    }
  }
  return right >= 0 ? { left, top, right: right + 1, bottom: bottom + 1 } : null
}

function normalizeDigit(mask: Uint8Array, width: number, height: number, box: { left: number; top: number; right: number; bottom: number }): Tensor28 {
  const cw = box.right - box.left
  const ch = box.bottom - box.top
  const scale = 20 / Math.max(cw, ch)
  const rw = Math.max(1, Math.round(cw * scale))
  const rh = Math.max(1, Math.round(ch * scale))
  const canvas = Array.from({ length: 28 }, () => Array<number>(28).fill(0))
  const left = Math.floor((28 - rw) / 2)
  const top = Math.floor((28 - rh) / 2)
  for (let y = 0; y < rh; y++) {
    const sourceY = box.top + Math.min(ch - 1, Math.floor(y / scale))
    for (let x = 0; x < rw; x++) {
      const sourceX = box.left + Math.min(cw - 1, Math.floor(x / scale))
      if (mask[sourceY * width + sourceX]) canvas[top + y][left + x] = 1
    }
  }
  return canvas
}

export async function segmentImage(image: DecodedImage): Promise<{ inputs: Tensor28[] }> {
  const mask = foregroundMask(image)
  const box = cropMask(image, mask)
  if (!box) throw Object.assign(new Error('La imagen no contiene trazos detectables.'), { status: 400 })

  const occupied = new Uint8Array(box.right - box.left)
  for (let x = box.left; x < box.right; x++) {
    for (let y = box.top; y < box.bottom; y++) {
      if (mask[y * image.width + x]) occupied[x - box.left] = 1
    }
  }

  let split: number | null = null
  let largestGap = 0
  let gapStart = 0
  for (let x = 1; x < occupied.length; x++) {
    if (occupied[x] && occupied[x - 1] === 0) gapStart = x
  }
  let run = 0
  for (let x = 0; x < occupied.length; x++) {
    if (!occupied[x]) {
      run++
      if (run > largestGap) {
        largestGap = run
        split = box.left + x - Math.floor(run / 2)
      }
    } else {
      run = 0
    }
  }
  void gapStart
  const shouldSplit = largestGap >= Math.max(3, (box.right - box.left) / 30)
  if (shouldSplit && split !== null) {
    const leftBox = { left: box.left, top: box.top, right: split, bottom: box.bottom }
    const rightBox = { left: split, top: box.top, right: box.right, bottom: box.bottom }
    if (leftBox.right - leftBox.left >= 2 && rightBox.right - rightBox.left >= 2) {
      return { inputs: [
        normalizeDigit(mask, image.width, image.height, leftBox),
        normalizeDigit(mask, image.width, image.height, rightBox),
      ] }
    }
  }
  return { inputs: [normalizeDigit(mask, image.width, image.height, box)] }
}

async function pngFromMask(image: DecodedImage, mask: Uint8Array): Promise<string> {
  const bytes = Buffer.alloc(image.width * image.height)
  for (let i = 0; i < mask.length; i++) bytes[i] = mask[i] ? 255 : 0
  return `data:image/png;base64,${(await sharp(bytes, {
    raw: { width: image.width, height: image.height, channels: 1 },
  }).png().toBuffer()).toString('base64')}`
}

export async function preprocessingPreview(
  image: DecodedImage,
  inputs: Tensor28[],
): Promise<Omit<RecognitionResponse, 'number' | 'confidence' | 'probabilities'>> {
  const mask = foregroundMask(image)
  const box = cropMask(image, mask)
  const grayscale = await sharp(Buffer.from(image.pixels), {
    raw: { width: image.width, height: image.height, channels: 1 },
  }).png().toBuffer()
  const binary = await pngFromMask(image, mask)
  let cropped = binary
  if (box) {
    const croppedBuffer = await sharp(Buffer.from(image.pixels), {
      raw: { width: image.width, height: image.height, channels: 1 },
    }).extract({
      left: box.left,
      top: box.top,
      width: box.right - box.left,
      height: box.bottom - box.top,
    }).png().toBuffer()
    cropped = `data:image/png;base64,${croppedBuffer.toString('base64')}`
  }
  const width = 28 * inputs.length
  const output = Buffer.alloc(width * 28)
  inputs.forEach((input, index) => {
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        output[y * width + index * 28 + x] = Math.round(input[y][x] * 255)
      }
    }
  })
  const processedBuffer = await sharp(output, {
    raw: { width, height: 28, channels: 1 },
  }).resize(width * 8, 224, { kernel: 'nearest' }).png().toBuffer()
  return {
    grayscale: `data:image/png;base64,${grayscale.toString('base64')}`,
    binary,
    cropped,
    processed: `data:image/png;base64,${processedBuffer.toString('base64')}`,
  }
}
