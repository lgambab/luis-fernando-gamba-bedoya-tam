import { useRef, useState, type DragEvent } from 'react'

type ModelType = 'dense' | 'cnn'

interface RecognitionResponse {
  number: number
  confidence: string
  grayscale: string
  binary: string
  cropped: string
  processed: string
  probabilities: [string, string][]
}

function Preview({ title, source }: { title: string; source?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <p className="mb-2 text-sm font-semibold text-slate-700">{title}</p>
      <div className="flex min-h-40 items-center justify-center rounded-lg bg-slate-50">
        {source
          ? <img src={source} alt={title} className="max-h-52 max-w-full object-contain" />
          : <span className="text-sm text-slate-400">Sin imagen</span>}
      </div>
    </div>
  )
}

function RecognitionSection({ model }: { model: ModelType }) {
  const isCnn = model === 'cnn'
  const [image, setImage] = useState<string>()
  const [result, setResult] = useState<RecognitionResponse>()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const acceptFile = (file: File) => {
    if (!/^image\/(png|jpeg|jpg)$/i.test(file.type)) {
      setError('Formato inválido. Use PNG, JPG o JPEG.')
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      setImage(String(reader.result))
      setResult(undefined)
      setError('')
    }
    reader.readAsDataURL(file)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) acceptFile(file)
  }

  const pasteFromClipboard = async () => {
    try {
      const items = await navigator.clipboard.read()
      for (const item of items) {
        const type = item.types.find((value) => /^image\/(png|jpeg|jpg)$/.test(value))
        if (type) {
          acceptFile(new File([await item.getType(type)], 'clipboard-image', { type }))
          return
        }
      }
      setError('El portapapeles no contiene una imagen PNG, JPG o JPEG.')
    } catch {
      setError('No fue posible leer el portapapeles.')
    }
  }

  const recognize = async () => {
    if (!image) {
      setError('Carga una imagen antes de reconocerla.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const response = await fetch(isCnn ? '/api/recognize-cnn' : '/api/recognize-dense', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageDataUrl: image }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Error al reconocer la imagen.')
      setResult(data)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Error al reconocer la imagen.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-xl font-bold text-slate-900">
        {isCnn ? 'Red Convolucional (CNN)' : 'Red Densa (MLP)'}
      </h2>
      <p className="mt-1 mb-5 text-sm text-slate-600">
        {isCnn
          ? 'Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(64) → Dense(10).'
          : 'Entrada plana 784 → Dense(64, ReLU) → Dense(64, ReLU) → Dense(10, Softmax).'}
      </p>

      <div
        className="rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-5 text-center"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        {image
          ? <img src={image} alt="Imagen seleccionada" className="mx-auto mb-4 max-h-56 rounded-lg" />
          : <p className="mb-4 text-sm text-slate-500">Arrastra una imagen aquí</p>}
        <div className="flex flex-wrap justify-center gap-2">
          <button type="button" className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white" onClick={() => fileRef.current?.click()}>
            Seleccionar archivo
          </button>
          <button type="button" className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700" onClick={pasteFromClipboard}>
            Pegar desde el portapapeles
          </button>
        </div>
        <input ref={fileRef} type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" className="hidden" onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) acceptFile(file)
        }} />
      </div>

      {error && <p className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      <button type="button" disabled={loading} onClick={recognize} className="mt-4 w-full rounded-lg bg-blue-600 px-4 py-3 font-bold text-white disabled:opacity-50">
        {loading ? 'Reconociendo...' : isCnn ? 'Reconocer número (CNN)' : 'Reconocer número (MLP)'}
      </button>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-slate-900 px-4 py-4 text-white">
          <p className="text-xs uppercase tracking-wide text-slate-400">Número reconocido</p>
          <p className="mt-1 text-3xl font-bold">{result?.number ?? '—'}</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
          <p className="text-xs uppercase tracking-wide text-slate-500">Confianza</p>
          <p className="mt-1 text-3xl font-bold text-slate-900">{result?.confidence ?? '—'}</p>
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Preview title="Escala de grises" source={result?.grayscale} />
        <Preview title="Blanco y negro" source={result?.binary} />
        <Preview title="Trazos recortados" source={result?.cropped} />
        <Preview title="Entrada final del modelo (28×28)" source={result?.processed} />
      </div>

      {result && (
        <div className="mt-5 overflow-hidden rounded-xl border border-slate-200">
          <table className="w-full text-sm">
            <thead className="bg-slate-100 text-left text-slate-700">
              <tr><th className="px-4 py-3">Clase</th><th className="px-4 py-3">Probabilidad</th></tr>
            </thead>
            <tbody>
              {result.probabilities.map(([label, probability]) => (
                <tr key={label} className="border-t border-slate-100">
                  <td className="px-4 py-2 text-slate-700">{label}</td>
                  <td className="px-4 py-2 font-mono text-slate-900">{probability}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default function App() {
  const [active, setActive] = useState<ModelType>('dense')
  return (
    <main className="min-h-screen bg-slate-100 px-4 py-8">
      <div className="mx-auto max-w-6xl">
        <h1 className="text-center text-3xl font-bold text-slate-900">Reconocimiento de números del 0 al 99</h1>
        <div className="mx-auto mt-6 mb-5 flex max-w-md rounded-xl border border-slate-200 bg-white p-1">
          <button type="button" onClick={() => setActive('dense')} className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold ${active === 'dense' ? 'bg-slate-900 text-white' : 'text-slate-600'}`}>Red Densa (MLP)</button>
          <button type="button" onClick={() => setActive('cnn')} className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold ${active === 'cnn' ? 'bg-blue-600 text-white' : 'text-slate-600'}`}>Red Convolucional (CNN)</button>
        </div>
        {active === 'dense' ? <RecognitionSection model="dense" /> : <RecognitionSection model="cnn" />}
      </div>
    </main>
  )
}
