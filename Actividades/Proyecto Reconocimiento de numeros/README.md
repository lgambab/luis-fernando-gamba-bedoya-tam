# Reconocimiento de números manuscritos del 0 al 99

Aplicación React + Express para comparar dos redes neuronales reales entrenadas
sobre MNIST:

- Red densa MLP: `784 -> Dense(64) -> Dense(64) -> Dense(10)`.
- Red convolucional CNN:
  `28x28x1 -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Dense(64) -> Dense(10)`.

Ambos modelos clasifican dígitos individuales del 0 al 9. El preprocesamiento
detecta uno o dos trazos manuscritos y combina sus resultados para reconocer
números del 0 al 99.

## Requisitos

- Windows
- Node.js LTS
- Python 3.10 o posterior para reentrenar la CNN
- TensorFlow y NumPy para el entrenamiento opcional

## Instalación

Desde la raíz del proyecto:

```powershell
npm install
```

Para reentrenar la CNN, instala las dependencias de Python:

```powershell
py -3.11 -m pip install tensorflow numpy
```

## Ejecutar

Inicia simultáneamente el frontend y el backend:

```powershell
npm start
```

Abre la aplicación en:

```text
http://localhost:5173
```

El backend Express se ejecuta en `http://localhost:3000`.

Para iniciarlos por separado:

```powershell
npm run server
npm run dev -- --host 127.0.0.1
```

## Interfaz

La página incluye dos pestañas:

- **Red Densa (MLP)**: modelo denso conservado para comparación.
- **Red Convolucional (CNN)**: modelo convolucional con inferencia TypeScript.

Cada sección permite cargar una imagen PNG, JPG o JPEG mediante selección,
arrastrar y soltar o portapapeles. Muestra el número reconocido, la confianza,
las vistas de preprocesamiento y las probabilidades de las clases 0–9.

## Endpoints

```text
POST /api/recognize-dense
POST /api/recognize-cnn
```

Cuerpo esperado:

```json
{
  "imageDataUrl": "data:image/png;base64,..."
}
```

La respuesta incluye `number`, `confidence`, `grayscale`, `binary`, `cropped`,
`processed` y `probabilities`.

## Estructura

```text
.
├── index.html
├── package.json
├── server.ts
├── server/
│   ├── preprocess.ts
│   ├── predict.ts
│   └── predictCnn.ts
├── scripts/
│   └── train_cnn.py
├── models/
│   ├── cnn_weights.json
│   ├── cnn_metrics.json
│   ├── dense_weights.json
│   └── mnist_cnn.keras
└── src/
    ├── App.tsx
    ├── App.css
    ├── index.css
    └── main.tsx
```

## Entrenar nuevamente la CNN

```powershell
npm run train:cnn
```

La CNN se entrena sobre MNIST, alcanza métricas reales de test y exporta sus
pesos a `models/cnn_weights.json`. Para usar otro conjunto de pesos, reemplaza
ese archivo y reinicia el servidor.

## Validación

```powershell
npm run lint
npm run build
```
