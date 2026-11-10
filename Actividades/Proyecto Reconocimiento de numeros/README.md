# Reconocimiento de números manuscritos del 0 al 99

Proyecto local para entrenar un clasificador de dígitos manuscritos con
TensorFlow/Keras y usarlo para reconocer números de uno o dos dígitos. El
modelo se entrena con MNIST y utiliza esta arquitectura:

```text
Input(784) -> Dense(64, relu) -> Dense(64, relu) -> Dense(10, softmax)
```

## Limitación de MNIST y estrategia para 0–99

MNIST contiene imágenes de **un solo dígito**, por lo que el modelo aprende
clases individuales del 0 al 9; no aprende directamente imágenes de números
como `42`.

Para reconocer números de hasta dos dígitos, la aplicación:

1. Detecta el área ocupada por los trazos.
2. Segmenta la imagen en uno o dos dígitos mediante proyección vertical.
3. Preprocesa cada región como una imagen MNIST de 28×28.
4. Predice cada región con el mismo clasificador.
5. Interpreta el primer resultado como decena y el segundo como unidad.
6. Combina ambos resultados como `decenas * 10 + unidades`.

Una imagen con un solo dígito representa un número del 0 al 9: se usa
decena `0` y el dígito reconocido como unidad. Una imagen con dos dígitos
representa un número del 10 al 99, incluyendo casos como `07`, que se
interpretan numéricamente como `7`.

## Requisitos

- Python 3.10 o posterior recomendado.
- TensorFlow/Keras.
- NumPy, Pillow, scikit-learn y Matplotlib.
- Gradio para la interfaz web local.
- Las dependencias están declaradas en `requirements.txt`.

El entrenamiento descarga MNIST únicamente cuando se ejecuta
`src/train.py`. No es necesario descargar ni incluir el dataset manualmente.

## Instalación en Windows

### PowerShell

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, puede ejecutarse una vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Símbolo del sistema (cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
(para salir del entorno)
deactivate
```

## Entrenamiento

El comando solicitado para entrenar desde la raíz es:

```bash
python src/train.py
```

Las épocas y el tamaño de lote son configurables:

```bash
python src/train.py --epochs 10 --batch-size 128
```

El script:

- carga MNIST;
- aplana cada imagen de 28×28 a 784 elementos;
- normaliza los píxeles entre 0 y 1;
- usa `validation_split=0.1`;
- optimiza con Adam y `sparse_categorical_crossentropy`;
- muestra `loss` y `accuracy` de entrenamiento y validación;
- evalúa `loss` y `accuracy` sobre `x_test` e `y_test`;
- imprime ejemplos de clase real y clase predicha.

Los artefactos se guardan en `models/`:

| Archivo | Descripción |
| --- | --- |
| `mnist_digits.keras` | Modelo Keras entrenado usado por la aplicación |
| `training_history.png` | Curvas de loss y accuracy |
| `confusion_matrix.png` | Matriz de confusión del conjunto de test |
| `confusion_matrix.npy` | Matriz de confusión en formato NumPy |
| `metrics.json` | Métricas de test y parámetros principales |
| `examples/correct_*.png` | Ejemplos clasificados correctamente |
| `examples/incorrect_*.png` | Ejemplos clasificados incorrectamente |

## Ejecutar la aplicación

La aplicación actual está implementada con **Gradio**, no con Streamlit. El
comando real desde la raíz del proyecto es:

```bash
python src/app.py
```

Después de iniciar, Gradio muestra una URL local. La interfaz incluye:

- carga y arrastrar/soltar de imágenes;
- vista previa de la imagen de entrada;
- botón **Reconocer número**;
- resultado numérico entre 0 y 99;
- confianza general de la predicción;
- imagen procesada de los dígitos detectados;
- tabla de probabilidades para las clases 0–9 de cada dígito;
- mensajes explícitos para imágenes vacías, inválidas o no segmentables.

No se documenta `streamlit run src/app.py` porque el archivo actual no es una
aplicación Streamlit y `requirements.txt` instala Gradio. Si se requiere
Streamlit, habría que implementar una interfaz Streamlit separada y añadir
`streamlit` como dependencia.

## Formato recomendado de imágenes

Se aceptan archivos PNG, JPG y JPEG. Para obtener mejores resultados:

- usa una imagen nítida y con resolución suficiente;
- escribe con trazos oscuros sobre fondo claro, o usa un fondo oscuro uniforme;
- deja separación horizontal visible entre dos dígitos;
- evita sombras, ruido, bordes y objetos adicionales;
- incluye uno o dos dígitos completos dentro de la imagen;
- para pruebas, una imagen cuadrada o aproximadamente rectangular funciona
  mejor que una fotografía inclinada.

El preprocesamiento convierte la imagen a escala de grises, detecta la
polaridad del fondo, elimina el fondo innecesario, conserva la proporción,
agrega padding para formar un cuadrado y normaliza la imagen final a una
entrada `(1, 784)` compatible con Keras.

## Ejemplos de uso

### Un dígito

Carga una imagen con `7`. El modelo predice una clase individual, por ejemplo:

```text
decenas = 0
unidades = 7
resultado = 0 * 10 + 7 = 7
```

### Dos dígitos

Carga una imagen con `42`. La aplicación segmenta las dos regiones y combina
sus predicciones:

```text
decenas = 4
unidades = 2
resultado = 4 * 10 + 2 = 42
```

La API de predicción se encuentra en `src/predict.py` y devuelve el número,
las decenas, las unidades, la confianza de cada dígito, la confianza general y
las probabilidades de las diez clases.

## Estructura del proyecto

```text
.
├── app.py                  # Wrapper de compatibilidad para la entrada antigua
├── src/
│   ├── __init__.py
│   ├── app.py              # Interfaz Gradio: python src/app.py
│   ├── predict.py          # Carga del modelo y predicción 0–99
│   ├── preprocess.py       # Grises, detección, segmentación y normalización
│   └── train.py            # Entrenamiento y evaluación MNIST
├── data/                   # Datos locales opcionales
├── models/                 # Modelo y artefactos generados
├── notebooks/              # Espacio para exploraciones reproducibles
├── requirements.txt
└── README.md
```

## Validación y errores comunes

Antes de instalar las dependencias, puede comprobarse la sintaxis:

```bash
python -m py_compile src/train.py src/preprocess.py src/predict.py src/app.py
```

Si falta `numpy`, TensorFlow, Pillow o Gradio, instala de nuevo:

```bash
python -m pip install -r requirements.txt
```

Si la aplicación indica que no existe `models/mnist_digits.keras`, ejecuta
primero:

```bash
python src/train.py
```

Si no se detectan trazos, la imagen está vacía, es demasiado pequeña o tiene
más de dos regiones, utiliza una imagen con fondo uniforme, dígitos visibles
y separación horizontal clara. El entrenamiento completo no se ejecuta como
parte de la documentación; debe iniciarse explícitamente con el comando
anterior.
