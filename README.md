# EEG.py - Procesamiento y Clasificación de Señales EEG Sintéticas

Este módulo implementa un flujo de trabajo de principio a fin para experimentos controlados de **clasificación binaria de imaginación motora** utilizando datos sintéticos tipo EEG.

---

## Flujo General del Script

El script ejecuta de forma automática los siguientes 6 pasos:

1. **Generación:** Crea señales temporales con componentes oscilatorias y ruido $1/f$.
2. **Dataset:** Construye un conjunto de datos etiquetado para dos clases (0 y 1).
3. **Extracción:** Obtiene representaciones tiempo-frecuencia mediante **STFT** y **CWT**.
4. **Normalización:** Normaliza las representaciones usando exclusivamente estadísticas de entrenamiento.
5. **Entrenamiento:** Entrena una **CNN multimodal** con ramas separadas para STFT y CWT.
6. **Evaluación:** Evalúa la corrida y exporta métricas individuales y consolidadas.

---

## Convenciones de Datos

Para mantener la consistencia en el flujo, el código utiliza las siguientes estructuras y formas de tensores:


| Variable | Tipo de Dato | Forma / Descripción |
| :--- | :--- | :--- |
| **`X_raw`** | `ndarray float32` | `(n_samples, n_points)` - Señales crudas en el dominio temporal. |
| **`y`** | `ndarray int32` | `(n_samples,)` - Etiquetas binarias para las dos clases (0 y 1). |
| **`X_stft` / `X_cwt`** | `ndarrays float32` | Tensores con canal explícito listos para la entrada del modelo. |
| **`stats`** | `dict` | Estadísticas min-max por rama para evitar la fuga de información (*data leakage*). |

---

## Archivos Generados

Al finalizar la ejecución, el script exporta los siguientes archivos de resultados y métricas:

* **Historial de entrenamiento:** `run_*_training_history.csv`
* **Resumen de entrenamiento:** `run_*_training_summary.json`
* **Matriz de confusión:** `run_*_confusion_matrix.csv`
* **Consolidado multi-semilla:** `multi_seed_*_summary.csv`
* **Métricas multi-semilla:** `multi_seed_*_summary.json`

---

## Notas Importantes de Diseño

* **Validación rigurosa:** La partición de entrenamiento / validación / prueba se realiza estrictamente **antes** de la normalización.
* **Fuga de información:** La normalización de los conjuntos de validación y prueba reutiliza las estadísticas calculadas en el conjunto de entrenamiento.
* **Estabilidad:** El script está diseñado para permitir la ejecución de múltiples semillas (seeds) para medir la estabilidad real del modelo.

