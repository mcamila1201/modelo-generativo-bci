"""EEG.py
Generación, procesamiento y clasificación de señales EEG sintéticas.

Resumen general:
    Este módulo implementa un flujo de trabajo de principio a fin para experimentos
    controlados de clasificación binaria de imaginación motora con datos
    sintéticos tipo EEG.

    El flujo general del script es:
        1. Generar señales temporales con componentes oscilatorias y ruido 1/f.
        2. Construir un conjunto de datos etiquetado para dos clases.
        3. Extraer representaciones tiempo-frecuencia con STFT y CWT.
        4. Normalizar las representaciones usando solo estadísticas de entrenamiento.
        5. Entrenar una CNN multimodal con ramas separadas para STFT y CWT.
        6. Evaluar la corrida y exportar métricas individuales y consolidadas.

Convenciones de datos:
    X_raw                       ndarray float32 con forma (n_samples, n_points)
                                Señales crudas en el dominio temporal.
    y                           ndarray int32 con forma (n_samples,)
                                Etiquetas binarias: 0 y 1.
    X_stft / X_cwt              ndarrays float32 con canal explícito
                                Tensores listos para entrar al modelo.
    stats                       dict
                                Estadísticas min-max por rama para evitar fuga de información.

Archivos generados:
    run_*_training_history.csv
    run_*_training_summary.json
    run_*_confusion_matrix.csv
    multi_seed_*_summary.csv
    multi_seed_*_summary.json

Notas:
    - La partición de entrenamiento/validación/prueba se hace antes de la normalización.
    - La normalización de validación y prueba reutiliza estadísticas de entrenamiento.
    - El script permite ejecutar múltiples semillas para medir estabilidad.
"""

# %%
import numpy as np                                                                                  # Operaciones numéricas base y manejo de arreglos.
import matplotlib.pyplot as plt                                                                     # Visualización de señales y mapas tiempo-frecuencia.
import pywt                                                                                         # Transformada wavelet continua.
import tensorflow as tf                                                                             # Entrenamiento y utilidades del modelo profundo.
import random                                                                                       # Semilla del generador pseudoaleatorio de Python.
import csv                                                                                          # Exportación tabular de métricas.
import json                                                                                         # Exportación estructurada de resúmenes.
from datetime import datetime                                                                       # Sellos de tiempo para nombrar artefactos.
from tensorflow.keras import layers, models                                                         # Capas y constructor funcional de Keras.
from scipy import signal                                                                            # Espectrograma STFT.
from sklearn.model_selection import train_test_split                                                # Particiones estratificadas reproducibles.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score               # Métricas de clasificación.
from pathlib import Path                                                                            # Manejo portable de rutas de salida.

# ============================================================================
# Sintesis de senales EEG
# ============================================================================

def vector_tem(fs, duration):
    """vector_tem
    Genera el vector temporal uniforme usado por todas las señales del experimento.

    Entradas:
        fs                      int             Frecuencia de muestreo en Hz.
        duration                float           Duración total del ensayo en segundos.

    Salidas:
        t                       ndarray         Vector temporal con `fs * duration` muestras.

    Notas:
        Se usa `endpoint=False` para evitar duplicar la muestra final cuando el
        vector se interpreta como una rejilla temporal periódica.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)                                # Rejilla temporal uniforme.
    return t                                                                                        # Devuelve el eje temporal del trial.

def gen_envolvente(t):
    """gen_envolvente
    Genera una envolvente temporal aleatoria para simular un patrón ERD/ERS.

    Entradas:
        t                       ndarray         Vector temporal del ensayo.

    Salidas:
        h_t                     ndarray         Envolvente multiplicativa aplicada a la señal oscilatoria.

    Notas:
        - La primera gaussiana modela una caída de potencia tipo ERD.
        - La segunda gaussiana modela un rebote corto tipo ERS.
        - Los parámetros se aleatorizan para introducir variabilidad entre ensayos.
    """
    t0    = np.random.uniform(0.5, 1.5)                                                             # Centro temporal del evento principal.
    sigma = np.random.uniform(0.5, 1)                                                               # Anchura del evento ERD.

    h     = np.exp(-(t - t0)**2 / (2 * sigma**2))                                                   # Campana principal del ERD.
    h_t   = 1 - 0.6 * h                                                                             # Atenúa amplitud alrededor del evento.
    h_t  += 0.3 * np.exp(-(t - (t0 + 1.5))**2 / (2 * 0.2**2))                                       # Añade rebote ERS más localizado.

    return h_t                                                                                      # Devuelve la envolvente temporal final.

def gen_osc_signal(t, f_k, M, label, h_t):
    """gen_osc_signal
    Construye la componente oscilatoria base de una señal EEG sintética.

    Entradas:
        t                       ndarray         Vector temporal del ensayo.
        f_k                     array-like      Frecuencias fundamentales de las bandas simuladas.
        M                       int             Número de armónicos por frecuencia fundamental.
        label                   int             Clase binaria del ensayo.
        h_t                     ndarray         Envolvente ERD/ERS usada para modular amplitud.

    Salidas:
        osc_signal              ndarray         Señal oscilatoria compuesta con modulación por clase.

    Notas:
        Se generan contribuciones en bandas mu y beta. El parámetro `alpha`
        cambia según la clase y la banda, lo que introduce diferencias
        discriminantes entre los trials.
    """
    osc_signal = np.zeros_like(t)                                                                   # Acumulador de la componente oscilatoria.

    for k in range(len(f_k)):                                                                       # Recorre cada frecuencia base definida para el trial.

        f = f_k[k]                                                                                  # Frecuencia base actual dentro del conjunto de bandas simuladas.

        # `alpha` controla cuánto afecta la envolvente a la amplitud según banda y clase.
        if f < 15:                                                                                  # Banda mu.
            if label == 0:                                                                          # Clase derecha.
                alpha = np.random.normal(-0.5, 0.25)                                                # ERD más fuerte para la clase 0.
            else:                                                                                   # Clase izquierda.
                alpha = np.random.normal(-0.3, 0.25)                                                # ERD algo menos intenso para la clase 1.
        else:                                                                                       # Banda beta.
            if label == 0:                                                                          # Clase derecha.
                alpha = np.random.normal(-0.2, 0.25)                                                # Modulación beta moderada para clase 0.
            else:                                                                                   # Clase izquierda.
                alpha = np.random.normal(-0.5,  0.25)                                               # Modulación beta más intensa para clase 1.

        a_global = np.random.uniform(3, 8)                                                          # Amplitud base compartida dentro de la banda.
        phi      = np.random.uniform(0, 2 * np.pi)                                                  # Fase común para conservar coherencia intrabanda.

        for m in range(1, M + 1):                                                                   # Suma armónicos de la frecuencia base actual.
            a_base      = a_global / (m**2)                                                         # Atenuación cuadrática de armónicos.
            A_t         = a_base * (1 + alpha * h_t)                                                # Modulación temporal dependiente de clase.
            osc_signal += A_t * np.sin(2 * np.pi * m * f * t + phi)                                 # Acumula el armónico actual en la señal compuesta.

    epsilon = np.random.normal(0, 0.0025, len(t))                                                   # Perturbación aleatoria de baja amplitud.
    epsilon = np.convolve(epsilon, np.ones(50)/50, mode='same')                                     # Suaviza el ruido fino para evitar picos abruptos.

    return osc_signal + epsilon                                                                     # Devuelve la componente oscilatoria con perturbación suavizada.

def gen_ruido(n_puntos, fs):
    """gen_ruido
    Genera ruido coloreado con pendiente espectral aproximada 1/f^beta.

    Entradas:
        n_puntos                int             Número de muestras del ensayo.
        fs                      int             Frecuencia de muestreo en Hz.

    Salidas:
        ruido_norm              ndarray         Ruido temporal coloreado y reescalado.

    Notas:
        El ruido se construye en frecuencia ponderando una FFT de ruido blanco y
        regresando al dominio temporal con la transformada inversa real.
    """
    beta         = np.random.uniform(1, 1.5)                                                        # Exponente espectral del ruido coloreado.
    sigma        = np.random.uniform(0.2, 0.8)                                                      # Escala global de amplitud del ruido.
    
    ruido_blanco = np.random.normal(0, 1, n_puntos)                                                 # Fuente base de ruido blanco gaussiano.

    fft_ruido    = np.fft.rfft(ruido_blanco)                                                        # Espectro real de la señal de ruido.
    frq          = np.fft.rfftfreq(n_puntos, d=1/fs)                                                # Frecuencias asociadas a la FFT real.

    frq[0]       = frq[1]                                                                           # Evita singularidad en la componente DC.
    frq          = np.maximum(frq, 0.5)                                                             # Recorta frecuencias muy pequeñas para estabilidad.

    filtro_exp   = 1 / (frq ** (beta / 2))                                                          # Pendiente espectral objetivo.
    weighted_fft = fft_ruido * filtro_exp                                                           # Aplica el coloreado en frecuencia.

    ruido        = np.fft.irfft(weighted_fft, n=n_puntos)                                           # Regresa el ruido al dominio temporal.
    ruido_norm  = sigma * (ruido / np.std(ruido))                                                   # Reescala por una desviación aleatoria controlada.

    return ruido_norm                                                                               # Devuelve el ruido coloreado final.

def gen_eeg(t, fs, f_k, M, label):
    """gen_eeg
    Genera una señal EEG sintética combinando dinámica oscilatoria y ruido coloreado.

    Entradas:
        t                       ndarray         Vector temporal del ensayo.
        fs                      int             Frecuencia de muestreo en Hz.
        f_k                     array-like      Frecuencias fundamentales de la señal oscilatoria.
        M                       int             Número de armónicos por frecuencia.
        label                   int             Etiqueta binaria del ensayo.

    Salidas:
        x_t                     ndarray         Señal EEG sintética final en el dominio temporal.
    """
    h_t        = gen_envolvente(t)                                                                  # Envolvente temporal ERD/ERS para este trial.
    osc_signal = gen_osc_signal(t, f_k, M, label, h_t)                                              # Componente rítmica discriminante.
    ruido      = gen_ruido(len(t), fs)                                                              # Componente de ruido coloreado.
    x_t        = osc_signal + ruido                                                                 # Mezcla final observada.

    return x_t                                                                                      # Devuelve la señal EEG sintética.


def gen_dataset(n_samples, fs, duration):
    """gen_dataset
    Genera un conjunto de ensayos EEG sintéticos para clasificación binaria.

    Entradas:
        n_samples               int             Número total de ensayos a sintetizar.
        fs                      int             Frecuencia de muestreo en Hz.
        duration                float           Duración de cada ensayo en segundos.

    Salidas:
        X                       ndarray         Matriz de señales crudas con forma `(n_samples, n_points)`.
        y                       ndarray         Vector de etiquetas binarias.
        t                       ndarray         Vector temporal compartido por todos los trials.
    """
    t = vector_tem(fs, duration)                                                                    # Eje temporal común a todo el dataset.
    M = 2                                                                                           # Número de armónicos por frecuencia base.

    X = np.empty((n_samples, len(t)), dtype=np.float32)                                             # Reserva matriz de señales crudas.
    y = np.empty(n_samples, dtype=np.int32)                                                         # Reserva vector de etiquetas.

    for idx in range(n_samples):                                                                    # Genera y etiqueta cada trial del dataset.

        label      = np.random.choice([0, 1])                                                       # Etiqueta binaria del trial actual.

        f_k        = np.array([
            np.random.uniform(8, 12),                                                               # Banda mu baja.
            np.random.uniform(10, 14),                                                              # Banda mu alta.
            np.random.uniform(18, 24),                                                              # Banda beta baja.
            np.random.uniform(20, 26)                                                               # Banda beta alta.
        ])                                                                                          # Variabilidad intra-banda entre trials.

        eeg_signal = gen_eeg(t, fs, f_k, M, label)                                                  # Señal sintética del trial.

        X[idx]     = eeg_signal.astype(np.float32)                                                  # Almacena señal en formato compacto.
        y[idx]     = label                                                                          # Guarda etiqueta asociada.

    return X, y, t                                                                                  # Devuelve señales, etiquetas y eje temporal compartido.


# ============================================================================
# Visualizacion y utilidades generales
# ============================================================================

def plot_sample(signal, t, label):
    """plot_sample
    Grafica una señal EEG sintética de ejemplo en el dominio temporal.

    Entradas:
        signal                  ndarray         Señal EEG a visualizar.
        t                       ndarray         Vector temporal asociado.
        label                   int             Etiqueta binaria del trial.
    """
    plt.figure(figsize=(12,4))                                                                      # Crea la figura principal.
    plt.plot(t, signal)                                                                             # Dibuja amplitud contra tiempo.
    plt.title(f"EEG Sintético - Clase: {'Izquierda' if label else 'Derecha'}")                      # Título descriptivo por clase.
    plt.xlabel("Tiempo (s)")                                                                        # Etiqueta del eje x.
    plt.ylabel("Amplitud (µV)")                                                                     # Etiqueta del eje y.
    plt.grid()                                                                                      # Facilita inspección visual.
    plt.show()                                                                                      # Muestra la figura.


def set_seed(seed=42):
    """set_seed
    Fija semillas de Python, NumPy y TensorFlow para mejorar reproducibilidad.

    Entradas:
        seed                    int             Semilla base de la corrida.
    """
    random.seed(seed)                                                                               # Generador pseudoaleatorio de Python.
    np.random.seed(seed)                                                                            # Generador de NumPy.
    tf.random.set_seed(seed)                                                                        # Generador de TensorFlow/Keras.


# ============================================================================
# Exportacion y evaluacion
# ============================================================================

def exportar_resultados(history, eval_metrics, eval_summary, output_dir, config):
    """exportar_resultados
    Exporta los resultados de una corrida individual a archivos CSV y JSON.

    Entradas:
        history                 keras.callbacks.History    Historial devuelto por `model.fit()`.
        eval_metrics            sequence[float]            Métricas escalares de `model.evaluate()`.
        eval_summary            dict                       Métricas derivadas y matriz de confusión.
        output_dir              pathlib.Path               Directorio de salida.
        config                  dict                       Configuración relevante de la corrida.

    Salidas:
        csv_path                pathlib.Path               Archivo con historial por época.
        json_path               pathlib.Path               Archivo con resumen global de la corrida.
        confusion_csv_path      pathlib.Path               Archivo con la matriz de confusión.
        summary                 dict                       Resumen consolidado de la corrida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)                                                   # Asegura que exista el directorio de salida.

    history_dict       = history.history                                                            # Convierte el historial a un diccionario serializable.
    run_id             = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{config['seed']}"     # Identificador único por corrida.
    csv_path           = output_dir / f"{run_id}_training_history.csv"                              # Ruta del historial por época.
    json_path          = output_dir / f"{run_id}_training_summary.json"                             # Ruta del resumen general.
    confusion_csv_path = output_dir / f"{run_id}_confusion_matrix.csv"                              # Ruta de la matriz de confusión.

    history_keys       = list(history_dict.keys())                                             # Nombres de las series registradas por Keras.
    n_epochs           = len(history_dict[history_keys[0]])                                         # Número efectivo de épocas ejecutadas.

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:                              # Abre el CSV donde se guardará el historial por época.
        writer = csv.writer(csv_file)                                                               # Escritor tabular del historial.
        writer.writerow(["epoch", *history_keys])                                                   # Encabezado del CSV.
        for epoch_idx in range(n_epochs):                                                           # Recorre todas las épocas registradas por Keras.
            writer.writerow(                                                                        # Escribe una fila con las métricas de la época actual.
                [epoch_idx + 1, *[float(history_dict[key][epoch_idx]) for key in history_keys]]
                                                                                                    # Métricas de la época actual.
            )

    best_epoch_idx    = int(np.argmin(history_dict["val_loss"]))                                    # Época con menor pérdida de validación.
    best_val_accuracy = None                                                                        # Valor opcional si Keras reportó accuracy de validación.
    if "val_accuracy" in history_dict:                                                              # Solo existe si Keras registró esa métrica.
        best_val_accuracy = float(history_dict["val_accuracy"][best_epoch_idx])                     # Accuracy correspondiente a la mejor época por `val_loss`.

    summary = {                                                                                     # Resumen serializable de la corrida individual.
        "config": config,                                                                           # Hiperparámetros y metadatos de la ejecución.
        "epochs_ran": n_epochs,                                                                     # Número real de épocas ejecutadas.
        "best_epoch_by_val_loss": best_epoch_idx + 1,                                               # Mejor época según validación.
        "best_val_loss": float(history_dict["val_loss"][best_epoch_idx]),                           # Menor pérdida de validación observada.
        "best_val_accuracy": best_val_accuracy,                                                     # Accuracy de validación asociada a la mejor época.
        "test_loss": float(eval_metrics[0]),                                                        # Pérdida final sobre test.
        "test_accuracy": float(eval_metrics[1]),                                                    # Accuracy final sobre test.
        "precision": float(eval_summary["precision"]),                                              # Precisión binaria en test.
        "recall": float(eval_summary["recall"]),                                                    # Recall binario en test.
        "f1_score": float(eval_summary["f1_score"]),                                                # F1-score binario en test.
        "confusion_matrix": eval_summary["confusion_matrix"],                                       # Matriz de confusión 2x2.
        "run_id": run_id,                                                                           # Identificador único de la corrida.
        "history_csv": str(csv_path),                                                               # Ruta del historial exportado.
        "summary_json": str(json_path),                                                             # Ruta del resumen JSON.
        "confusion_matrix_csv": str(confusion_csv_path),                                            # Ruta del CSV de matriz de confusión.
    }

    with confusion_csv_path.open("w", newline="", encoding="utf-8") as csv_file:                    # Abre el CSV dedicado a la matriz de confusión.
        writer = csv.writer(csv_file)                                                               # Escritor de la matriz de confusión.
        writer.writerow(["", "Pred_0", "Pred_1"])                                                   # Encabezado de columnas.
        writer.writerow(["True_0", *eval_summary["confusion_matrix"][0]])                           # Fila de la clase real 0.
        writer.writerow(["True_1", *eval_summary["confusion_matrix"][1]])                           # Fila de la clase real 1.

    with json_path.open("w", encoding="utf-8") as json_file:                                        # Abre el JSON donde se persistirá el resumen global.
        json.dump(summary, json_file, indent=2)                                                     # Serializa el resumen completo en JSON.

    return csv_path, json_path, confusion_csv_path, summary                                         # Devuelve rutas exportadas y resumen listo para reutilizar.


def exportar_resumen_consolidado(run_summaries, output_dir):
    """exportar_resumen_consolidado
    Exporta un resumen agregado de varias corridas ejecutadas con seeds distintas.

    Entradas:
        run_summaries           list[dict]      Resúmenes individuales de cada corrida.
        output_dir              pathlib.Path    Directorio donde se guardan los consolidados.

    Salidas:
        csv_path                pathlib.Path    Archivo tabular con una fila por corrida.
        json_path               pathlib.Path    Archivo JSON con estadísticas agregadas.
        aggregate_summary       dict            Resumen global con medias, desviaciones y mejor corrida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)                                                   # Asegura la carpeta de salida.

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')                                           # Marca temporal para identificar el consolidado.
    csv_path   = output_dir / f"multi_seed_{timestamp}_summary.csv"                                 # Ruta del consolidado tabular.
    json_path  = output_dir / f"multi_seed_{timestamp}_summary.json"                                # Ruta del consolidado estructurado.

    fieldnames = [                                                                                  # Columnas del CSV consolidado multi-seed.
        "run_id",                                                                                   # Identificador único de la corrida.
        "seed",                                                                                     # Seed usada para reproducibilidad.
        "epochs_ran",                                                                               # Número de épocas ejecutadas.
        "best_epoch_by_val_loss",                                                                   # Mejor época según validación.
        "best_val_loss",                                                                            # Menor pérdida de validación.
        "best_val_accuracy",                                                                        # Accuracy en la mejor época.
        "test_loss",                                                                                # Pérdida final sobre test.
        "test_accuracy",                                                                            # Accuracy final sobre test.
        "precision",                                                                                # Precisión final.
        "recall",                                                                                   # Recall final.
        "f1_score",                                                                                 # F1-score final.
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:                              # Abre el CSV con una fila por corrida.
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)                                    # Escritor de diccionarios para el consolidado.
        writer.writeheader()                                                                        # Escribe encabezados del archivo tabular.
        for summary in run_summaries:                                                               # Recorre cada resumen individual ya exportado.
            writer.writerow(                                                                        # Registra una fila resumida de la corrida actual.
                {
                    "run_id": summary["run_id"],                                                    # Identificador único de la corrida.
                    "seed": summary["config"]["seed"],                                              # Seed usada en esa ejecución.
                    "epochs_ran": summary["epochs_ran"],                                            # Épocas efectivamente ejecutadas.
                    "best_epoch_by_val_loss": summary["best_epoch_by_val_loss"],                    # Mejor época por criterio de validación.
                    "best_val_loss": summary["best_val_loss"],                                      # Mejor pérdida de validación.
                    "best_val_accuracy": summary["best_val_accuracy"],                              # Accuracy de validación asociada.
                    "test_loss": summary["test_loss"],                                              # Pérdida final en test.
                    "test_accuracy": summary["test_accuracy"],                                      # Accuracy final en test.
                    "precision": summary["precision"],                                              # Precisión final en test.
                    "recall": summary["recall"],                                                    # Recall final en test.
                    "f1_score": summary["f1_score"],                                                # F1-score final en test.
                }
            )

    test_accuracies = np.array([summary["test_accuracy"] for summary in run_summaries], dtype=np.float32)
                                                                                                    # Accuracy por corrida.
    f1_scores       = np.array([summary["f1_score"] for summary in run_summaries], dtype=np.float32)# F1-score por corrida.
    best_run        = max(run_summaries, key=lambda summary: summary["test_accuracy"])              # Mejor corrida según accuracy.

    aggregate_summary = {                                                                           # Resumen agregado a través de múltiples seeds.
        "n_runs": len(run_summaries),                                                               # Número total de corridas incluidas.
        "seeds": [summary["config"]["seed"] for summary in run_summaries],                          # Seeds consideradas en el consolidado.
        "mean_test_accuracy": float(np.mean(test_accuracies)),                                      # Accuracy promedio entre corridas.
        "std_test_accuracy": float(np.std(test_accuracies)),                                        # Variabilidad de accuracy entre corridas.
        "mean_f1_score": float(np.mean(f1_scores)),                                                 # F1 promedio entre corridas.
        "std_f1_score": float(np.std(f1_scores)),                                                   # Variabilidad de F1 entre corridas.
        "best_run_by_test_accuracy": {
            "run_id": best_run["run_id"],                                                           # Identificador de la mejor corrida.
            "seed": best_run["config"]["seed"],                                                     # Seed de la mejor corrida.
            "test_accuracy": best_run["test_accuracy"],                                             # Accuracy alcanzada por la mejor corrida.
            "f1_score": best_run["f1_score"],                                                       # F1-score asociado a la mejor corrida.
        },
        "runs": run_summaries,                                                                      # Detalle completo de cada corrida individual.
        "summary_csv": str(csv_path),                                                               # Ruta al CSV consolidado.
    }

    with json_path.open("w", encoding="utf-8") as json_file:                                        # Abre el JSON consolidado multi-seed.
        json.dump(aggregate_summary, json_file, indent=2)                                           # Serializa estadísticas agregadas y detalle de corridas.

    return csv_path, json_path, aggregate_summary                                                   # Devuelve rutas del consolidado y sus estadísticas agregadas.


def evaluar_modelo(model, X_test, y_test):
    """evaluar_modelo
    Evalúa el modelo en prueba y deriva métricas de clasificación binarias.

    Entradas:
        model                   keras.Model     Modelo entrenado y compilado.
        X_test                  list[ndarray]   Entradas de prueba para ambas ramas del modelo.
        y_test                  ndarray         Etiquetas reales del conjunto de prueba.

    Salidas:
        eval_metrics            list[float]     Métricas escalares devueltas por Keras.
        eval_summary            dict            Precisión, recall, F1 y matriz de confusión.
    """
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)                                        # Calcula pérdida y accuracy en test.
    y_prob       = model.predict(X_test, verbose=0).ravel()                                         # Probabilidades predichas por el modelo.
    y_pred       = (y_prob >= 0.5).astype(np.int32)                                                 # Umbral binario estándar en 0.5.

    confusion    = confusion_matrix(y_test, y_pred, labels=[0, 1])                                  # Matriz de confusión ordenada por clases 0 y 1.
    eval_summary = {                                                                                # Métricas derivadas para reporte y exportación.
        "precision": precision_score(y_test, y_pred, zero_division=0),                              # Precisión binaria evitando errores por división entre cero.
        "recall": recall_score(y_test, y_pred, zero_division=0),                                    # Recall binario evitando errores por división entre cero.
        "f1_score": f1_score(y_test, y_pred, zero_division=0),                                      # F1-score balanceado entre precisión y recall.
        "confusion_matrix": confusion.tolist(),                                                     # Matriz convertida a lista para serialización JSON.
    }

    return eval_metrics, eval_summary                                                               # Devuelve métricas escalares y resumen derivado para exportación.


# ============================================================================
# Extraccion de caracteristicas y preparacion de datos
# ============================================================================

def extraer_representaciones(x, fs, escalas):
    """extraer_representaciones
    Obtiene dos vistas tiempo-frecuencia de una misma señal temporal.

    Entradas:
        x                       ndarray         Señal EEG en el dominio temporal.
        fs                      int             Frecuencia de muestreo en Hz.
        escalas                 array-like      Escalas usadas para la CWT.

    Salidas:
        stft_map                ndarray         Espectrograma STFT en dB.
        cwt_map                 ndarray         Magnitud de la CWT con wavelet Morlet.
    """
    _, _, Sxx = signal.spectrogram(x, fs=fs, nperseg=64, noverlap=48)                               # Espectrograma de corta duración.
    stft_map  = (10 * np.log10(Sxx + 1e-10)).astype(np.float32)                                     # Conversión a dB y compactación de tipo.

    coef, _   = pywt.cwt(x, escalas, 'morl', sampling_period=1 / fs)                                # Coeficientes wavelet continuos.
    cwt_map   = np.abs(coef).astype(np.float32)                                                     # Se conserva la magnitud para clasificación.

    return stft_map, cwt_map                                                                        # Devuelve ambas vistas tiempo-frecuencia del mismo trial.


def normalizar_minmax(X, stats=None):
    """normalizar_minmax
    Normaliza un arreglo con escalamiento min-max y devuelve las estadísticas usadas.

    Entradas:
        X                       ndarray         Arreglo a reescalar.
        stats                   dict|None       Estadísticas precomputadas con claves `min` y `max`.

    Salidas:
        X_norm                  ndarray         Arreglo normalizado al rango [0, 1].
        stats_out               dict            Estadísticas efectivamente usadas.
    """
    if stats is None:                                                                               # Calcula estadísticas nuevas cuando no se proporcionan referencias previas.
        x_min = float(X.min())                                                                      # Mínimo observado en el arreglo actual.
        x_max = float(X.max())                                                                      # Máximo observado en el arreglo actual.
    else:                                                                                           # Reutiliza estadísticas ya calculadas en entrenamiento.
        x_min = stats["min"]                                                                        # Reutiliza mínimo de train.
        x_max = stats["max"]                                                                        # Reutiliza máximo de train.

    scale  = max(x_max - x_min, 1e-8)                                                               # Evita división por cero.
    X_norm = ((X - x_min) / scale).astype(np.float32)                                               # Reescala a [0, 1] en float32.

    return X_norm, {"min": x_min, "max": x_max}                                                     # Devuelve datos y estadísticas usadas.


def preparar_data(X_raw, fs, escalas=None, stats=None):
    """preparar_data
    Convierte señales crudas en tensores STFT y CWT listos para la CNN.

    Entradas:
        X_raw                   ndarray         Señales temporales con forma `(n_samples, n_points)`.
        fs                      int             Frecuencia de muestreo en Hz.
        escalas                 array-like|None Escalas para la CWT.
        stats                   dict|None       Estadísticas de entrenamiento para reutilizar en validación y prueba.

    Salidas:
        X_stft                  ndarray         Tensores STFT con canal explícito.
        X_cwt                   ndarray         Tensores CWT con canal explícito.
        stats_out               dict            Estadísticas min-max por rama.
    """
    if escalas is None:                                                                             # Usa un conjunto estándar de escalas cuando no se especifica otro.
        escalas = np.arange(1, 64)                                                                  # Escalas por defecto para la CWT.

    n_samples         = X_raw.shape[0]                                                              # Número de trials a transformar.
    stft_ref, cwt_ref = extraer_representaciones(X_raw[0], fs, escalas)                             # Primer trial para inferir tamaños.

    X_stft            = np.empty((n_samples, *stft_ref.shape), dtype=np.float32)                    # Reserva memoria para la rama STFT.
    X_cwt             = np.empty((n_samples, *cwt_ref.shape), dtype=np.float32)                     # Reserva memoria para la rama CWT.
    X_stft[0]         = stft_ref                                                                    # Inserta representación de referencia.
    X_cwt[0]          = cwt_ref                                                                     # Inserta representación de referencia.

    for idx in range(1, n_samples):                                                                 # Procesa el resto de los trials tras usar el primero como referencia.
        stft_map, cwt_map = extraer_representaciones(X_raw[idx], fs, escalas)                       # Extrae ambas vistas del trial.
        X_stft[idx]       = stft_map                                                                # Guarda STFT del trial.
        X_cwt[idx]        = cwt_map                                                                 # Guarda CWT del trial.

    stft_stats = None if stats is None else stats["stft"]                                           # Estadísticas de train para STFT.
    cwt_stats  = None if stats is None else stats["cwt"]                                            # Estadísticas de train para CWT.

    X_stft, stft_stats = normalizar_minmax(X_stft, stft_stats)                                      # Normalización STFT sin fuga de información.
    X_cwt, cwt_stats   = normalizar_minmax(X_cwt, cwt_stats)                                        # Normalización CWT sin fuga de información.

    return (
        np.expand_dims(X_stft, -1),                                                                 # Añade canal explícito para la rama STFT.
        np.expand_dims(X_cwt, -1),                                                                  # Añade canal explícito para la rama CWT.
        {"stft": stft_stats, "cwt": cwt_stats},                                                     # Devuelve estadísticas para reutilizar en otros splits.
    )


def visualizar_representaciones(X_stft, X_cwt):
    """visualizar_representaciones
    Muestra un ejemplo de las representaciones STFT y CWT del conjunto de datos.

    Entradas:
        X_stft                  ndarray         Tensor de espectrogramas STFT.
        X_cwt                   ndarray         Tensor de escalogramas CWT.
    """
    plt.figure(figsize=(14, 6))                                                                     # Figura con dos paneles comparativos.

    plt.subplot(1, 2, 1)                                                                            # Panel izquierdo: STFT.
    plt.title("STFT (Espectrograma)")                                                               # Título del panel STFT.
    plt.pcolormesh(X_stft[0, :, :, 0], shading='gouraud', cmap='jet')                               # Primer espectrograma del lote.
    plt.ylabel('Frecuencia (Bins)')                                                                 # Eje vertical de STFT.
    plt.xlabel('Tiempo')                                                                            # Eje horizontal de STFT.

    plt.subplot(1, 2, 2)                                                                            # Panel derecho: CWT.
    plt.title("CWT - Wavelet Morlet (Escalograma)")                                                 # Título del panel CWT.
    plt.pcolormesh(X_cwt[0, :, :, 0], shading='gouraud', cmap='magma')                              # Primer escalograma del lote.
    plt.ylabel('Escalas (Frecuencia)')                                                              # Eje vertical de CWT.
    plt.xlabel('Tiempo')                                                                            # Eje horizontal de CWT.

    plt.tight_layout()                                                                              # Ajusta espacios entre subgráficas.
    plt.show()                                                                                      # Muestra la visualización.


# ============================================================================
# Modelo
# ============================================================================

def build_model(stft_shape, cwt_shape, dense_units=32, head_dropout=0.45):
    """build_model
    Construye la CNN multimodal que fusiona representaciones STFT y CWT.

    Entradas:
        stft_shape              tuple           Forma de entrada de la rama STFT.
        cwt_shape               tuple           Forma de entrada de la rama CWT.
        dense_units             int             Número de neuronas en la cabeza de fusión.
        head_dropout            float           Dropout aplicado tras la capa densa de fusión.

    Salidas:
        model                   keras.Model     Modelo compilado para clasificación binaria.
    """
    # Rama STFT: captura patrones locales sobre el espectrograma.
    in_s = layers.Input(shape=stft_shape, name="input_stft")                                        # Entrada dedicada a espectrogramas STFT.
    s    = layers.Conv2D(32, (3,3), activation='elu', padding='same', name="conv_stft")(in_s)       # Filtros locales sobre STFT.
    s    = layers.MaxPooling2D((2,2))(s)                                                            # Reduce resolución espacial.
    s    = layers.Dropout(0.3)(s)                                                                   # Regulariza activaciones de la rama STFT.
    s    = layers.Flatten()(s)                                                                      # Convierte mapas a vector de características.

    # Rama CWT: extrae descriptores de patrones multiescala.
    in_w = layers.Input(shape=cwt_shape, name="input_cwt")                                          # Entrada dedicada a escalogramas CWT.
    w    = layers.Conv2D(32, (3,3), activation='elu', padding='same')(in_w)                         # Primer bloque convolucional de CWT.
    w    = layers.MaxPooling2D((2,4))(w)                                                            # Compresión anisotrópica sobre escalas y tiempo.
    w    = layers.Conv2D(64, (3,3), activation='elu', padding='same')(w)                            # Segundo bloque convolucional de CWT.
    w    = layers.MaxPooling2D((2,4))(w)                                                            # Nueva reducción de dimensionalidad.
    w    = layers.Dropout(0.35)(w)                                                                  # Regularización de la rama CWT.
    w    = layers.Flatten()(w)                                                                      # Vectoriza descriptores aprendidos.

    # Fusión: concatena ambas ramas y produce una predicción binaria.
    merged = layers.Concatenate()([s, w])                                                           # Une descriptores STFT y CWT.
    dense  = layers.Dense(dense_units, activation='elu')(merged)                                    # Capa densa de integración multimodal.
    dense  = layers.Dropout(head_dropout)(dense)                                                    # Regulariza la cabeza clasificadora.

    out    = layers.Dense(1, activation='sigmoid')(dense)                                           # Probabilidad de la clase positiva.

    model  = models.Model(inputs=[in_s, in_w], outputs=out)                                         # Construye el grafo funcional completo.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])               # Configura entrenamiento supervisado.
    
    return model                                                                                    # Devuelve el modelo compilado listo para entrenarse.


# ============================================================================
# Ejecucion de experimentos
# ============================================================================

def run_experimento(seed, n_samples, fs, duration, output_dir, training_config, visualizar=False):
    """run_experimento
    Ejecuta una corrida completa del flujo de trabajo con una semilla específica.

    Entradas:
        seed                    int             Semilla de la corrida.
        n_samples               int             Número total de señales a generar.
        fs                      int             Frecuencia de muestreo en Hz.
        duration                float           Duración de cada ensayo en segundos.
        output_dir              pathlib.Path    Directorio de exportación.
        training_config         dict            Hiperparámetros del entrenamiento.
        visualizar              bool            Si es True, muestra ejemplos gráficos.

    Salidas:
        model                   keras.Model     Modelo entrenado.
        history                 keras.callbacks.History Historial del ajuste.
        summary                 dict            Resumen exportable de la corrida.
    """
    tf.keras.backend.clear_session()                                                                # Limpia estado global de Keras entre corridas.

    set_seed(seed)                                                                                  # Fija semilla antes de cualquier aleatoriedad.

    X_raw, y, t = gen_dataset(n_samples, fs, duration)                                              # Genera dataset sintético completo.
    if visualizar:                                                                                  # Permite inspección visual opcional del primer trial generado.
        plot_sample(X_raw[0], t, y[0])                                                              # Muestra un trial representativo.

    unique, counts = np.unique(y, return_counts=True)                                               # Resume el balance de clases generado.
    for u, c in zip(unique, counts):                                                         # Recorre el conteo de muestras por clase.
        print(f"Seed {seed} | Clase {u}: {c} señales")                                              # Informa el balance de clases observado.

    test_size = 0.2                                                                                 # Fracción total reservada para prueba final.
    val_size_within_train = 0.25                                                                    # Equivale al 20 % del total tras reservar test.

    try:                                                                                            # Intenta crear particiones estratificadas consistentes.
        idx_train_val, idx_test = train_test_split(                                                 # Primera partición: train+val contra test.
            np.arange(n_samples),                                                                   # Índices completos del dataset antes de separar.
            test_size=test_size,                                                                    # Reserva la fracción definida para test.
            random_state=seed,                                                                      # Hace reproducible la partición.
            stratify=y,                                                                             # Conserva la proporción de clases en test.
        )
        idx_train, idx_val = train_test_split(                                                      # Segunda partición: train contra validación.
            idx_train_val,                                                                          # Parte únicamente del subconjunto train+val.
            test_size=val_size_within_train,                                                        # Reserva la fracción de validación interna.
            random_state=seed,                                                                      # Mantiene reproducibilidad de la separación.
            stratify=y[idx_train_val],                                                              # Conserva el balance de clases en validación.
        )
    except ValueError as exc:                                                                       # Captura fallos cuando alguna clase queda con muy pocas muestras.
        raise ValueError(                                                                           # Reempaqueta el error con una guía más clara para ajustar el experimento.
            "No fue posible crear particiones estratificadas train/val/test. "
            "Aumenta `n_samples` o revisa el balance entre clases."
        ) from exc

    X_raw_train = X_raw[idx_train]                                                                  # Señales de entrenamiento.
    X_raw_val   = X_raw[idx_val]                                                                    # Señales de validación.
    X_raw_test  = X_raw[idx_test]                                                                   # Señales de prueba.
    y_train     = y[idx_train]                                                                      # Etiquetas de entrenamiento.
    y_val       = y[idx_val]                                                                        # Etiquetas de validación.
    y_test      = y[idx_test]                                                                       # Etiquetas de prueba.

    # Las estadísticas de train se reutilizan en validación y prueba para evitar fuga de información.
    X_stft_train, X_cwt_train, norm_stats = preparar_data(X_raw_train, fs=fs)                       # Ajusta transformaciones con train.
    X_stft_val, X_cwt_val, _              = preparar_data(X_raw_val, fs=fs, stats=norm_stats)       # Reutiliza stats en validación.
    X_stft_test, X_cwt_test, _            = preparar_data(X_raw_test, fs=fs, stats=norm_stats)      # Reutiliza stats en prueba.

    if visualizar:                                                                                  # Muestra las representaciones tiempo-frecuencia si se solicita.
        visualizar_representaciones(X_stft_train, X_cwt_train)

    print(f"Seed {seed} | Dataset STFT train: {X_stft_train.shape}")                                # Verifica forma del tensor STFT de entrenamiento.
    print(f"Seed {seed} | Dataset Wavelet train: {X_cwt_train.shape}")                              # Verifica forma del tensor CWT de entrenamiento.
    print(f"Seed {seed} | Dataset STFT val: {X_stft_val.shape}")                                    # Verifica forma del tensor STFT de validación.
    print(f"Seed {seed} | Dataset Wavelet val: {X_cwt_val.shape}")                                  # Verifica forma del tensor CWT de validación.
    print(f"Seed {seed} | Dataset STFT test: {X_stft_test.shape}")                                  # Verifica forma del tensor STFT de prueba.
    print(f"Seed {seed} | Dataset Wavelet test: {X_cwt_test.shape}")                                # Verifica forma del tensor CWT de prueba.

    model = build_model(                                                                            # Crea el modelo con las formas inferidas del dataset transformado.
        X_stft_train.shape[1:],                                                                     # Forma espacial y canal de la rama STFT.
        X_cwt_train.shape[1:],                                                                      # Forma espacial y canal de la rama CWT.
        dense_units=training_config["dense_units"],                                                 # Tamaño de la capa densa de fusión.
        head_dropout=training_config["head_dropout"],                                               # Regularización aplicada en la cabeza final.
    )

    X_train = [X_stft_train, X_cwt_train]                                                           # Entradas multimodales de entrenamiento.
    X_val   = [X_stft_val, X_cwt_val]                                                               # Entradas multimodales de validación.
    X_test  = [X_stft_test, X_cwt_test]                                                             # Entradas multimodales de prueba.

    early_stopping = tf.keras.callbacks.EarlyStopping(                                              # Detiene entrenamiento cuando val_loss deja de mejorar.
        monitor='val_loss',                                                                         # Usa la pérdida de validación como criterio principal.
        patience=4,                                                                                 # Tolera varias épocas sin mejora antes de detener.
        restore_best_weights=True,                                                                  # Recupera los pesos de la mejor época observada.
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(                                               # Reduce la tasa de aprendizaje en mesetas.
        monitor='val_loss',                                                                         # Observa la misma métrica para sincronizar la estrategia.
        factor=0.5,                                                                                 # Reduce la tasa de aprendizaje a la mitad cuando hay estancamiento.
        patience=2,                                                                                 # Espera menos épocas que el early stopping para reaccionar antes.
        min_lr=1e-5,                                                                                # Impone un límite inferior a la tasa de aprendizaje.
    )
    history = model.fit(                                                                            # Ajusta el modelo con validación explícita.
        X_train,                                                                                    # Entradas multimodales del conjunto de entrenamiento.
        y_train,                                                                                    # Etiquetas binarias de entrenamiento.
        validation_data=(X_val, y_val),                                                             # Hold-out interno para monitorear generalización.
        epochs=training_config["epochs"],                                                           # Máximo de épocas permitidas.
        batch_size=training_config["batch_size"],                                                   # Tamaño de minibatch usado en el ajuste.
        callbacks=[early_stopping, reduce_lr],                                                      # Callbacks para control adaptativo del entrenamiento.
    )

    eval_metrics, eval_summary = evaluar_modelo(model, X_test, y_test)                              # Evalúa en el hold-out final.
    csv_path, json_path, confusion_csv_path, summary = exportar_resultados(                         # Exporta artefactos de la corrida.
        history,                                                                                    # Historial completo del entrenamiento.
        eval_metrics,                                                                               # Métricas directas devueltas por Keras.
        eval_summary,                                                                               # Métricas derivadas y matriz de confusión.
        output_dir,                                                                                 # Carpeta donde se guardarán los artefactos.
        {
            "seed": seed,                                                                           # Seed de la corrida actual.
            "n_samples": n_samples,                                                                 # Número total de señales generadas.
            "fs": fs,                                                                               # Frecuencia de muestreo usada en la síntesis.
            "duration": duration,                                                                   # Duración temporal de cada trial.
            "test_size": test_size,                                                                 # Proporción reservada para prueba.
            "val_size_within_train": val_size_within_train,                                         # Proporción de validación dentro de train+val.
            "batch_size": training_config["batch_size"],                                            # Tamaño de minibatch usado al entrenar.
            "epochs": training_config["epochs"],                                                    # Máximo de épocas configuradas.
            "dense_units": training_config["dense_units"],                                          # Ancho de la cabeza densa de fusión.
            "head_dropout": training_config["head_dropout"],                                        # Dropout aplicado en la cabeza clasificadora.
        },
    )

    print(f"Seed {seed} | Test loss: {summary['test_loss']:.4f}")                                   # Reporta pérdida final sobre test.
    print(f"Seed {seed} | Test accuracy: {summary['test_accuracy']:.4f}")                           # Reporta accuracy final sobre test.
    print(f"Seed {seed} | Precision: {summary['precision']:.4f}")                                   # Reporta precisión binaria.
    print(f"Seed {seed} | Recall: {summary['recall']:.4f}")                                         # Reporta recall binario.
    print(f"Seed {seed} | F1-score: {summary['f1_score']:.4f}")                                     # Reporta F1-score binario.
    print(f"Seed {seed} | Historial CSV guardado en: {csv_path}")                                   # Informa la ruta del historial por época.
    print(f"Seed {seed} | Matriz de confusion CSV guardada en: {confusion_csv_path}")               # Informa la ruta del CSV de confusión.
    print(f"Seed {seed} | Resumen JSON guardado en: {json_path}")                                   # Informa la ruta del resumen serializado.

    return model, history, summary                                                                  # Devuelve modelo, historial y resumen para análisis posterior.


def main():
    """main
    Define la configuración base y ejecuta el conjunto principal de corridas.

    Salidas:
        run_summaries           list[dict]      Resúmenes individuales por semilla.
        aggregate_summary       dict            Resumen agregado de múltiples semillas.
    """
    seeds           = [7, 21, 42, 84, 123]                                                          # Lista de seeds a ejecutar en esta corrida.
    n_samples       = 10000                                                                         # Número total de trials sintéticos.
    fs              = 250                                                                           # Frecuencia de muestreo en Hz.
    duration        = 4                                                                             # Duración de cada señal en segundos.
    training_config = {                                                                             # Hiperparámetros base compartidos por todas las seeds.
        "epochs":       20,                                                                         # Máximo de épocas por corrida.
        "batch_size":   32,  # 16                                                                   # Tamaño de minibatch.
        "dense_units":  64,  # 32                                                                   # Unidades de la cabeza densa.
        "head_dropout": 0.5, # 0.45                                                                 # Dropout de la cabeza clasificadora.
    }
    output_dir      = Path(__file__).resolve().parent / "resultados"                                # Carpeta de artefactos del experimento.
    run_summaries   = []                                                                            # Acumulador de resúmenes individuales.

    for idx, seed in enumerate(seeds):                                                         # Recorre cada seed y conserva el orden de ejecución.
        print(f"\n===== Ejecutando seed {seed} ({idx + 1}/{len(seeds)}) =====")                     # Traza progreso dentro del conjunto de corridas.
        _, _, summary = run_experimento(                                                            # Ejecuta una corrida completa para la seed actual.
            seed=seed,                                                                              # Seed activa en esta iteración.
            n_samples=n_samples,                                                                    # Cantidad de trials a sintetizar.
            fs=fs,                                                                                  # Frecuencia de muestreo del experimento.
            duration=duration,                                                                      # Duración temporal de cada señal.
            output_dir=output_dir,                                                                  # Carpeta destino de artefactos.
            training_config=training_config,                                                        # Hiperparámetros comunes a la corrida.
            visualizar=(idx == 0),                                                                  # Solo visualiza en la primera corrida.
        )
        run_summaries.append(summary)                                                               # Guarda el resumen para el consolidado final.

    consolidated_csv_path, consolidated_json_path, aggregate_summary = exportar_resumen_consolidado(# Consolida todas las corridas ejecutadas.
        run_summaries,                                                                              # Resúmenes individuales acumulados.
        output_dir,                                                                                 # Carpeta donde se escribirá el consolidado.
    )

    print("\n===== Resumen Multi-Seed =====")                                                       # Encabezado del bloque de resultados agregados.
    print(f"Seeds evaluadas: {aggregate_summary['seeds']}")                                         # Lista las seeds incluidas en el consolidado.
    print(f"Accuracy promedio: {aggregate_summary['mean_test_accuracy']:.4f} +/- {aggregate_summary['std_test_accuracy']:.4f}")
                                                                                                    # Resume media y dispersión de accuracy.
    print(f"F1 promedio: {aggregate_summary['mean_f1_score']:.4f} +/- {aggregate_summary['std_f1_score']:.4f}")
                                                                                                    # Resume media y dispersión de F1.
    print(                                                                                          # Resume la seed con mejor desempeño en el conjunto de prueba.
        "Mejor seed por test_accuracy: "
        f"{aggregate_summary['best_run_by_test_accuracy']['seed']} "
        f"({aggregate_summary['best_run_by_test_accuracy']['test_accuracy']:.4f})"
    )                                                                                               # Reporta la corrida con mejor accuracy en test.
    print(f"Resumen consolidado CSV guardado en: {consolidated_csv_path}")                          # Informa la ruta del CSV multi-seed.
    print(f"Resumen consolidado JSON guardado en: {consolidated_json_path}")                        # Informa la ruta del JSON multi-seed.

    return run_summaries, aggregate_summary                                                         # Devuelve detalle por seed y resumen agregado multi-seed.


if __name__ == "__main__":                                                                          # Ejecuta `main()` solo cuando el archivo se lanza directamente.
    run_summaries, aggregate_summary = main()                                                       # Punto de entrada cuando se ejecuta como script.
