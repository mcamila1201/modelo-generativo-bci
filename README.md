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
