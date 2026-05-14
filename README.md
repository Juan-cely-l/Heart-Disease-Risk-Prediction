# Heart Disease Risk Prediction

Sistema académico de analítica predictiva para estimar riesgo de enfermedad cardiaca con regresión logística interpretable, datos clínicos estructurados y una proyección responsable hacia despliegue en la nube.

> Este proyecto es un prototipo educativo de apoyo analítico. No reemplaza criterio médico, diagnóstico clínico ni evaluación profesional de salud.

## Resumen Ejecutivo

Las enfermedades cardiovasculares siguen siendo una de las problemáticas de salud más críticas del mundo. Según la OMS, son la principal causa de muerte global y en 2022 causaron cerca de 19.8 millones de muertes, equivalentes a aproximadamente el 32% de las muertes mundiales. La misma fuente resalta la importancia de detectar el riesgo cardiovascular de forma temprana para iniciar manejo, consejería y tratamiento oportunos.

Este proyecto aborda esa necesidad desde Ingeniería de Sistemas, seguridad y datos: construye un pipeline de machine learning con NumPy, Pandas y Matplotlib para analizar variables clínicas, entrenar un modelo de regresión logística, evaluar su desempeño, interpretar los factores de riesgo y exportar artefactos listos para inferencia.

## Problema Que Resuelve

Muchas decisiones preventivas en salud cardiovascular llegan tarde porque los datos clínicos disponibles no siempre se convierten en señales tempranas, comprensibles y accionables. El problema no es solo predecir: es convertir datos en una alerta interpretable, reproducible y segura que pueda apoyar decisiones humanas.

El proyecto propone una base técnica para responder:

- ¿Qué variables clínicas se relacionan más con presencia de enfermedad cardiaca?
- ¿Cómo estimar una probabilidad de riesgo con un modelo entendible?
- ¿Cómo preparar un modelo para un entorno de inferencia controlado?
- ¿Qué controles de seguridad y gobierno de datos serían necesarios antes de llevarlo a producción?

## Solución Propuesta

El repositorio implementa un flujo completo de analítica predictiva:

1. Carga del dataset `Heart_Disease_Prediction.csv`.
2. Limpieza y transformación del objetivo `Heart Disease` a variable binaria.
3. Análisis exploratorio de datos, distribución de clases, correlaciones y outliers.
4. Selección de seis variables clínicas relevantes:
   - `Thallium`
   - `Number of vessels fluro`
   - `Chest pain type`
   - `Exercise angina`
   - `Sex`
   - `Age`
5. División estratificada entrenamiento/prueba.
6. Normalización Z-score usando estadísticas del conjunto de entrenamiento.
7. Implementación de regresión logística desde cero con NumPy.
8. Evaluación con accuracy, precision, recall, F1-score y matriz de confusión.
9. Regularización L2 y comparación de valores de lambda.
10. Exportación de pesos, sesgo y parámetros de normalización para inferencia.
11. Documentación de despliegue explorado en Amazon SageMaker mediante capturas en `ASSETS/`.

## Resultado Actual

El análisis documentado reporta un desempeño aproximado de 88.89% de accuracy en el conjunto de prueba para el modelo principal. La matriz de confusión incluida en la documentación del notebook muestra:

| Clase real / predicha | Predicho sin enfermedad | Predicho con enfermedad |
| --- | ---: | ---: |
| Sin enfermedad | 42 | 3 |
| Con enfermedad | 6 | 30 |

Interpretación académica: el modelo logra una base predictiva prometedora para un prototipo, pero requiere validación con más datos, revisión clínica, evaluación de sesgos y monitoreo antes de cualquier uso real.

## Usuarios Beneficiados

- **Estudiantes y docentes:** como caso práctico de machine learning interpretable aplicado a salud.
- **Analistas de datos:** como ejemplo de flujo reproducible desde EDA hasta exportación de modelo.
- **Equipos de salud digital:** como base conceptual para herramientas de apoyo a prevención.
- **Instituciones de salud:** como proyección futura para priorizar tamizajes, seguimiento preventivo o alertas tempranas, siempre con supervisión médica.

## Propuesta de Valor

El valor del proyecto está en unir tres dimensiones:

- **Datos:** convierte variables clínicas en evidencia cuantitativa.
- **Seguridad:** reconoce que los datos de salud requieren confidencialidad, control de acceso y despliegue responsable.
- **Ingeniería de Sistemas:** organiza el proceso como un sistema reproducible, evaluable y proyectable a infraestructura cloud.

A diferencia de un modelo de caja negra, la regresión logística permite explicar qué variables empujan la predicción hacia mayor o menor riesgo.

## Tendencia Tecnológica Seleccionada

**IA confiable y explicable para salud predictiva.**

Es la tendencia más coherente con este repositorio porque el proyecto ya trabaja con predicción de riesgo en salud, interpretación de variables clínicas, métricas de desempeño y una proyección de despliegue en cloud. La tendencia exige que los sistemas de IA en salud no solo sean precisos, sino también entendibles, seguros, auditables y centrados en las personas.

### Tres Características Principales

1. **Explicabilidad e interpretabilidad:** las predicciones deben poder explicarse para que médicos, pacientes y equipos técnicos entiendan qué factores influyen en el riesgo.
2. **Seguridad, privacidad y gobierno de datos:** los datos clínicos deben protegerse con controles de acceso, cifrado, trazabilidad y reglas claras de uso.
3. **Validación continua y supervisión humana:** el modelo debe evaluarse con métricas, monitoreo, revisión de sesgos y participación de profesionales de salud antes de impactar decisiones reales.

## Arquitectura General

```text
Dataset CSV
    |
    v
Notebook de analisis y entrenamiento
    |
    |-- Limpieza del objetivo
    |-- EDA y correlaciones
    |-- Seleccion de variables
    |-- Split estratificado
    |-- Normalizacion Z-score
    |-- Regresion logistica con NumPy
    |-- Regularizacion L2
    v
Evaluacion del modelo
    |
    |-- Accuracy / Precision / Recall / F1
    |-- Matriz de confusion
    |-- Interpretacion de coeficientes
    v
Artefactos exportados
    |
    |-- model_params.json
    |-- model_weights.npy
    |-- model_bias.npy
    |-- normalization_mu.npy
    |-- normalization_sigma.npy
    v
Proyeccion de inferencia
    |
    |-- Handler documentado en el notebook
    |-- Evidencia visual de SageMaker en ASSETS/
```

## Flujo de Datos

1. El archivo `Heart_Disease_Prediction.csv` entrega 270 registros con variables clínicas.
2. La columna `Heart Disease` se estandariza y se convierte a `Heart Disease_bin`.
3. El notebook revisa calidad de datos: nulos, duplicados, distribución y correlaciones.
4. Se seleccionan seis variables predictoras por relevancia estadística e interpretabilidad.
5. Se divide el dataset en entrenamiento y prueba con proporción 70/30 estratificada.
6. Se calculan `mu` y `sigma` solo sobre entrenamiento para evitar fuga de datos.
7. Se entrena el modelo y se evalúa sobre datos no usados para entrenar.
8. Se exportan los artefactos necesarios para reproducir inferencias.
9. Una inferencia esperada recibe un JSON clínico, normaliza los valores, calcula probabilidad y retorna nivel de riesgo.

## Decisiones y Riesgos de Seguridad

Estado actual:

- El dataset del repositorio no incluye identificadores personales directos.
- No existe una API productiva local en el repositorio; el flujo principal vive en el notebook.
- El notebook documenta un handler de inferencia para SageMaker, pero `inference.py` no está persistido como archivo independiente en la raíz.
- Los archivos del modelo están versionados como artefactos académicos.

Controles necesarios antes de producción:

- Cifrado de datos en reposo y en tránsito.
- IAM de mínimo privilegio para notebooks, buckets y endpoints cloud.
- Validación estricta del payload de entrada.
- Evitar logs con datos clínicos sensibles.
- Monitoreo de drift, desempeño y falsos negativos.
- Auditoría de accesos e inferencias.
- Revisión de sesgos por edad, sexo y subpoblaciones.
- Consentimiento, anonimización y cumplimiento normativo si se usan datos reales de pacientes.

## Alcance y Limitaciones

Implementado:

- Notebook completo de análisis, entrenamiento y evaluación.
- Regresión logística desde cero con NumPy.
- Regularización L2.
- Visualizaciones y explicación de variables.
- Exportación de artefactos del modelo.
- Evidencia visual de exploración de despliegue en SageMaker.

No implementado como código independiente:

- API web local.
- Interfaz gráfica.
- Pipeline automatizado de CI/CD.
- Validación clínica externa.
- Monitoreo productivo del modelo.
- Archivo `inference.py` separado del notebook.

## Estructura del Proyecto

```text
Heart-Disease-Risk-Prediction/
|-- heart_disease_lr_analysis.ipynb   # Notebook principal de analisis y entrenamiento
|-- Heart_Disease_Prediction.csv      # Dataset clinico usado por el notebook
|-- model_params.json                 # Configuracion exportada del modelo
|-- model_weights.npy                 # Pesos del modelo
|-- model_bias.npy                    # Sesgo del modelo
|-- normalization_mu.npy              # Medias para normalizacion
|-- normalization_sigma.npy           # Desviaciones estandar para normalizacion
|-- ASSETS/                           # Capturas de evidencia del despliegue explorado
|-- docs/
|   |-- SUSTENTACION.md               # Base academica de la presentacion oral
|   |-- GUION_ORAL.md                 # Guion oral de alto impacto
|   |-- PROMPT_GAMMA.md               # Prompt listo para generar diapositivas
|-- README.md                         # Documentacion principal
|-- .gitignore
```

## Cómo Ejecutar

Crear y activar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar dependencias mínimas:

```bash
pip install numpy pandas matplotlib jupyter
```

Abrir el notebook:

```bash
jupyter notebook heart_disease_lr_analysis.ipynb
```

Ejecutar las celdas en orden para reproducir el análisis y regenerar los artefactos del modelo.

## Sustentación Oral

Los archivos de apoyo están en `docs/`:

- `docs/SUSTENTACION.md`: narrativa académica del proyecto, tendencia, problema, solución, usuarios e impacto.
- `docs/GUION_ORAL.md`: guion listo para exposición oral.
- `docs/PROMPT_GAMMA.md`: prompt para crear una presentación visual en Gamma.

## Fuentes de Contexto

- OMS, enfermedades cardiovasculares: https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29
- OMS, ética y gobernanza de IA en salud: https://www.who.int/publications/i/item/9789240029200
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI Research, características de IA confiable: https://www.nist.gov/ai-research
