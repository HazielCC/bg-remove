
Un proyecto full-stack para remover fondos de imágenes usando Next.js (frontend) y FastAPI con Python ML (backend).

## Requisitos Previos

## Configuración Inicial

### 1. Instalar `uv`

```bash
# En Windows (usando pipx o descarga directa)
pip install uv
# O en macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verifica la instalación:

```bash
uv --version
```

### 2. Sincronizar dependencias del backend

```bash
cd backend
uv sync
cd ..
```

Esto instalará todas las dependencias de Python en un virtual environment administrado por `uv`.

### 3. Instalar dependencias de frontend

```bash
pnpm install
```

## Desarrollo

### Opción 1: Correr todo en paralelo (frontend + backend)

```bash
pnpm dev:all
```

Este comando inicia simultáneamente:

### Opción 2: Correr solo frontend

```bash
pnpm dev
```

### Opción 3: Correr solo backend

```bash
pnpm dev:back
```

## Estructura del Proyecto

```
├── app/              # Frontend Next.js (React, TypeScript)
├── backend/          # Backend FastAPI (Python ML)
├── public/           # Assets estáticos
└── scripts/          # Scripts de setup
```

## Modelos ML

El proyecto incluye modelos MODNet para segmentación de imágenes:

## Gemini 3: Evaluación de Datasets

Puedes activar evaluación de calidad de dataset con Gemini 3 para priorizar imágenes útiles antes del entrenamiento.

Variables de entorno (backend):

```bash
GEMINI_API_KEY=tu_api_key
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_DEFAULT_MAX_IMAGES=200
```

Flujo:

1. Ve a `Fine Tune > Datasets`.
2. En un dataset local, pulsa `Gemini Assess`.
3. El sistema guarda resultados en:
   - `backend/data/<dataset_id>/gemini_assessment.jsonl`
   - `backend/data/<dataset_id>/gemini_assessment_summary.json`

## Información Adicional
