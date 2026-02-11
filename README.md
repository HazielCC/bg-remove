
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

## Información Adicional
