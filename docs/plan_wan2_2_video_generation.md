# Plan de Integración Simplificado: Wan2.2-TI2V-5B (Sandbox / Experimental)

## Fase 1 — Integración Mínima (1 Día de Trabajo)

**Objetivo:** Agregar una ruta experimental de generación de video dentro del backend existente de forma síncrona, manteniendo todo en un solo proceso. Sin infraestructura extra.

### 1. Arquitectura Simplificada

```text
Next.js UI (app/video-test/page.tsx)
    ↓
FastAPI endpoint (POST /api/video/generate)
    ↓
Wan2.2 inference (backend/ml/wan_video.py)
    ↓
Guardar video local (backend/exports/videos/)
    ↓
Devolver URL estática al frontend
```

**Lo que NO se usará (fuera de alcance para pruebas):**
❌ Redis / Celery / BackgroundTasks
❌ Workers externos
❌ Almacenamiento S3 (solo disco local)
❌ Sistemas complejos de observabilidad / WebSockets

### 2. Estructura del Backend

Se agregarán únicamente los siguientes componentes:

```text
/backend
 ├── routers/
 │    └── video.py             # Endpoint único POST /video/generate
 ├── ml/
 │    └── wan_video.py         # Carga perezosa (lazy-load) del modelo Wan2.2
 └── models/
      └── schemas_video.py     # Esquema Pydantic para el request
```

#### Flujo Backend:
1. Recibir Request (`Prompt`, `Duration`).
2. Pasar al wrapper `wan_video.generate()`.
3. El wrapper carga el modelo (solo la primera vez gracias a una variable global).
4. Se ejecuta la inferencia (bloqueando el request, lo cual es aceptable para pruebas locales).
5. Se guarda el archivo `.mp4` en el disco local.
6. Se responde con la ruta del archivo generado.

### 3. Estructura del Frontend

Se creará una vista mínima para interactuar con la API.

```text
/app/video-test/page.tsx
```

**Componentes mínimos:**
* Textarea para el Prompt.
* Botón "Generar Video" (deshabilitado durante la carga).
* Reproductor de video (`<video>`) para mostrar el resultado final.

#### Flujo UI:
1. Usuario escribe el prompt.
2. `POST /api/video/generate` (bloquea la UI mostrando un "Cargando...").
3. Espera la respuesta (minutos).
4. Muestra el video renderizado.

### 4. Estimación de Tiempos

| Tarea | Tiempo Estimado |
| :--- | :--- |
| Integrar endpoint FastAPI | 2 horas |
| Wrapper del modelo Wan2.2 | 3 horas |
| UI básica en Next.js | 2 horas |
| Pruebas locales | 2 horas |
| **Total Aproximado** | **1 día de trabajo** |