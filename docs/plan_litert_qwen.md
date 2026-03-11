# Plan de Implementación: Qwen2.5-1.5B-Instruct con LiteRT (Web)

Este documento detalla los pasos para implementar un chat local en el navegador utilizando el modelo `Qwen2.5-1.5B-Instruct` optimizado con LiteRT (anteriormente TFLite).

## Objetivo
Crear una experiencia de chat completamente aislada en el frontend (Next.js) donde el modelo se ejecuta en el dispositivo del usuario mediante WebAssembly/WebGPU, sin necesidad de un servidor backend.

## Fases de Implementación

### Fase 1: Preparación del Entorno y Modelo
1.  **Script de Descarga:** Crear un script (e.g., `scripts/download_qwen_litert.sh`) para descargar el modelo `.tflite` (versión `dynamic_int8`) desde Hugging Face.
2.  **Almacenamiento Local:** El modelo se guardará en `public/models/litert/qwen2.5-1.5b-int8.tflite` (debe ser excluido de git por su tamaño, ~1.5GB).
3.  **Dependencias:** Instalar las librerías necesarias en el frontend:
    *   `@mediapipe/tasks-genai` (Para la inferencia del LLM en la web).

### Fase 2: Motor de Inferencia (Web Worker)
Dado que la inferencia de un LLM bloquea el hilo principal del navegador, es crucial aislar el procesamiento en un Web Worker.
1.  **Crear el Worker:** `app/edge-chat/llm.worker.ts`
2.  **Inicialización:** Configurar `LlmInference` de MediaPipe Tasks GenAI para cargar el modelo desde la URL pública.
3.  **Comunicación:** Establecer un sistema de mensajes (PostMessage) para:
    *   Informar sobre el progreso de carga del modelo.
    *   Recibir prompts del usuario.
    *   Emitir los tokens generados en streaming (respuesta parcial).
    *   Notificar la finalización de la generación.

### Fase 3: Interfaz de Usuario (UI) en Next.js
1.  **Nueva Ruta:** Crear la estructura de página en `app/edge-chat/page.tsx`.
2.  **Estado de la Aplicación:** Gestionar estados complejos:
    *   `isDownloading` / `isLoadingModel`
    *   `messages` (Historial de la conversación)
    *   `isGenerating`
3.  **Diseño (UI/UX):**
    *   Indicador visual claro del progreso de carga (es un archivo grande).
    *   Interfaz de chat moderna (estilo ChatGPT).
    *   Uso de Vanilla CSS o Tailwind (según convención del proyecto) para asegurar un renderizado rápido y fluido.

### Fase 4: Optimización (Vercel React Best Practices)
Aplicar reglas de rendimiento de React:
*   `rerender-functional-setstate`: Usar actualizaciones funcionales de estado al hacer streaming de tokens para evitar dependencias innecesarias y re-renders.
*   `rerender-move-effect-to-event`: Mantener la lógica de interacción en los handlers de eventos en lugar de `useEffect`.
*   Asegurar que el streaming de texto no cause re-renders pesados de toda la lista de mensajes.

## Consideraciones Técnicas
*   **Memoria (RAM):** El dispositivo del usuario necesitará al menos 4-6GB de RAM libre para cargar y ejecutar el modelo de 1.5B parámetros de manera fluida.
*   **Aceleración Hardware:** MediaPipe intentará usar WebGPU si está disponible en el navegador del usuario, haciendo fallback a WebAssembly (CPU) si es necesario.
*   **Contexto:** Se recomienda limitar el contexto histórico enviado al modelo (ej. últimos 4-6 mensajes) para evitar agotar la memoria y mantener tiempos de respuesta rápidos.
