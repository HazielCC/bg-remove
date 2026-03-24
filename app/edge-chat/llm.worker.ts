import { LlmInference, FilesetResolver } from '@mediapipe/tasks-genai';

let llmInference: LlmInference | null = null;

type WorkerMessage =
  | { type: 'INIT'; payload: { modelPath: string } }
  | { type: 'GENERATE'; payload: { prompt: string } };

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;

  if (type === 'INIT') {
    try {
      self.postMessage({ type: 'STATUS', message: 'Descargando motor y modelo LiteRT (~1.5GB)...' });
      const genai = await FilesetResolver.forGenAiTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
      );
      
      self.postMessage({ type: 'STATUS', message: 'Cargando modelo en memoria...' });
      
      llmInference = await LlmInference.createFromOptions(genai, {
        baseOptions: {
          modelAssetPath: payload.modelPath,
        },
        maxTokens: 1024,
      });
      self.postMessage({ type: 'INIT_SUCCESS' });
    } catch (error: unknown) {
      const message = getErrorMessage(error);
      console.error("Worker INIT error:", error);
      self.postMessage({ type: 'ERROR', message });
    }
  }

  if (type === 'GENERATE') {
    if (!llmInference) {
      self.postMessage({ type: 'ERROR', message: 'El modelo no está inicializado.' });
      return;
    }
    try {
      self.postMessage({ type: 'STATUS', message: 'Generando...' });
      
      llmInference.generateResponse(payload.prompt, (partialResult: string, done: boolean) => {
        self.postMessage({
          type: 'PARTIAL_RESULT',
          text: partialResult,
          done: done,
        });
      });
    } catch (error: unknown) {
      const message = getErrorMessage(error);
      console.error("Worker GENERATE error:", error);
      self.postMessage({ type: 'ERROR', message });
    }
  }
};
