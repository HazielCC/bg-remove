import { LlmInference, FilesetResolver } from '@mediapipe/tasks-genai';

let llmInference: LlmInference | null = null;

self.onmessage = async (event: MessageEvent) => {
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
    } catch (error: any) {
      console.error("Worker INIT error:", error);
      self.postMessage({ type: 'ERROR', message: error.message });
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
    } catch (error: any) {
      console.error("Worker GENERATE error:", error);
      self.postMessage({ type: 'ERROR', message: error.message });
    }
  }
};
