'use client';

import React, { useEffect, useRef, useState } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export default function EdgeChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState<string>('Esperando inicialización...');
  const [loadProgress, setLoadProgress] = useState<number | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  useEffect(() => {
    const initialize = async () => {
      // Inicializar el Web Worker
      const worker = new Worker(new URL('./llm.worker.ts', import.meta.url));
      workerRef.current = worker;

      worker.onmessage = (e) => {
        const { type, message, text, done } = e.data;

        if (type === 'STATUS') {
          setStatus(message);
        } else if (type === 'INIT_SUCCESS') {
          setIsReady(true);
          setStatus('Modelo listo.');
        } else if (type === 'ERROR') {
          setStatus(`Error: ${message}`);
          setIsGenerating(false);
        } else if (type === 'PARTIAL_RESULT') {
          setMessages((prev) => {
            const newMessages = [...prev];
            const lastIndex = newMessages.length - 1;

            if (lastIndex >= 0 && newMessages[lastIndex].role === 'assistant') {
              newMessages[lastIndex] = {
                ...newMessages[lastIndex],
                content: newMessages[lastIndex].content + text,
              };
            } else {
              newMessages.push({ role: 'assistant', content: text });
            }
            return newMessages;
          });

          if (done) {
            setIsGenerating(false);
            setStatus('Modelo listo.');
          }
        }
      };

      // Descarga manual del modelo para mostrar porcentaje
      const modelPath = '/models/litert/qwen3.5-0.8b-int8.tflite';
      const modelUrl = `${window.location.origin}${modelPath}`;

      const fetchWithProgress = async (url: string) => {
        const cache = await caches.open('llm-models');
        const cachedResponse = await cache.match(url);
        
        if (cachedResponse) {
          setStatus('Cargando modelo desde caché local...');
          setLoadProgress(100);
          const blob = await cachedResponse.blob();
          return URL.createObjectURL(blob);
        }

        setStatus('Iniciando descarga (800MB aprox)...');
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
        
        const contentLength = parseInt(resp.headers.get('Content-Length') || '838860800', 10); // Fallback to ~800MB
        if (!resp.body) return url;

        const reader = resp.body.getReader();
        let received = 0;
        
        // Creamos un stream para ir guardando en caché conforme se descarga
        const stream = new ReadableStream({
          async start(controller) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                controller.close();
                break;
              }
              if (value) {
                received += value.length;
                const pct = Math.floor((received / contentLength) * 100);
                setLoadProgress(pct);
                setStatus(`Descargando modelo: ${pct}%`);
                controller.enqueue(value);
              }
            }
          }
        });

        // Clonamos la respuesta original y le inyectamos nuestro stream
        const responseToCache = new Response(stream, { headers: resp.headers });
        await cache.put(url, responseToCache);
        
        // Leemos desde la caché para asegurar persistencia
        const savedResponse = await cache.match(url);
        const blob = await savedResponse!.blob();
        return URL.createObjectURL(blob);
      };

      try {
        const urlToUse = await fetchWithProgress(modelUrl);
        worker.postMessage({
          type: 'INIT',
          payload: { modelPath: urlToUse }
        });
      } catch (err: unknown) {
        console.error('error downloading model', err);
        setStatus('Error descargando modelo');
      }
    };

    initialize();

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating || !isReady || !workerRef.current) return;

    const userMessage: Message = { role: 'user', content: input };
    setInput('');
    setIsGenerating(true);

    setMessages((prev) => {
      const updatedMessages = [...prev, userMessage];
      const history = updatedMessages.slice(-6); // Últimos 6 mensajes
      
      let prompt = '<|im_start|>system\nYou are a helpful and concise AI assistant.<|im_end|>\n';
      for (const msg of history) {
        prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
      }
      prompt += '<|im_start|>assistant\n';

      workerRef.current?.postMessage({
        type: 'GENERATE',
        payload: { prompt },
      });

      return updatedMessages;
    });
  };

  return (
    <div className="relative flex flex-col h-[calc(100vh-2rem)] max-w-4xl mx-auto p-4 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 rounded-xl my-4 shadow-xl border border-zinc-200 dark:border-zinc-800">
      {/* overlay while loading model */}
      {!isReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 dark:bg-zinc-900/80 z-50">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          {loadProgress !== null && (
            <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-200">{loadProgress}%</p>
          )}
        </div>
      )}
      <header className="py-4 border-b border-zinc-200 dark:border-zinc-800">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          Edge Chat
          <span className="text-xs bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100 px-2 py-1 rounded-full font-medium">Local / Offline</span>
        </h1>
        <p className="text-sm text-zinc-500 mt-1">
          Estado: <span className="font-medium text-emerald-600 dark:text-emerald-400">{status}</span>
        </p>
        {!isReady && (
          <p className="text-xs text-amber-600 dark:text-amber-400 mt-2">
            Nota: La primera vez que cargues esta página, tu navegador descargará un modelo de ~800MB (Qwen3.5-0.8B). Esto puede tardar unos minutos dependiendo de tu conexión. Una vez cargado, permanecerá en caché.
          </p>
        )}
      </header>

      <main className="flex-1 overflow-y-auto py-4 space-y-4 pr-2">
        {messages.length === 0 ? (
          <div className="text-center text-zinc-400 my-10 flex flex-col items-center">
            <svg xmlns="http://www.w3.org/-svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mb-4 opacity-50"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            <p>Envía un mensaje para comenzar a chatear localmente.</p>
            <p className="text-sm mt-2 opacity-70">El procesamiento se hace 100% en tu navegador. Nada se envía a la nube.</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`p-3 rounded-xl max-w-[85%] ${msg.role === 'user'
                  ? 'bg-blue-600 text-white ml-auto rounded-tr-sm'
                  : 'bg-zinc-100 dark:bg-zinc-800 rounded-tl-sm'
                }`}
            >
              <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
            </div>
          ))
        )}

        {isGenerating && status === 'Generando...' && messages[messages.length - 1]?.role === 'user' && (
          <div className="p-4 rounded-xl max-w-[85%] bg-zinc-100 dark:bg-zinc-800 rounded-tl-sm w-16 flex justify-center">
            <span className="flex gap-1 items-center">
              <div className="w-2 h-2 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
        <form onSubmit={handleSubmit} className="flex gap-2 relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={!isReady || isGenerating}
            placeholder={isReady ? "Escribe un mensaje..." : "Cargando modelo en memoria..."}
            className="flex-1 p-3 pr-24 border border-zinc-300 dark:border-zinc-700 rounded-xl bg-transparent disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm transition-all"
          />
          <button
            type="submit"
            disabled={!isReady || isGenerating || !input.trim()}
            className="absolute right-1 top-1 bottom-1 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 disabled:bg-zinc-400 font-medium transition-colors"
          >
            Enviar
          </button>
        </form>
      </footer>
    </div>
  );
}
