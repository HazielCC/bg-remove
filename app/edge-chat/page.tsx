'use client';

import React, { useState, useEffect, useRef } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export default function EdgeChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState<string>('Esperando inicialización...');
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const workerRef = useRef<Worker | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  useEffect(() => {
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

    // Usar la ruta estática para cargar el modelo
    worker.postMessage({ 
      type: 'INIT', 
      payload: { modelPath: '/models/litert/qwen2.5-1.5b-int8.tflite' } 
    });

    return () => {
      worker.terminate();
    };
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating || !isReady || !workerRef.current) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    
    // Qwen2.5 prompt format: <|im_start|>role
content<|im_end|>

    const history = [...messages, userMessage].slice(-6); // Últimos 6 mensajes
    let prompt = '<|im_start|>system
You are a helpful and concise AI assistant.<|im_end|>
';
    for (const msg of history) {
      prompt += `<|im_start|>${msg.role}
${msg.content}<|im_end|>
`;
    }
    prompt += '<|im_start|>assistant
';

    setIsGenerating(true);
    workerRef.current.postMessage({
      type: 'GENERATE',
      payload: { prompt },
    });
    
    setInput('');
  };

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)] max-w-4xl mx-auto p-4 bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 rounded-xl my-4 shadow-xl border border-zinc-200 dark:border-zinc-800">
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
            Nota: La primera vez que cargues esta página, tu navegador descargará un modelo de ~1.5GB (Qwen2.5-1.5B). Esto puede tardar unos minutos dependiendo de tu conexión. Una vez cargado, permanecerá en caché.
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
              className={`p-3 rounded-xl max-w-[85%] ${
                msg.role === 'user' 
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
