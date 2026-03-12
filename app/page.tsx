import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center p-8">
      <div className="max-w-4xl w-full space-y-8 text-center">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">
            Remoción de Fondo con MODNet
          </h1>
          <p className="text-secondary mt-2">
            Matteo de retratos con capacidades de fine-tuning y asistentes locales
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-left">
          <Link
            href="/remove-bg"
            className="border rounded-xl p-6 hover:border-accent hover:bg-success dark:hover:bg-blue-950/20 transition-colors group dark:border-neutral-700 flex flex-col items-center text-center md:items-start md:text-left"
          >
            <div className="text-3xl mb-3">🖼️</div>
            <h2 className="text-lg font-semibold group-hover:text-accent">
              Remover Fondo
            </h2>
            <p className="text-sm text-secondary mt-1">
              Ejecuta inferencia MODNet en el navegador. Sube una imagen y obtén la máscara alfa.
            </p>
          </Link>

          <Link
            href="/layered"
            className="border rounded-xl p-6 hover:border-accent hover:bg-success dark:hover:bg-yellow-950/20 transition-colors group dark:border-neutral-700 flex flex-col items-center text-center md:items-start md:text-left"
          >
            <div className="text-3xl mb-3">🪄</div>
            <h2 className="text-lg font-semibold group-hover:text-accent">
              Smart Layers
            </h2>
            <p className="text-sm text-secondary mt-1">
              Descompone imágenes en capas RGBA inteligentes usando Qwen-Image-Layered localmente.
            </p>
          </Link>

          <Link
            href="/fine-tune"
            className="border rounded-xl p-6 hover:border-accent hover:bg-success dark:hover:bg-green-950/20 transition-colors group dark:border-neutral-700 flex flex-col items-center text-center md:items-start md:text-left"
          >
            <div className="text-3xl mb-3">🏋️</div>
            <h2 className="text-lg font-semibold group-hover:text-accent">
              Fine-Tune MODNet
            </h2>
            <p className="text-sm text-secondary mt-1">
              Gestiona datasets, entrena, monitorea, compara modelos y exporta a ONNX.
            </p>
          </Link>

          <Link
            href="/edge-chat"
            className="border rounded-xl p-6 hover:border-accent hover:bg-success dark:hover:bg-purple-950/20 transition-colors group dark:border-neutral-700 flex flex-col items-center text-center md:items-start md:text-left relative overflow-hidden"
          >
            <div className="text-3xl mb-3">💬</div>
            <h2 className="text-lg font-semibold group-hover:text-accent flex gap-2 items-center">
              Edge Chat
              <span className="text-[10px] bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300 px-1.5 py-0.5 rounded font-bold uppercase tracking-wide">Local</span>
            </h2>
            <p className="text-sm text-secondary mt-1">
              Conversa con el modelo Qwen 1.5B ejecutándose 100% en tu navegador vía WebGPU/Wasm.
            </p>
          </Link>
        </div>

        <p className="text-xs text-muted">
          Based on{" "}
          <a
            href="https://huggingface.co/Xenova/modnet"
            target="_blank"
            rel="noopener"
            className="underline hover:text-accent"
          >
            Xenova/modnet
          </a>{" "}
          · PyTorch + Apple Silicon (MPS)
        </p>
      </div>
    </main>
  );
}
