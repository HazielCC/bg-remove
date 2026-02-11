import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center p-8">
      <div className="max-w-xl w-full space-y-8 text-center">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">
            Remoción de Fondo con MODNet
          </h1>
          <p className="text-secondary mt-2">
            Matteo de retratos con capacidades de fine-tuning
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Link
            href="/remove-bg"
            className="border rounded-xl p-6 hover:border-accent hover:bg-success dark:hover:bg-blue-950/20 transition-colors group dark:border-neutral-700"
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
            href="/fine-tune"
            className="border rounded-xl p-6 hover:border-accent hover:bg-success dark:hover:bg-green-950/20 transition-colors group dark:border-neutral-700"
          >
            <div className="text-3xl mb-3">🏋️</div>
            <h2 className="text-lg font-semibold group-hover:text-accent">
              Fine-Tune MODNet
            </h2>
            <p className="text-sm text-secondary mt-1">
              Gestiona datasets, entrena, monitorea, compara modelos y exporta a ONNX.
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
