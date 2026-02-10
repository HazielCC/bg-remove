import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center p-8">
      <div className="max-w-xl w-full space-y-8 text-center">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">
            MODNet Background Removal
          </h1>
          <p className="text-neutral-500 mt-2">
            Portrait matting with fine-tuning capabilities
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Link
            href="/remove-bg"
            className="border rounded-xl p-6 hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950/20 transition-colors group dark:border-neutral-700"
          >
            <div className="text-3xl mb-3">🖼️</div>
            <h2 className="text-lg font-semibold group-hover:text-blue-600">
              Remove Background
            </h2>
            <p className="text-sm text-neutral-500 mt-1">
              Run MODNet inference in the browser. Upload an image and get the alpha matte.
            </p>
          </Link>

          <Link
            href="/fine-tune"
            className="border rounded-xl p-6 hover:border-green-400 hover:bg-green-50 dark:hover:bg-green-950/20 transition-colors group dark:border-neutral-700"
          >
            <div className="text-3xl mb-3">🏋️</div>
            <h2 className="text-lg font-semibold group-hover:text-green-600">
              Fine-Tune MODNet
            </h2>
            <p className="text-sm text-neutral-500 mt-1">
              Manage datasets, train, monitor, compare models, and export to ONNX.
            </p>
          </Link>
        </div>

        <p className="text-xs text-neutral-400">
          Based on{" "}
          <a
            href="https://huggingface.co/Xenova/modnet"
            target="_blank"
            rel="noopener"
            className="underline hover:text-neutral-600"
          >
            Xenova/modnet
          </a>{" "}
          · PyTorch + Apple Silicon (MPS)
        </p>
      </div>
    </main>
  );
}
