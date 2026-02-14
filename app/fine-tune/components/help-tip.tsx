"use client";

interface HelpTipProps {
    text: string;
}

export default function HelpTip({ text }: HelpTipProps) {
    return (
        <span className="relative inline-flex items-center group align-middle">
            <button
                type="button"
                className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-neutral-400 text-[10px] font-bold text-neutral-500 hover:text-neutral-800 hover:border-neutral-500 dark:border-neutral-600 dark:text-neutral-400 dark:hover:text-neutral-200"
                aria-label={text}
            >
                ?
            </button>
            <span
                role="tooltip"
                className="pointer-events-none absolute left-1/2 top-full z-30 mt-2 w-56 max-w-[calc(100vw-2rem)] -translate-x-1/2 rounded-md bg-neutral-900 px-2 py-1.5 text-[11px] leading-snug text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 sm:w-64"
            >
                {text}
            </span>
        </span>
    );
}
