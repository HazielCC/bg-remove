"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
    { href: "/fine-tune/datasets", label: "Datasets", icon: "📦" },
    { href: "/fine-tune/train", label: "Training", icon: "🏋️" },
    { href: "/fine-tune/monitor", label: "Monitor", icon: "📊" },
    { href: "/fine-tune/compare", label: "Compare", icon: "🔍" },
    { href: "/fine-tune/export", label: "Export", icon: "📤" },
];

export default function FineTuneLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const pathname = usePathname();

    return (
        <div className="flex min-h-screen">
            {/* Sidebar */}
            <aside className="w-56 border-r border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 flex flex-col">
                <div className="p-4 border-b border-neutral-200 dark:border-neutral-800">
                    <Link href="/" className="text-xs text-secondary hover:text-accent dark:hover:text-neutral-300">
                        ← Back
                    </Link>
                    <h2 className="text-lg font-bold mt-1">MODNet Fine-Tune</h2>
                    <p className="text-xs text-secondary">Training & Evaluation</p>
                </div>
                <nav className="flex-1 p-2 space-y-1">
                    {NAV_ITEMS.map((item) => {
                        const active = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${active
                                    ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium"
                                    : "text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-900"
                                    }`}
                            >
                                <span>{item.icon}</span>
                                {item.label}
                            </Link>
                        );
                    })}
                </nav>
                <div className="p-4 border-t border-neutral-200 dark:border-neutral-800">
                    <Link
                        href="/remove-bg"
                        className="text-xs text-neutral-500 hover:text-blue-600 transition-colors"
                    >
                        → Inference Demo
                    </Link>
                </div>
            </aside>

            {/* Main content */}
            <main className="flex-1 overflow-auto">
                {children}
            </main>
        </div>
    );
}
