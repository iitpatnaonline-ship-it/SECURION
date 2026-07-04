import React from 'react';

export const QueryLog: React.FC = () => {
    return (
        <section className="py-16 px-6 md:px-16 relative z-10 w-full">
            <div className="max-w-3xl mx-auto flex flex-col gap-4 text-left">
                <h2 className="font-mono text-lg text-white mb-8 text-center uppercase tracking-[0.2em]">Operational Query Log</h2>

                <details className="glass-panel rounded-none group overflow-hidden border border-white/10">
                    <summary className="flex justify-between items-center font-mono text-sm font-semibold cursor-pointer p-6 list-none uppercase tracking-wider select-none">
                        <span className="text-white">How does SECURION achieve sub-15ms latency?</span>
                        <span className="transition group-open:rotate-180 text-cyan-500 text-xs">▼</span>
                    </summary>
                    <div className="text-[#94a3b8] px-6 pb-6 font-mono text-xs uppercase border-t border-white/5 pt-4">
                        By processing lightweight computer vision layers directly on localized edge nodes, bypassing heavy cloud round-trips.
                    </div>
                </details>

                <details className="glass-panel rounded-none group overflow-hidden border border-white/10">
                    <summary className="flex justify-between items-center font-mono text-sm font-semibold cursor-pointer p-6 list-none uppercase tracking-wider select-none">
                        <span className="text-white">Can it integrate with existing legacy CCTV hardware?</span>
                        <span className="transition group-open:rotate-180 text-cyan-500 text-xs">▼</span>
                    </summary>
                    <div className="text-[#94a3b8] px-6 pb-6 font-mono text-xs uppercase border-t border-white/5 pt-4">
                        Yes, it hooks directly into any standard RTSP video stream overlay seamlessly.
                    </div>
                </details>
            </div>
        </section>
    );
};