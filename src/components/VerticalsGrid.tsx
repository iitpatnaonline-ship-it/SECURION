import React from 'react';

const verticals = [
    { metric: '01', title: 'Smart Cities', desc: 'Centralized urban monitoring pipelines with edge threat verification structures.' },
    { metric: '02', title: 'Commercial Spaces', desc: 'Enterprise server networks isolating anomalous loitering or access signals.' },
    { metric: '03', title: 'Critical Infrastructure', desc: 'Hardware arrays maintaining 24/7 autonomous zone protection matrices.' }
];

export const VerticalsGrid: React.FC = () => {
    return (
        <section className="w-full max-w-7xl mx-auto px-6 md:px-16 mt-20 border-t border-white/5 pt-16">
            {/* Structural Header */}
            <div className="text-left mb-12">
                <h4 className="font-mono text-[9px] text-cyan-400 tracking-[0.2em] uppercase mb-1">Target Sectors</h4>
                <h2 className="font-sans text-xl md:text-2xl font-bold text-white uppercase tracking-wide">Deployment Matrices</h2>
            </div>

            {/* Grid Allocation Framework */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-mono">
                {verticals.map((v) => (
                    <div
                        key={v.metric}
                        className="bg-[#0b0f19]/40 backdrop-blur-md border border-white/10 p-6 flex flex-col justify-between hover:border-cyan-500/40 transition-all duration-300 group relative overflow-hidden"
                    >
                        {/* Background Corner Glow Accent */}
                        <div className="absolute top-0 right-0 w-16 h-16 bg-cyan-500/5 blur-xl group-hover:bg-cyan-500/10 transition-all" />

                        <div>
                            <div className="text-xs text-cyan-500/60 mb-4">{v.metric}//</div>
                            <h3 className="font-sans text-base font-bold text-white uppercase mb-2 tracking-wide group-hover:text-cyan-400 transition-colors">
                                {v.title}
                            </h3>
                            <p className="text-[11px] leading-relaxed text-slate-400 uppercase">
                                {v.desc}
                            </p>
                        </div>

                        {/* Tactical Interactive Prompt Indicator */}
                        <div className="text-[9px] text-slate-500 mt-6 flex items-center gap-1 group-hover:text-cyan-400 transition-colors">
                            SYS.READY <span className="opacity-0 group-hover:opacity-100 transition-opacity">→</span>
                        </div>
                    </div>
                ))}
            </div>
        </section>
    );
};