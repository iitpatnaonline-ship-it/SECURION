import React from 'react';

const coreFeatures = [
    { icon: 'psychology', id: '01 // CORE', title: 'Smart Human Detection', desc: 'Analyzes identity, time, and location in real-time.' },
    { icon: 'security', id: '02 // EVAL', title: 'Intelligent Threat Assessment', desc: 'Calculates dynamic risk level using our contextual algorithm.' },
    { icon: 'filter_alt', id: '03 // FLTR', title: 'False Alarm Reduction', desc: 'Advanced filtering, thresholding and silent logging.' },
    { icon: 'layers', id: '04 // VLD', title: 'Multi-frame Confirmation', desc: 'Validates detections across multiple frames before alerting.' },
    { icon: 'memory', id: '05 // LOGIC', title: 'Intelligent Decision Engine', desc: 'Generates alerts only for genuine threats, not just events.' },
    { icon: 'show_chart', id: '06 // MON', title: 'Real-time Monitoring', desc: 'Processes and monitors live feeds instantly.' }
];

export const FeaturesGrid: React.FC = () => {
    return (
        <section className="py-12 px-6 md:px-16 relative z-10 w-full" id="features">
            <div className="max-w-7xl mx-auto mb-8 text-center">
                <h2 className="font-sans text-2xl text-white uppercase tracking-wider font-semibold">Smart Security Features</h2>
            </div>
            <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {coreFeatures.map((f) => (
                    <div key={f.id} className="glass-panel p-6 rounded-none flex gap-4 items-start border-l-2 border-l-cyan-500/50 hover:border-l-cyan-500 feature-card transition-all duration-500 hover:translate-y-[-2px] hover:shadow-[0_0_25px_rgba(0,200,255,0.15)] hover:border-cyan-500 text-left">
                        <div className="reticle reticle-tl"></div><div className="reticle reticle-tr"></div>
                        <div className="reticle reticle-bl"></div><div className="reticle reticle-br"></div>
                        <span className="text-cyan-500 font-mono text-2xl">⚡</span>
                        <div>
                            <div className="text-[10px] font-mono text-cyan-500/70 mb-1">{f.id}</div>
                            <h3 className="font-sans text-sm font-semibold text-white uppercase tracking-wider mb-2">{f.title}</h3>
                            <p className="font-mono text-xs text-[#94a3b8]">{f.desc}</p>
                        </div>
                    </div>
                ))}
            </div>
        </section>
    );
};