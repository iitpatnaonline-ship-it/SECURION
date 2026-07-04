import React, { useState } from 'react';

export const ContactFooter: React.FC = () => {
    const [email, setEmail] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        alert(`ACCESS REQUEST REGISTERED FOR: ${email.toUpperCase()}`);
        setEmail('');
    };

    return (
        <section className="w-full max-w-7xl mx-auto px-6 md:px-16 mt-24 border-t border-white/5 pt-16">
            <div className="flex flex-col lg:flex-row items-stretch justify-between gap-12">
                {/* Left Side Info Grid */}
                <div className="w-full lg:w-1/2 text-left flex flex-col justify-between">
                    <div>
                        <h4 className="font-mono text-[9px] text-cyan-400 tracking-[0.2em] uppercase mb-1">System Access</h4>
                        <h2 className="font-sans text-xl md:text-2xl font-bold text-white uppercase tracking-wide mb-4">Request Deployment</h2>
                        <p className="font-mono text-xs text-slate-400 uppercase leading-relaxed max-w-md">
                            Connect with our operational technical integration team to deploy the SECURION neural network pipeline to your current surveillance hardware infrastructure.
                        </p>
                    </div>

                    {/* Status Indicators */}
                    <div className="font-mono text-[10px] text-slate-500 mt-8 lg:mt-0 flex gap-6 uppercase">
                        <div>Node Status: <span className="text-emerald-500">Online</span></div>
                        <div>Core Latency: <span className="text-cyan-400">12ms</span></div>
                    </div>
                </div>

                {/* Right Side Form Panel */}
                <div className="w-full lg:w-1/2 bg-[#0b0f19]/40 backdrop-blur-md border border-white/10 p-6 md:p-8 font-mono">
                    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                        <div className="text-left">
                            <label className="text-[9px] text-cyan-400 tracking-wider uppercase block mb-1.5">Secure Email Channel</label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="ENTER SECURE EMAIL NODE..."
                                required
                                className="w-full bg-black/40 border border-white/10 p-3 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-400/60 font-mono tracking-wide rounded-none"
                            />
                        </div>

                        <button
                            type="submit"
                            className="w-full bg-cyan-500/10 text-cyan-400 border border-cyan-400 text-[10px] tracking-widest py-3 hover:bg-cyan-400/20 transition-all uppercase font-bold"
                        >
                            Initialize Node Connection
                        </button>
                    </form>
                </div>
            </div>

            {/* Global Terminal Footer Matrix */}
            <footer className="mt-24 border-t border-white/5 pt-8 pb-4 flex flex-col md:flex-row justify-between items-center gap-4 font-mono text-[9px] text-slate-600 tracking-wider uppercase">
                <div>© 2026 SECURION SYSTEM ARCHITECTURE. ALL RIGHTS RESERVED.</div>
                <div className="flex gap-6">
                    <span className="hover:text-cyan-400/60 cursor-pointer">Protocol Logs</span>
                    <span className="hover:text-cyan-400/60 cursor-pointer">Security Core</span>
                    <span className="text-cyan-500/50">SYS_VER_4.2.0</span>
                </div>
            </footer>
        </section>
    );
};