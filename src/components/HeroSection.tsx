import React from 'react';

export const HeroSection: React.FC = () => {
  return (
    <div className="flex flex-col gap-6 text-left max-w-xl">
      {/* Top Tag Signal */}
      <div className="inline-flex items-center gap-2 bg-black/40 border border-cyan-500/30 px-3 py-1 w-fit font-mono">
        <span className="w-1.5 h-1.5 bg-amber-500 rounded-none animate-pulse" />
        <span className="text-[9px] text-cyan-400 tracking-[0.2em] uppercase">Active Intelligence Grid</span>
      </div>

      {/* Main High-Intelligence Heading Header */}
      <h1 className="font-sans text-3xl md:text-5xl font-bold text-white leading-tight uppercase tracking-tight">
        Smart AI Security That<br />
        <span className="text-white/40">Eliminates False Alarms.</span>
      </h1>

      {/* Left-Border Accent Typography Block */}
      <p className="font-mono text-xs text-slate-400 border-l-2 border-cyan-400/50 pl-4 uppercase leading-relaxed">
        SECURION analyzes identity, time, location, and conditions to reduce false alarms.
        A context-aware neural network for high-security environments.
      </p>

      {/* Dynamic Command CTA Buttons */}
      <div className="flex flex-wrap gap-4 mt-2 font-mono">
        <button className="bg-cyan-500/10 text-cyan-400 border border-cyan-400 text-[10px] tracking-widest px-6 py-3 hover:bg-cyan-400/20 transition-all uppercase cyber glow">
          Explore the System
        </button>
        <button className="bg-black/40 text-white/70 border border-white/20 text-[10px] tracking-widest px-6 py-3 hover:bg-white/10 transition-all uppercase">
          View Architecture
        </button>
      </div>
    </div>
  );
};