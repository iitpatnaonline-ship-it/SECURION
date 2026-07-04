import React from 'react';

export const PipelineTrack: React.FC = () => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full mt-12 text-left font-mono">
      {/* Steps Pipeline Matrix Map */}
      <div className="lg:col-span-2 glass-panel p-6 rounded-none border border-white/10 hud-brackets flex flex-col justify-between">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h4 className="text-[9px] text-cyan-500 tracking-[0.2em] uppercase mb-1">Pipeline Architecture</h4>
            <h3 className="font-sans text-base font-bold text-white uppercase tracking-wider">System Execution Track</h3>
          </div>
          <div className="text-[8px] text-white/30 border border-white/10 px-2 py-0.5">v.3.2 ACTIVE</div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 py-2">
          {[
            { stp: '01', label: 'Camera Input' },
            { stp: '02', label: 'Face Recog' },
            { stp: '03', label: 'Verify Ident' },
            { stp: '04', label: 'Behavior Check' },
            { stp: '05', label: 'Risk Score' }
          ].map((s, idx) => (
            <React.Fragment key={s.stp}>
              <div className="flex flex-col items-center text-center">
                <span className="text-[8px] text-cyan-500/70 mb-1">STP.{s.stp}</span>
                <div className="w-10 h-10 border border-cyan-500/30 flex items-center justify-center bg-cyan-500/5 text-cyan-400 text-xs shadow-[0_0_10px_rgba(0,200,255,0.05)]">
                  {idx === 4 ? '✦' : '■'}
                </div>
                <span className="text-[8px] text-slate-400 mt-1 uppercase leading-none">{s.label}</span>
              </div>
              {idx < 4 && <span className="text-cyan-500/30 hidden sm:block">→</span>}
            </React.Fragment>
          ))}
        </div>

        {/* Dynamic Matrix Weight Assignment Block */}
        <div className="flex flex-col sm:flex-row gap-4 mt-6 pt-4 border-t border-white/5">
          <div className="flex-1">
            <div className="text-[9px] text-cyan-400 uppercase mb-1">Algorithmic Weighting</div>
            <div className="text-sm text-white font-bold tracking-wide">RISK = IDENT + ZONE + TIME</div>
          </div>
          <div className="flex items-center gap-2 text-[8px] text-slate-400">
            <span className="bg-black/40 border border-white/10 px-2 py-1">IDENT_W: 50%</span>
            <span className="bg-black/40 border border-white/10 px-2 py-1">ZONE_W: 30%</span>
          </div>
        </div>
      </div>

      {/* Telemetry Analytical Score Array Module */}
      <div className="lg:col-span-1 glass-panel p-6 rounded-none border border-white/10 hud-brackets flex flex-col justify-between">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-[9px] text-cyan-500 tracking-[0.2em] uppercase">Telemetry Indicators</h4>
          <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
        </div>

        <div className="flex items-center justify-center py-2">
          <div className="text-center">
            <div className="text-3xl font-bold text-white tracking-tighter">87 <span className="text-xs text-slate-500">/100</span></div>
            <div className="text-[8px] text-rose-500 border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 uppercase tracking-widest mt-1 font-bold">CRITICAL</div>
          </div>
        </div>

        <div className="flex flex-col gap-1.5 text-[9px] text-slate-400 uppercase pt-2 border-t border-white/5">
          <div className="flex justify-between"><span>Ident: <span className="text-white">Unknown</span></span><span className="text-rose-400">+50</span></div>
          <div className="flex justify-between"><span>Zone: <span className="text-white">Rstr.Zone</span></span><span className="text-amber-400">+30</span></div>
          <div className="flex justify-between"><span>Time: <span className="text-white">Night_Op</span></span><span className="text-cyan-400">+20</span></div>
        </div>
      </div>
    </div>
  );
};