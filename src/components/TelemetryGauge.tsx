import React, { useState, useEffect } from 'react';

export const TelemetryGauge: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([
    "[INFO] YOLOv8: Processing Frame 14092...",
    "[STATUS] Node Sync: 12ms",
    "[ALERT] Context Engine: Risk Recalculated - ELEVATED",
    "[INFO] Identity Match: 99.8% Confidence",
    "[STATUS] Buffer Cleared. Ready."
  ]);

  useEffect(() => {
    const logPool = [
      "[INFO] Sensor Array 4: Calibration Complete",
      "[WARN] Unrecognized pattern in Sector 7",
      "[INFO] YOLOv8: Processing Frame 14093...",
      "[STATUS] Node Sync: 11ms",
      "[ALERT] Context Engine: Risk Recalculated - HIGH"
    ];
    const interval = setInterval(() => {
      setLogs(prev => [...prev.slice(1), logPool[Math.floor(Math.random() * logPool.length)]]);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-panel p-8 rounded-none relative overflow-hidden flex flex-col border border-white/10 hover:border-cyan-500/30 transition-colors duration-500 hud-brackets hud-brackets-inner w-full">
      <div className="absolute top-2 left-2 text-[8px] font-mono text-cyan-500/50">[SYS_OPR_2026]</div>
      <div className="absolute top-2 right-2 text-[8px] font-mono text-cyan-500/50">[LAT: 25.5941 N]</div>
      <div className="absolute bottom-2 left-2 text-[8px] font-mono text-cyan-500/50">[LON: 85.1376 E]</div>
      <div className="absolute bottom-2 right-2 text-[8px] font-mono text-cyan-500/50">OPT: ACTIVE</div>

      {/* Laser Scan Line simulation */}
      <div className="absolute inset-0 pointer-events-none z-10 flex">
        <div className="h-[1px] w-full bg-cyan-500/50 shadow-[0_0_10px_#00c8ff] animate-scan-line"></div>
      </div>

      <div className="flex justify-between items-center mb-8 relative z-20 border-b border-white/10 pb-4">
        <span className="font-mono text-xs tracking-widest uppercase text-cyan-500">Command Center</span>
        <span className="text-amber-500 font-mono text-[10px] animate-pulse">● SENSORS_ACTIVE</span>
      </div>

      <div className="flex-grow flex items-center justify-center relative mb-8 z-20 min-h-[220px]">
        <div className="w-52 h-52 rounded-full border border-dashed border-white/20 relative flex items-center justify-center">
          <div className="absolute inset-2 border border-cyan-500/30 rounded-full border-t-transparent animate-spin" style={{ animationDuration: '8s' }}></div>
          <div className="text-center bg-black/60 backdrop-blur-sm rounded-full w-32 h-32 flex flex-col items-center justify-center border border-white/5 z-10">
            <div className="font-mono text-3xl font-bold text-white mb-1">98.2%</div>
            <div className="font-mono text-[8px] text-cyan-500 tracking-[0.2em] uppercase">Threat Context</div>
          </div>
        </div>
      </div>

      {/* Embedded Systems Log Stream */}
      <div className="bg-[#030712] font-mono text-[9px] p-3 rounded-none border border-white/10 h-32 overflow-hidden relative z-20 text-[#94a3b8] shadow-inner text-left">
        <div className="absolute top-0 left-0 w-full bg-white/5 p-1 px-2 border-b border-white/10 flex justify-between z-30 backdrop-blur-md">
          <span className="text-cyan-500 uppercase tracking-widest">SYS.LOG</span>
          <span className="text-white/30">v2.4.1</span>
        </div>
        <div className="flex flex-col gap-1.5 mt-6 relative z-20">
          {logs.map((log, idx) => (
            <div key={idx} className={log.includes('ALERT') || log.includes('WARN') ? 'text-amber-500' : ''}>
              <span className="text-cyan-500 mr-2">&gt;</span>{log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};