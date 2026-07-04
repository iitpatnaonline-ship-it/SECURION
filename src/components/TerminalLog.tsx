import React, { useState, useEffect } from 'react';

const mockPool = [
  "[INFO] YOLOv8: Processing Frame 14092...",
  "[STATUS] Node Sync: 12ms",
  "[ALERT] Context Engine: Risk Recalculated - ELEVATED",
  "[INFO] Identity Match: 99.8% Confidence",
  "[STATUS] Buffer Cleared. Ready.",
  "[INFO] Sensor Array 4: Calibration Complete",
  "[WARN] Unrecognized pattern in Sector 7",
  "[INFO] Camera Stream Layer 1: Active Connection Established",
  "[STATUS] Memory Buffer Isolation at 0.04%"
];

export const TerminalLog: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([
    "> [INFO] SECURION Initializing Neural Network...",
    "> [STATUS] Local Node Core Sync: Ready"
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      const randomLine = mockPool[Math.floor(Math.random() * mockPool.length)];
      setLogs((prev) => {
        const updated = [...prev, `> ${randomLine}`];
        // Memory leak na ho isliye screen par maximum 6 logs hi rakhenge
        return updated.slice(-6);
      });
    }, 1200); // Har 1.2 second mein naya tactical log generate hoga

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-[#030712] font-mono p-4 border border-white/10 w-80 rounded-none shadow-inner text-left mt-4 relative">
      <div className="flex justify-between items-center border-b border-white/10 pb-1.5 mb-3">
        <span className="text-cyan-400 text-[10px] tracking-widest uppercase">SYS.LOG</span>
        <span className="text-white/30 text-[9px]">v2.4.1</span>
      </div>
      
      <div className="flex flex-col gap-2 min-h-30px justify-end">
        {logs.map((log, index) => {
          // Color coding logs for elite look
          let textColor = "text-slate-400";
          if (log.includes("ALERT")) textColor = "text-amber-500";
          if (log.includes("WARN")) textColor = "text-red-500";
          if (log.includes("STATUS")) textColor = "text-cyan-400";

          return (
            <div key={index} className={`text-[10px] truncate ${textColor} transition-all duration-300`}>
              {log}
            </div>
          );
        })}
      </div>
    </div>
  );
};