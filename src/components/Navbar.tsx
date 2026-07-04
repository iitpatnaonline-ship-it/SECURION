import React from 'react';

export const Navbar: React.FC = () => {
  return (
    <nav className="fixed top-0 w-full z-50 bg-[#051424]/80 backdrop-blur-xl border-b border-white/10">
      <div className="flex justify-between items-center px-6 md:px-16 py-4 max-w-7xl mx-auto">
        {/* Brand Logo HUD Element */}
        <div className="font-sans text-lg font-bold tracking-tighter text-white flex items-center gap-2">
          <span className="text-cyan-400 font-mono text-sm">✦</span> SECURION
        </div>
        
        {/* Central Nav Links Map */}
        <div className="hidden md:flex gap-8 items-center">
          {['Features', 'How It Works', 'Live Dashboard', 'Solutions', 'Contact Us'].map((item) => (
            <a 
              key={item} 
              href={`#${item.toLowerCase().replace(/ /g, '-')}`} 
              className="font-mono text-[10px] tracking-widest text-slate-400 hover:text-cyan-400 transition-colors uppercase"
            >
              {item}
            </a>
          ))}
        </div>

        {/* Tactical Action Button */}
        <button className="hidden md:block bg-transparent border border-cyan-400 text-cyan-400 font-mono text-[10px] tracking-widest px-5 py-2 hover:bg-cyan-400/10 transition-all uppercase">
          Get Started Now
        </button>
      </div>
    </nav>
  );
};