import { Navbar } from './components/Navbar';
import { HeroSection } from './components/HeroSection';
import { TelemetryGauge } from './components/TelemetryGauge';
import { FeaturesGrid } from './components/FeaturesGrid';
import { PipelineTrack } from './components/PipelineTrack';
import { VerticalsGrid } from './components/VerticalsGrid';
import { QueryLog } from './components/QueryLog';
import { ContactFooter } from './components/ContactFooter';

function App() {
  return (
    <div className="min-h-screen bg-[#051424] text-[#d4e4fa] relative overflow-x-hidden selection:bg-cyan-500/20 selection:text-cyan-500 antialiased">
      {/* Background Matrix Blueprint Grid System Line Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.015)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.015)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none z-0" />

      <Navbar />

      <div className="relative z-10 pt-32 max-w-7xl mx-auto px-6 md:px-16 flex flex-col items-center">

        {/* SECTION 1: Upper Cinematic Command Workspace Header Grid Split */}
        <div className="flex flex-col lg:flex-row items-stretch justify-between gap-12 w-full mb-16">
          <div className="w-full lg:w-1/2 flex items-center">
            <HeroSection />
          </div>
          <div className="w-full lg:w-1/2 flex">
            <TelemetryGauge />
          </div>
        </div>

        {/* SECTION 2: 6-Card Custom Smart Feature Reticle Matrix */}
        <FeaturesGrid />

        {/* SECTION 3: System Pipeline Track & Calculation Formula Metrics */}
        <div className="w-full my-8" id="how-it-works">
          <PipelineTrack />
        </div>

        {/* SECTION 4: Autonomous Security Engineering Statement Block */}
        <section className="py-16 w-full text-left border-t border-b border-white/5 my-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <h2 className="font-sans text-3xl font-bold text-white leading-tight uppercase tracking-tight">
              Autonomous Security<br /><span className="text-white/60">Engineered by One Mind.</span>
            </h2>
            <p className="font-mono text-sm text-[#94a3b8] border-l-2 border-cyan-500/50 pl-4 uppercase">
              SECURION is a next-generation decentralized AI threat detection grid built for ultra-low latency edge devices.
            </p>
          </div>
        </section>

        {/* SECTION 5: Priority Alpha/Beta/Gamma Sector Selection Cards */}
        <VerticalsGrid />

        {/* SECTION 6: Operational FAQ Query Logs Dropdown Matrix */}
        <QueryLog />

        {/* SECTION 7: Secure Request Access Channel Form & Bottom Logs Footer */}
        <ContactFooter />
      </div>
    </div>
  );
}

export default App;