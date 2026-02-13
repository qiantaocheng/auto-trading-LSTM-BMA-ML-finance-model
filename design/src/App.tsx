import { useState } from 'react';
import { DirectPrediction } from './components/DirectPrediction';
import { Monitor } from './components/Monitor';
import { Database } from './components/Database';
import { Activity } from 'lucide-react';
import mascotImage from 'figma:asset/29d18e8913a2b19e2d036e2bcc43e1be0d51ebb8.png';

type Tab = 'direct' | 'monitor' | 'database';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('direct');

  return (
    <div className="min-h-screen bg-[#F0F8FF] p-6 md:p-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 bg-white rounded-3xl shadow-xl border-2 border-[#E8F4F8] p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-[#70C6E8]/10 to-transparent rounded-full -mr-32 -mt-32"></div>
          
          <div className="flex items-center justify-between relative">
            <div className="flex items-center gap-6">
              <div className="bg-gradient-to-br from-[#70C6E8] to-[#5AB8DD] p-4 rounded-3xl shadow-lg">
                <Activity className="w-12 h-12 text-white" />
              </div>
              <div>
                <h1 className="text-[#2A3B55] text-3xl md:text-4xl font-bold mb-1">
                  Trader AutoPilot
                </h1>
                <p className="text-[#70C6E8] font-semibold">
                  Your friendly trading companion 
                </p>
              </div>
            </div>
            
            {/* Mascot in top-right corner */}
            <div className="hidden md:block">
           
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mb-8 flex flex-wrap gap-3">
          <button
            onClick={() => setActiveTab('direct')}
            className={`px-8 py-4 rounded-full font-bold transition-all duration-200 ${
              activeTab === 'direct'
                ? 'bg-gradient-to-r from-[#70C6E8] to-[#5AB8DD] text-white shadow-lg scale-105'
                : 'bg-white text-[#2A3B55] border-2 border-[#E8F4F8] hover:border-[#70C6E8] hover:shadow-md'
            }`}
          >
             Direct Prediction
          </button>
          
          <button
            onClick={() => setActiveTab('monitor')}
            className={`px-8 py-4 rounded-full font-bold transition-all duration-200 ${
              activeTab === 'monitor'
                ? 'bg-gradient-to-r from-[#70C6E8] to-[#5AB8DD] text-white shadow-lg scale-105'
                : 'bg-white text-[#2A3B55] border-2 border-[#E8F4F8] hover:border-[#70C6E8] hover:shadow-md'
            }`}
          >
             Monitor
          </button>
          
          <button
            onClick={() => setActiveTab('database')}
            className={`px-8 py-4 rounded-full font-bold transition-all duration-200 ${
              activeTab === 'database'
                ? 'bg-gradient-to-r from-[#70C6E8] to-[#5AB8DD] text-white shadow-lg scale-105'
                : 'bg-white text-[#2A3B55] border-2 border-[#E8F4F8] hover:border-[#70C6E8] hover:shadow-md'
            }`}
          >
             Database
          </button>
        </div>

        {/* Main Content */}
        <div>
          {activeTab === 'direct' && <DirectPrediction />}
          {activeTab === 'monitor' && <Monitor />}
          {activeTab === 'database' && <Database />}
        </div>

        {/* Footer */}
        <div className="mt-16 text-center">
          <div className="inline-flex items-center gap-3 bg-white border-2 border-[#E8F4F8] px-8 py-4 rounded-full shadow-md">
            <div className="w-2 h-2 rounded-full bg-[#4ECDC4] animate-pulse"></div>
            <p className="text-[#70C6E8] font-bold">
              System Online • All Good! • Happy Trading! 
            </p>
            <div className="w-2 h-2 rounded-full bg-[#4ECDC4] animate-pulse"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
