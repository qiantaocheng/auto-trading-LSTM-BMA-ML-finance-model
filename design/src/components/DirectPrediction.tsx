import { useState } from 'react';
import { PillButton } from './PillButton';
import { RoundInput } from './RoundInput';
import { RoundTable } from './RoundTable';
import { Play, Download, Sparkles } from 'lucide-react';
import mascotImage from 'figma:asset/29d18e8913a2b19e2d036e2bcc43e1be0d51ebb8.png';

export function DirectPrediction() {
  const [isRunning, setIsRunning] = useState(false);
  const [predictions, setPredictions] = useState<any[]>([]);

  const handleRunPrediction = () => {
    setIsRunning(true);
    
    // Simulate prediction process
    setTimeout(() => {
      setPredictions([
        { ticker: 'AAPL', score: '87.5', ema: '+2.34%' },
        { ticker: 'GOOGL', score: '92.1', ema: '+3.12%' },
        { ticker: 'MSFT', score: '85.3', ema: '+1.89%' },
        { ticker: 'TSLA', score: '78.9', ema: '-0.54%' },
        { ticker: 'AMZN', score: '90.2', ema: '+2.67%' },
      ]);
      setIsRunning(false);
    }, 1500);
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="bg-gradient-to-br from-[#70C6E8] to-[#5AB8DD] p-3 rounded-2xl">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-[#2A3B55] text-xl font-bold">Feed the Data 🎯</h3>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-end">
          <div className="lg:col-span-2">
            <RoundInput
              label="Snapshot ID"
              value="b35a35db-352b-43d8-ace8-4a5467dc1da5"
              readOnly
            />
          </div>
          
          <PillButton 
            variant="primary" 
            onClick={handleRunPrediction}
            disabled={isRunning}
            className="flex items-center justify-center gap-2 w-full"
          >
            <Play className="w-5 h-5" />
            {isRunning ? 'Running Magic...' : 'Run Prediction'}
          </PillButton>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-6 hover:shadow-xl transition-shadow">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 rounded-full bg-[#70C6E8] animate-pulse"></div>
            <div className="text-[#70C6E8] font-bold text-xs uppercase tracking-wider">Last Run</div>
          </div>
          <div className="text-[#2A3B55] font-bold text-lg">2/8/2026 14:23:15</div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-6 hover:shadow-xl transition-shadow">
          <div className="flex items-center gap-3 mb-2">
            <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-[#FFB84D] animate-pulse' : 'bg-[#4ECDC4]'}`}></div>
            <div className="text-[#70C6E8] font-bold text-xs uppercase tracking-wider">Status</div>
          </div>
          <div className="text-[#2A3B55] font-bold text-lg">{isRunning ? '🔄 Running' : '✓ Complete'}</div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-6 hover:shadow-xl transition-shadow">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 rounded-full bg-[#70C6E8]"></div>
            <div className="text-[#70C6E8] font-bold text-xs uppercase tracking-wider">Output</div>
          </div>
          <button className="text-[#2A3B55] font-bold text-lg hover:text-[#70C6E8] transition-colors flex items-center gap-2">
            <Download className="w-5 h-5" />
            Download.xlsx
          </button>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8">
        <div className="flex items-center gap-4 mb-6">
          <h3 className="text-[#2A3B55] text-xl font-bold">Prediction Results 📊</h3>
        </div>
        
        <RoundTable
          headers={['Ticker', 'Score', 'EMA (4d)']}
          data={predictions.map(p => [
            <span className="font-bold text-[#70C6E8]">{p.ticker}</span>,
            <span className="font-mono">{p.score}</span>,
            <span className={`font-bold ${p.ema.startsWith('+') ? 'text-[#4ECDC4]' : 'text-[#FF6B9D]'}`}>{p.ema}</span>
          ])}
          emptyMessage="Waiting for data snacks... 🍪"
          emptyImage={mascotImage}
        />
      </div>
    </div>
  );
}
