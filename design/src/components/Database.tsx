import { useState } from 'react';
import { PillButton } from './PillButton';
import { RoundInput } from './RoundInput';
import { Plus, Trash2, RefreshCw, Database as DatabaseIcon } from 'lucide-react';
import mascotImage from 'figma:asset/29d18e8913a2b19e2d036e2bcc43e1be0d51ebb8.png';

export function Database() {
  const [tickerInput, setTickerInput] = useState('');
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  const [tickers, setTickers] = useState([
    { ticker: 'AAPL', tag: 'TECH', source: 'MANUAL', added: '2/1/2026' },
    { ticker: 'GOOGL', tag: 'TECH', source: 'AUTO', added: '2/2/2026' },
    { ticker: 'MSFT', tag: 'TECH', source: 'MANUAL', added: '2/3/2026' },
    { ticker: 'TSLA', tag: 'AUTO', source: 'AUTO', added: '2/4/2026' },
    { ticker: 'AMZN', tag: 'ECOM', source: 'MANUAL', added: '2/5/2026' },
  ]);

  const handleAddManual = () => {
    if (tickerInput.trim()) {
      const newTicker = {
        ticker: tickerInput.toUpperCase(),
        tag: 'MANUAL',
        source: 'MANUAL',
        added: new Date().toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric' })
      };
      setTickers([newTicker, ...tickers]);
      setTickerInput('');
    }
  };

  const handleDeleteSelected = () => {
    setTickers(tickers.filter((_, idx) => !selectedRows.has(idx)));
    setSelectedRows(new Set());
  };

  const handleRefresh = () => {
    // Simulate refresh
    console.log('Refreshing database...');
  };

  const toggleRowSelection = (idx: number) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(idx)) {
      newSelected.delete(idx);
    } else {
      newSelected.add(idx);
    }
    setSelectedRows(newSelected);
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="bg-gradient-to-br from-[#70C6E8] to-[#5AB8DD] p-3 rounded-2xl">
            <DatabaseIcon className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-[#2A3B55] text-xl font-bold">Manage Your Tickers 🎮</h3>
        </div>
        
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="flex-1">
            <RoundInput
              placeholder="Enter ticker symbol..."
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAddManual()}
            />
          </div>
          
          <div className="flex gap-3 flex-wrap">
            <PillButton 
              variant="primary" 
              onClick={handleAddManual}
              className="flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              Add
            </PillButton>
            
            <PillButton 
              variant="danger" 
              onClick={handleDeleteSelected}
              disabled={selectedRows.size === 0}
              className="flex items-center gap-2"
            >
              <Trash2 className="w-5 h-5" />
              Delete
            </PillButton>
            
            <PillButton 
              variant="secondary" 
              onClick={handleRefresh}
              className="flex items-center gap-2"
            >
              <RefreshCw className="w-5 h-5" />
              Refresh
            </PillButton>
          </div>
        </div>
      </div>

      {/* Database Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-[#70C6E8] to-[#5AB8DD] rounded-3xl shadow-lg p-6 text-white">
          <div className="font-bold text-sm uppercase tracking-wider opacity-90 mb-2">Total</div>
          <div className="font-bold text-4xl">{tickers.length}</div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-6 hover:shadow-xl transition-shadow">
          <div className="text-[#70C6E8] font-bold text-sm uppercase tracking-wider mb-2">Manual</div>
          <div className="text-[#2A3B55] font-bold text-4xl">
            {tickers.filter(t => t.source === 'MANUAL').length}
          </div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-6 hover:shadow-xl transition-shadow">
          <div className="text-[#70C6E8] font-bold text-sm uppercase tracking-wider mb-2">Auto</div>
          <div className="text-[#2A3B55] font-bold text-4xl">
            {tickers.filter(t => t.source === 'AUTO').length}
          </div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-6 hover:shadow-xl transition-shadow">
          <div className="text-[#70C6E8] font-bold text-sm uppercase tracking-wider mb-2">Selected</div>
          <div className="text-[#2A3B55] font-bold text-4xl">{selectedRows.size}</div>
        </div>
      </div>

      {/* Tickers Table */}
      <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8">
        <div className="mb-6">
          <h3 className="text-[#2A3B55] text-xl font-bold">Ticker Database 🗃️</h3>
        </div>
        
        <div className="bg-white rounded-2xl overflow-hidden shadow-sm border-2 border-[#E8F4F8]">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gradient-to-r from-[#70C6E8] to-[#5AB8DD]">
                  <th className="px-6 py-4 text-left text-white font-bold text-sm uppercase tracking-wide w-16">
                    <span className="text-2xl">☐</span>
                  </th>
                  <th className="px-6 py-4 text-left text-white font-bold text-sm uppercase tracking-wide">
                    Ticker
                  </th>
                  <th className="px-6 py-4 text-left text-white font-bold text-sm uppercase tracking-wide">
                    Tag
                  </th>
                  <th className="px-6 py-4 text-left text-white font-bold text-sm uppercase tracking-wide">
                    Source
                  </th>
                  <th className="px-6 py-4 text-left text-white font-bold text-sm uppercase tracking-wide">
                    Added
                  </th>
                </tr>
              </thead>
              <tbody>
                {tickers.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-16 text-center">
                      <img 
                        src={mascotImage} 
                        alt="Empty state" 
                        className="w-32 h-32 mx-auto mb-4 opacity-60"
                      />
                      <div className="text-[#A0B4C0] font-semibold">No tickers yet! Let's catch some! 🎣</div>
                    </td>
                  </tr>
                ) : (
                  tickers.map((ticker, idx) => (
                    <tr 
                      key={idx} 
                      className={`border-b border-[#E8F4F8] hover:bg-[#F0F8FF] transition-colors ${selectedRows.has(idx) ? 'bg-[#E8F4F8]' : ''}`}
                    >
                      <td className="px-6 py-4">
                        <button 
                          onClick={() => toggleRowSelection(idx)}
                          className="cursor-pointer hover:scale-110 transition-transform text-2xl"
                        >
                          {selectedRows.has(idx) ? '☑' : '☐'}
                        </button>
                      </td>
                      <td className="px-6 py-4 text-[#70C6E8] font-bold">
                        {ticker.ticker}
                      </td>
                      <td className="px-6 py-4 text-[#2A3B55] font-semibold">
                        <span className="bg-[#F0F8FF] px-3 py-1 rounded-full text-sm">
                          {ticker.tag}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-[#2A3B55] font-semibold">
                        <span className={`px-3 py-1 rounded-full text-sm ${ticker.source === 'MANUAL' ? 'bg-[#FFE8F0]' : 'bg-[#E8F4F8]'}`}>
                          {ticker.source}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-[#2A3B55] font-semibold">
                        {ticker.added}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
