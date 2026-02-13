import { useState } from 'react';
import { RoundTable } from './RoundTable';
import { TrendingUp, DollarSign, Clock, BarChart3 } from 'lucide-react';
import mascotImage from 'figma:asset/29d18e8913a2b19e2d036e2bcc43e1be0d51ebb8.png';

export function Monitor() {
  const [portfolioData] = useState([
    { ticker: 'AAPL', quantity: '150', lastPrice: '$175.43', marketValue: '$26,314.50' },
    { ticker: 'GOOGL', quantity: '85', lastPrice: '$142.65', marketValue: '$12,125.25' },
    { ticker: 'MSFT', quantity: '200', lastPrice: '$378.91', marketValue: '$75,782.00' },
    { ticker: 'TSLA', quantity: '50', lastPrice: '$248.50', marketValue: '$12,425.00' },
  ]);

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-[#70C6E8] to-[#5AB8DD] rounded-3xl shadow-lg p-8 text-white relative overflow-hidden">
          <div className="absolute -top-4 -right-4 opacity-10">
            <TrendingUp className="w-32 h-32" />
          </div>
          <div className="relative">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-5 h-5" />
              <div className="font-bold text-sm uppercase tracking-wider opacity-90">Net Liquidation</div>
            </div>
            <div className="font-bold text-4xl mb-2">$126,646.75</div>
            <div className="bg-white/20 px-3 py-1 rounded-full inline-block font-bold text-sm">
              +2.34% Today 🎉
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8 relative overflow-hidden hover:shadow-xl transition-shadow">
          <div className="absolute -top-4 -right-4 opacity-5">
            <DollarSign className="w-32 h-32 text-[#70C6E8]" />
          </div>
          <div className="relative">
            <div className="flex items-center gap-2 mb-3">
              <DollarSign className="w-5 h-5 text-[#70C6E8]" />
              <div className="text-[#70C6E8] font-bold text-sm uppercase tracking-wider">Cash Available</div>
            </div>
            <div className="text-[#2A3B55] font-bold text-4xl mb-2">$873,353.25</div>
            <div className="text-[#A0B4C0] font-semibold text-sm">Ready for adventures! 💰</div>
          </div>
        </div>
        
        <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8 relative overflow-hidden hover:shadow-xl transition-shadow">
          <div className="absolute -top-4 -right-4 opacity-5">
            <Clock className="w-32 h-32 text-[#70C6E8]" />
          </div>
          <div className="relative">
            <div className="flex items-center gap-2 mb-3">
              <Clock className="w-5 h-5 text-[#70C6E8]" />
              <div className="text-[#70C6E8] font-bold text-sm uppercase tracking-wider">Last Update</div>
            </div>
            <div className="text-[#2A3B55] font-bold text-2xl mb-1">2/8/2026</div>
            <div className="text-[#2A3B55] font-bold text-2xl">4:41:20 PM</div>
          </div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-[#2A3B55] text-xl font-bold">Portfolio Holdings 📈</h3>
          <div className="bg-[#70C6E8] text-white px-4 py-2 rounded-full font-bold text-sm">
            {portfolioData.length} Positions
          </div>
        </div>
        
        <RoundTable
          headers={['Ticker', 'Quantity', 'Last Price', 'Market Value']}
          data={portfolioData.map(p => [
            <span className="font-bold text-[#70C6E8]">{p.ticker}</span>,
            p.quantity,
            p.lastPrice,
            <span className="font-bold text-[#2A3B55]">{p.marketValue}</span>
          ])}
        />
      </div>

      {/* Performance Chart Placeholder */}
      <div className="bg-white rounded-3xl shadow-lg border-2 border-[#E8F4F8] p-8">
        <div className="mb-6">
          <h3 className="text-[#2A3B55] text-xl font-bold">Portfolio Performance 📊</h3>
        </div>
        <div className="h-80 bg-gradient-to-br from-[#F0F8FF] to-[#E8F4F8] rounded-2xl flex items-center justify-center">
          <div className="text-center">
            <img 
              src={mascotImage} 
              alt="Mascot" 
              className="w-40 h-40 mx-auto mb-4 opacity-40"
            />
            <BarChart3 className="w-16 h-16 text-[#70C6E8] mx-auto mb-3 opacity-30" />
            <div className="text-[#A0B4C0] font-bold text-lg">Chart coming soon! 🎨</div>
          </div>
        </div>
      </div>
    </div>
  );
}
