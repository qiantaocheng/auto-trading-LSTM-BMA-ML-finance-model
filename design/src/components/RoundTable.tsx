interface RoundTableProps {
  headers: string[];
  data: (string | number | React.ReactNode)[][];
  emptyMessage?: string;
  emptyImage?: string;
}

export function RoundTable({ headers, data, emptyMessage = 'No data yet!', emptyImage }: RoundTableProps) {
  return (
    <div className="bg-white rounded-2xl overflow-hidden shadow-sm border-2 border-[#E8F4F8]">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gradient-to-r from-[#70C6E8] to-[#5AB8DD]">
              {headers.map((header, idx) => (
                <th 
                  key={idx} 
                  className="px-6 py-4 text-left text-white font-bold text-sm uppercase tracking-wide"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.length === 0 ? (
              <tr>
                <td 
                  colSpan={headers.length} 
                  className="px-6 py-16 text-center"
                >
                  {emptyImage && (
                    <img 
                      src={emptyImage} 
                      alt="Empty state" 
                      className="w-32 h-32 mx-auto mb-4 opacity-60"
                    />
                  )}
                  <div className="text-[#A0B4C0] font-semibold">{emptyMessage}</div>
                </td>
              </tr>
            ) : (
              data.map((row, rowIdx) => (
                <tr 
                  key={rowIdx} 
                  className="border-b border-[#E8F4F8] hover:bg-[#F0F8FF] transition-colors"
                >
                  {row.map((cell, cellIdx) => (
                    <td 
                      key={cellIdx} 
                      className="px-6 py-4 text-[#2A3B55] font-semibold"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
