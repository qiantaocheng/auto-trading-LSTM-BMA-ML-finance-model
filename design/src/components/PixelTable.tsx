interface PixelTableProps {
  headers: string[];
  data: (string | number | React.ReactNode)[][];
  emptyMessage?: string;
}

export function PixelTable({ headers, data, emptyMessage = 'No data available' }: PixelTableProps) {
  return (
    <div className="border border-[#E0E6ED] bg-white overflow-hidden rounded-sm">
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-sm">
          <thead>
            <tr className="bg-[#F8FAFC] border-b-2 border-[#E0E6ED]">
              {headers.map((header, idx) => (
                <th 
                  key={idx} 
                  className="px-6 py-4 text-left uppercase tracking-wider text-[#2C3E50] font-medium"
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
                  className="px-6 py-12 text-center text-[#B0BEC5]"
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              data.map((row, rowIdx) => (
                <tr 
                  key={rowIdx} 
                  className="border-b border-[#E0E6ED] hover:bg-[#F8FAFC] transition-colors"
                >
                  {row.map((cell, cellIdx) => (
                    <td 
                      key={cellIdx} 
                      className="px-6 py-4 text-[#2C3E50]"
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