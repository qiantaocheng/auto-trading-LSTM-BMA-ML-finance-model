interface RoundInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export function RoundInput({ label, className = '', ...props }: RoundInputProps) {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-[#2A3B55] font-semibold text-sm px-2">
          {label}
        </label>
      )}
      <input
        className={`bg-white text-[#2A3B55] border-2 border-[#E8F4F8] px-5 py-3 rounded-full placeholder:text-[#A0B4C0] focus:outline-none focus:border-[#70C6E8] focus:shadow-lg transition-all ${className}`}
        {...props}
      />
    </div>
  );
}
