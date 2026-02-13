interface PixelInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export function PixelInput({ label, className = '', ...props }: PixelInputProps) {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-[#2C3E50] font-mono text-sm uppercase tracking-wider">
          {label}
        </label>
      )}
      <input
        className={`bg-white text-[#2C3E50] border-2 border-[#E0E6ED] px-4 py-3 font-mono placeholder:text-[#B0BEC5] focus:outline-none focus:border-[#92B0CB] focus:shadow-[0_0_0_3px_rgba(146,176,203,0.1)] transition-all ${className}`}
        {...props}
      />
    </div>
  );
}