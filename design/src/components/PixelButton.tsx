interface PixelButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger';
  children: React.ReactNode;
}

export function PixelButton({ variant = 'primary', children, className = '', ...props }: PixelButtonProps) {
  const baseStyles = 'px-6 py-3 font-mono transition-all duration-100 disabled:opacity-50 disabled:cursor-not-allowed active:translate-y-[1px]';
  
  const variants = {
    primary: 'bg-[#92B0CB] text-white border-2 border-[#7A98B5] shadow-[3px_3px_0_#6A88A5] hover:bg-[#A2C0DB] active:shadow-none',
    secondary: 'bg-white text-[#2C3E50] border-2 border-[#92B0CB] shadow-[3px_3px_0_#7A98B5] hover:bg-[#F8FAFC] active:shadow-none',
    danger: 'bg-[#E74C3C] text-white border-2 border-[#C0392B] shadow-[3px_3px_0_#A93226] hover:bg-[#EC7063] active:shadow-none',
  };
  
  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}