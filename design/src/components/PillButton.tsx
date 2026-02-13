interface PillButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  children: React.ReactNode;
}

export function PillButton({ variant = 'primary', children, className = '', ...props }: PillButtonProps) {
  const baseStyles = 'px-6 py-3 rounded-full font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95';
  
  const variants = {
    primary: 'bg-[#70C6E8] text-white hover:bg-[#5AB8DD] shadow-md hover:shadow-lg',
    secondary: 'bg-white text-[#2A3B55] border-2 border-[#70C6E8] hover:bg-[#F0F8FF] shadow-md',
    danger: 'bg-[#FF6B9D] text-white hover:bg-[#FF5A8D] shadow-md hover:shadow-lg',
    success: 'bg-[#4ECDC4] text-white hover:bg-[#3DBCB3] shadow-md hover:shadow-lg',
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
