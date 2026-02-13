// Resolve ambiguity between WPF and WinForms types
// (caused by UseWindowsForms=true for system tray NotifyIcon)
global using Application = System.Windows.Application;
global using UserControl = System.Windows.Controls.UserControl;
global using MessageBox = System.Windows.MessageBox;
global using Color = System.Windows.Media.Color;
global using Point = System.Windows.Point;
global using ComboBox = System.Windows.Controls.ComboBox;
global using TextBox = System.Windows.Controls.TextBox;
global using Button = System.Windows.Controls.Button;
