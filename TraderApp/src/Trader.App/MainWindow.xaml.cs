using System.Windows;
using Trader.App.ViewModels;

namespace Trader.App;

public partial class MainWindow : Window
{
    public MainWindow(ShellViewModel shellViewModel)
    {
        InitializeComponent();
        DataContext = shellViewModel;
    }
}
