using System.Windows.Controls;
using Trader.App.ViewModels.Pages;

namespace Trader.App.Views;

public partial class ConnectionView : UserControl
{
    public ConnectionView()
    {
        InitializeComponent();
        Loaded += ConnectionView_Loaded;
    }

    private void ConnectionView_Loaded(object sender, System.Windows.RoutedEventArgs e)
    {
        // Pre-fill PasswordBox from ViewModel (PasswordBox can't be bound)
        if (DataContext is ConnectionViewModel vm && !string.IsNullOrEmpty(vm.IbPassword))
        {
            PasswordBox.Password = vm.IbPassword;
        }
    }

    private void PasswordBox_PasswordChanged(object sender, System.Windows.RoutedEventArgs e)
    {
        if (DataContext is ConnectionViewModel vm)
        {
            vm.IbPassword = PasswordBox.Password;
        }
    }
}
