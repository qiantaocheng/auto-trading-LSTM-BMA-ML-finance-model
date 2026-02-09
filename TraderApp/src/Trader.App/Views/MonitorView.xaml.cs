using System.Windows;
using System.Windows.Controls;
using Trader.App.ViewModels.Pages;

namespace Trader.App.Views;

public partial class MonitorView : UserControl
{
    public MonitorView()
    {
        InitializeComponent();
    }

    private void ConnectButton_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is MonitorViewModel vm)
        {
            vm.Connect();
        }
    }

    private void DisconnectButton_Click(object sender, RoutedEventArgs e)
    {
        if (DataContext is MonitorViewModel vm)
        {
            vm.Disconnect();
        }
    }

    private void TradingModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (DataContext is MonitorViewModel vm && sender is ComboBox comboBox)
        {
            if (comboBox.SelectedItem is ComboBoxItem item && item.Content is string mode)
            {
                vm.SwitchTradingMode(mode);
            }
        }
    }

    private void ClientIdTextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (DataContext is MonitorViewModel vm && sender is TextBox textBox)
        {
            if (int.TryParse(textBox.Text, out int clientId) && clientId >= 0)
            {
                vm.UpdateClientId(clientId);
            }
        }
    }
}
