using Trader.App.ViewModels.Pages;

namespace Trader.App.ViewModels;

public class ShellViewModel
{
    public ShellViewModel(
        DirectPredictionViewModel directPrediction,
        MonitorViewModel monitor,
        DatabaseViewModel database,
        ConnectionViewModel connection)
    {
        DirectPrediction = directPrediction;
        Monitor = monitor;
        Database = database;
        Connection = connection;
    }

    public DirectPredictionViewModel DirectPrediction { get; }
    public MonitorViewModel Monitor { get; }
    public DatabaseViewModel Database { get; }
    public ConnectionViewModel Connection { get; }
}
