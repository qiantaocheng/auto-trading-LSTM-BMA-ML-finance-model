namespace Trader.Core.Options;

public sealed class PythonOptions
{
    public string Executable { get; set; } = string.Empty;
    public string PredictScript { get; set; } = string.Empty;
    public string PortfolioScript { get; set; } = string.Empty;
    public string TradingScript { get; set; } = string.Empty;
    public string HmmScript { get; set; } = string.Empty;
    public string EarningsScript { get; set; } = string.Empty;
    public string ParquetPath { get; set; } = string.Empty;
    public string DefaultSnapshot { get; set; } = string.Empty;
}

public sealed class PolygonOptions
{
    public string ApiKey { get; set; } = string.Empty;
}

public sealed class DatabaseOptions
{
    public string FileName { get; set; } = "TraderApp.db";
}

public sealed class IBKROptions
{
    public string Mode { get; set; } = "Paper";
    public string Host { get; set; } = "127.0.0.1";
    public int PaperPort { get; set; } = 4002;  // Paper Trading port
    public int LivePort { get; set; } = 4001;   // Live Trading port
    public int ClientId { get; set; } = 123;

    public int GetPort() => Mode.Equals("Live", StringComparison.OrdinalIgnoreCase) ? LivePort : PaperPort;
}
