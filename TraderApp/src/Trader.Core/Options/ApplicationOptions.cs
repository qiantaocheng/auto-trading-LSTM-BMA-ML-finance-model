namespace Trader.Core.Options;

public sealed class PythonOptions
{
    public string Executable { get; set; } = string.Empty;
    public string PredictScript { get; set; } = string.Empty;
    public string PortfolioScript { get; set; } = string.Empty;
    public string TradingScript { get; set; } = string.Empty;
    public string HmmScript { get; set; } = string.Empty;
    public string EarningsScript { get; set; } = string.Empty;
    public string EtfRotationScript { get; set; } = string.Empty;
    public string ParquetPath { get; set; } = string.Empty;
    public string DefaultSnapshot { get; set; } = string.Empty;
    public string RetrainScript { get; set; } = string.Empty;
}

public sealed class PolygonOptions
{
    public string ApiKey { get; set; } = string.Empty;
}

public sealed class DatabaseOptions
{
    public string FileName { get; set; } = "TraderApp.db";
}

public sealed class EtfRotationOptions
{
    public int RebalanceDays { get; set; } = 21;
    public double Deadband { get; set; } = 0.05;
    public double DeadbandUp { get; set; } = 0.02;
    public double DeadbandDown { get; set; } = 0.05;
    public double MaxStep { get; set; } = 0.15;
    public int MinHoldDays { get; set; } = 5;
    public double CapitalRatio { get; set; } = 0.75;
    public bool UseHmm { get; set; } = false;
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

public sealed class AutoConnectOptions
{
    public bool Enabled { get; set; }
    public string IbLoginId { get; set; } = string.Empty;
    public string IbPassword { get; set; } = string.Empty;
    public string TradingMode { get; set; } = "paper";
    public string IbcConfigPath { get; set; } = @"D:\trade\TraderApp\ibc\IBCWin-3.23.0\config.ini";
    public string StartGatewayBat { get; set; } = @"D:\trade\TraderApp\ibc\IBCWin-3.23.0\StartGateway.bat";
    public int PreMarketMinutes { get; set; } = 60;
}

public sealed class TelegramOptions
{
    public string BotToken { get; set; } = string.Empty;
    public string ChatId { get; set; } = string.Empty;
    public bool Enabled => !string.IsNullOrEmpty(BotToken) && !string.IsNullOrEmpty(ChatId);
}
