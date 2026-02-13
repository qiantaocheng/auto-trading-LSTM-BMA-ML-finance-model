using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Options;
using Trader.Core.Options;

namespace Trader.Core.Services;

public interface IWritableOptionsService
{
    void UpdateTradingMode(string mode);
    void UpdateClientId(int clientId);
    void UpdateAutoConnectEnabled(bool enabled);
    void UpdateAutoConnectCredentials(string loginId, string password, string tradingMode);
    void UpdateAutoConnectPreMarket(int minutes);
    void UpdateDefaultSnapshot(string snapshotId);
}

public sealed class WritableOptionsService : IWritableOptionsService
{
    private readonly IOptionsMonitor<IBKROptions> _ibkrOptions;
    private readonly IOptionsMonitor<AutoConnectOptions> _autoConnectOptions;
    private readonly IOptionsMonitor<PythonOptions> _pythonOptions;
    private readonly string _appsettingsPath;

    public WritableOptionsService(
        IOptionsMonitor<IBKROptions> ibkrOptions,
        IOptionsMonitor<AutoConnectOptions> autoConnectOptions,
        IOptionsMonitor<PythonOptions> pythonOptions,
        string appsettingsPath)
    {
        _ibkrOptions = ibkrOptions;
        _autoConnectOptions = autoConnectOptions;
        _pythonOptions = pythonOptions;
        _appsettingsPath = appsettingsPath;
    }

    public void UpdateTradingMode(string mode)
    {
        if (mode != "Paper" && mode != "Live")
            throw new ArgumentException($"Invalid mode: {mode}. Must be 'Paper' or 'Live'.", nameof(mode));

        var currentOptions = _ibkrOptions.CurrentValue;
        typeof(IBKROptions).GetProperty(nameof(IBKROptions.Mode))!.SetValue(currentOptions, mode);
        UpdateJsonSection("IBKR", "Mode", mode);
    }

    public void UpdateClientId(int clientId)
    {
        if (clientId < 0)
            throw new ArgumentException("Client ID must be non-negative.", nameof(clientId));

        var currentOptions = _ibkrOptions.CurrentValue;
        typeof(IBKROptions).GetProperty(nameof(IBKROptions.ClientId))!.SetValue(currentOptions, clientId);
        UpdateJsonSection("IBKR", "ClientId", clientId);
    }

    public void UpdateAutoConnectEnabled(bool enabled)
    {
        var opts = _autoConnectOptions.CurrentValue;
        typeof(AutoConnectOptions).GetProperty(nameof(AutoConnectOptions.Enabled))!.SetValue(opts, enabled);
        UpdateJsonSection("AutoConnect", "Enabled", enabled);
    }

    public void UpdateAutoConnectCredentials(string loginId, string password, string tradingMode)
    {
        var opts = _autoConnectOptions.CurrentValue;
        typeof(AutoConnectOptions).GetProperty(nameof(AutoConnectOptions.IbLoginId))!.SetValue(opts, loginId);
        typeof(AutoConnectOptions).GetProperty(nameof(AutoConnectOptions.IbPassword))!.SetValue(opts, password);
        typeof(AutoConnectOptions).GetProperty(nameof(AutoConnectOptions.TradingMode))!.SetValue(opts, tradingMode);

        UpdateJsonSection("AutoConnect", new Dictionary<string, object>
        {
            ["IbLoginId"] = loginId,
            ["IbPassword"] = password,
            ["TradingMode"] = tradingMode,
        });

        // Sync to IBC config.ini
        SyncIbcConfig(loginId, password, tradingMode);
    }

    public void UpdateAutoConnectPreMarket(int minutes)
    {
        var opts = _autoConnectOptions.CurrentValue;
        typeof(AutoConnectOptions).GetProperty(nameof(AutoConnectOptions.PreMarketMinutes))!.SetValue(opts, minutes);
        UpdateJsonSection("AutoConnect", "PreMarketMinutes", minutes);
    }

    public void UpdateDefaultSnapshot(string snapshotId)
    {
        if (string.IsNullOrWhiteSpace(snapshotId))
            throw new ArgumentException("Snapshot ID cannot be empty.", nameof(snapshotId));

        var opts = _pythonOptions.CurrentValue;
        typeof(PythonOptions).GetProperty(nameof(PythonOptions.DefaultSnapshot))!.SetValue(opts, snapshotId);
        UpdateJsonSection("Python", "DefaultSnapshot", snapshotId);
    }

    // ── IBC config.ini sync ─────────────────────────────────────────────

    private void SyncIbcConfig(string loginId, string password, string tradingMode)
    {
        try
        {
            var ibcConfigPath = _autoConnectOptions.CurrentValue.IbcConfigPath;
            if (string.IsNullOrEmpty(ibcConfigPath) || !File.Exists(ibcConfigPath))
                return;

            var lines = File.ReadAllLines(ibcConfigPath);
            for (int i = 0; i < lines.Length; i++)
            {
                if (Regex.IsMatch(lines[i], @"^IbLoginId="))
                    lines[i] = $"IbLoginId={loginId}";
                else if (Regex.IsMatch(lines[i], @"^IbPassword="))
                    lines[i] = $"IbPassword={password}";
                else if (Regex.IsMatch(lines[i], @"^TradingMode="))
                    lines[i] = $"TradingMode={tradingMode}";
            }
            File.WriteAllLines(ibcConfigPath, lines);
        }
        catch
        {
            // Non-critical — IBC config sync failure doesn't block app
        }
    }

    // ── JSON file helpers ─────────────────────────────────────────────

    private void UpdateJsonSection(string sectionName, string propertyName, object value)
    {
        UpdateJsonSection(sectionName, new Dictionary<string, object> { [propertyName] = value });
    }

    private void UpdateJsonSection(string sectionName, Dictionary<string, object> updates)
    {
        try
        {
            if (!File.Exists(_appsettingsPath)) return;

            var json = File.ReadAllText(_appsettingsPath);
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
            {
                writer.WriteStartObject();
                foreach (var property in root.EnumerateObject())
                {
                    if (property.Name == sectionName)
                    {
                        writer.WritePropertyName(sectionName);
                        writer.WriteStartObject();
                        foreach (var prop in property.Value.EnumerateObject())
                        {
                            if (updates.TryGetValue(prop.Name, out var newVal))
                            {
                                writer.WritePropertyName(prop.Name);
                                WriteJsonValue(writer, newVal);
                            }
                            else
                            {
                                prop.WriteTo(writer);
                            }
                        }
                        writer.WriteEndObject();
                    }
                    else
                    {
                        property.WriteTo(writer);
                    }
                }
                writer.WriteEndObject();
            }

            File.WriteAllText(_appsettingsPath, System.Text.Encoding.UTF8.GetString(stream.ToArray()));
        }
        catch
        {
            // Ignore file write errors - in-memory update is still valid
        }
    }

    private static void WriteJsonValue(Utf8JsonWriter writer, object value)
    {
        switch (value)
        {
            case string s: writer.WriteStringValue(s); break;
            case int i: writer.WriteNumberValue(i); break;
            case bool b: writer.WriteBooleanValue(b); break;
            case double d: writer.WriteNumberValue(d); break;
            default: throw new NotSupportedException($"Property type {value.GetType()} not supported");
        }
    }
}
