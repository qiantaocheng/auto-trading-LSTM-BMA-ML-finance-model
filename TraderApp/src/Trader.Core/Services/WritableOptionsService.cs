using System.Text.Json;
using Microsoft.Extensions.Options;
using Trader.Core.Options;

namespace Trader.Core.Services;

public interface IWritableOptionsService
{
    void UpdateTradingMode(string mode);
    void UpdateClientId(int clientId);
}

public sealed class WritableOptionsService : IWritableOptionsService
{
    private readonly IOptionsMonitor<IBKROptions> _ibkrOptions;
    private readonly string _appsettingsPath;

    public WritableOptionsService(IOptionsMonitor<IBKROptions> ibkrOptions, string appsettingsPath)
    {
        _ibkrOptions = ibkrOptions;
        _appsettingsPath = appsettingsPath;
    }

    public void UpdateTradingMode(string mode)
    {
        if (mode != "Paper" && mode != "Live")
        {
            throw new ArgumentException($"Invalid mode: {mode}. Must be 'Paper' or 'Live'.", nameof(mode));
        }

        // Update in-memory options
        var currentOptions = _ibkrOptions.CurrentValue;
        typeof(IBKROptions).GetProperty(nameof(IBKROptions.Mode))!.SetValue(currentOptions, mode);

        // Update appsettings.json file
        UpdateJsonFile("Mode", mode);
    }

    public void UpdateClientId(int clientId)
    {
        if (clientId < 0)
        {
            throw new ArgumentException("Client ID must be non-negative.", nameof(clientId));
        }

        // Update in-memory options
        var currentOptions = _ibkrOptions.CurrentValue;
        typeof(IBKROptions).GetProperty(nameof(IBKROptions.ClientId))!.SetValue(currentOptions, clientId);

        // Update appsettings.json file
        UpdateJsonFile("ClientId", clientId);
    }

    private void UpdateJsonFile(string propertyName, object value)
    {
        try
        {
            if (!File.Exists(_appsettingsPath))
            {
                return;
            }

            var json = File.ReadAllText(_appsettingsPath);
            var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Rebuild JSON with updated IBKR property
            var updatedJson = RebuildJsonWithProperty(root, propertyName, value);
            File.WriteAllText(_appsettingsPath, updatedJson);
        }
        catch
        {
            // Ignore file write errors - in-memory update is still valid
        }
    }

    private string RebuildJsonWithProperty(JsonElement root, string propertyName, object value)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();

            foreach (var property in root.EnumerateObject())
            {
                if (property.Name == "IBKR")
                {
                    writer.WritePropertyName("IBKR");
                    writer.WriteStartObject();

                    foreach (var ibkrProp in property.Value.EnumerateObject())
                    {
                        if (ibkrProp.Name == propertyName)
                        {
                            // Write the updated property with correct type
                            writer.WritePropertyName(propertyName);
                            if (value is string strValue)
                            {
                                writer.WriteStringValue(strValue);
                            }
                            else if (value is int intValue)
                            {
                                writer.WriteNumberValue(intValue);
                            }
                            else
                            {
                                throw new NotSupportedException($"Property type {value.GetType()} not supported");
                            }
                        }
                        else
                        {
                            ibkrProp.WriteTo(writer);
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

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }
}
