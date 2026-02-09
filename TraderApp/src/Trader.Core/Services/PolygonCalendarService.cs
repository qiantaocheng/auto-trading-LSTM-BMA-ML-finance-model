using System.Net.Http.Json;
using System.Text.Json;

namespace Trader.Core.Services;

public sealed class PolygonCalendarService
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;

    public PolygonCalendarService(HttpClient httpClient, string apiKey)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _httpClient.BaseAddress ??= new Uri("https://api.polygon.io/");
    }

    public async Task<MarketStatusPayload?> GetCurrentStatusAsync(CancellationToken cancellationToken = default)
    {
        var uri = $"v1/marketstatus/now?apiKey={_apiKey}";
        using var response = await _httpClient.GetAsync(uri, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        return await JsonSerializer.DeserializeAsync<MarketStatusPayload>(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<MarketHoliday>> GetUpcomingClosuresAsync(CancellationToken cancellationToken = default)
    {
        var uri = $"v1/marketstatus/upcoming?apiKey={_apiKey}";
        using var response = await _httpClient.GetAsync(uri, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        var holidays = await JsonSerializer.DeserializeAsync<List<MarketHoliday>>(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        return holidays ?? new List<MarketHoliday>();
    }
}

public sealed record MarketStatusPayload(
    string? Market,
    string? ServerTime,
    string? Exchange,
    bool? IsOpen,
    MarketStatus? Forex,
    MarketStatus? Equity,
    MarketStatus? Crypto);

public sealed record MarketStatus(string? Market, string? State, string? Close); 

public sealed record MarketHoliday(
    string Name,
    string Date,
    string Status,
    string? Open,
    string? Close);
