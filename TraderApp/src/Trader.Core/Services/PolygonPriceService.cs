using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.Core.Services;

/// <summary>
/// Fetches delayed stock prices from Polygon API (no IBKR dependency).
/// Uses batch snapshot endpoint with individual prev-close fallback.
/// </summary>
public sealed class PolygonPriceService
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public PolygonPriceService(HttpClient httpClient, string apiKey)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _httpClient.BaseAddress ??= new Uri("https://api.polygon.io/");
        _httpClient.Timeout = TimeSpan.FromSeconds(10);
    }

    /// <summary>
    /// Get delayed prices for multiple symbols in one call.
    /// Returns a dictionary of symbol → price.
    /// </summary>
    public async Task<Dictionary<string, decimal>> GetPricesAsync(IEnumerable<string> symbols, CancellationToken ct = default)
    {
        var symbolList = symbols.ToList();
        if (symbolList.Count == 0 || string.IsNullOrEmpty(_apiKey))
            return new Dictionary<string, decimal>();

        var prices = new Dictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);

        // Try batch snapshot first (most current delayed data)
        try
        {
            var tickers = string.Join(",", symbolList);
            var uri = $"v2/snapshot/locale/us/markets/stocks/tickers?tickers={tickers}&apiKey={_apiKey}";
            using var response = await _httpClient.GetAsync(uri, ct).ConfigureAwait(false);

            if (response.IsSuccessStatusCode)
            {
                var payload = await response.Content.ReadFromJsonAsync<SnapshotResponse>(JsonOptions, ct).ConfigureAwait(false);
                if (payload?.Tickers is not null)
                {
                    foreach (var t in payload.Tickers)
                    {
                        var price = ExtractPrice(t);
                        if (price > 0)
                            prices[t.Ticker ?? ""] = price;
                    }
                }
            }
        }
        catch
        {
            // Fall through to individual prev close
        }

        // Fallback: individual prev close for any missing symbols
        foreach (var symbol in symbolList)
        {
            if (prices.ContainsKey(symbol)) continue;

            try
            {
                var uri = $"v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={_apiKey}";
                using var response = await _httpClient.GetAsync(uri, ct).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    var payload = await response.Content.ReadFromJsonAsync<PrevCloseResponse>(JsonOptions, ct).ConfigureAwait(false);
                    if (payload?.Results is { Count: > 0 })
                    {
                        var close = payload.Results[0].C;
                        if (close > 0)
                            prices[symbol] = (decimal)close;
                    }
                }
            }
            catch
            {
                // Skip this symbol
            }
        }

        return prices;
    }

    /// <summary>
    /// Get delayed price for a single symbol.
    /// </summary>
    public async Task<decimal?> GetPriceAsync(string symbol, CancellationToken ct = default)
    {
        var prices = await GetPricesAsync(new[] { symbol }, ct).ConfigureAwait(false);
        return prices.TryGetValue(symbol, out var price) ? price : null;
    }

    private static decimal ExtractPrice(SnapshotTicker t)
    {
        // Priority: lastTrade → day close → prevDay close
        if (t.LastTrade?.P is > 0)
            return (decimal)t.LastTrade.P;
        if (t.Day?.C is > 0)
            return (decimal)t.Day.C;
        if (t.PrevDay?.C is > 0)
            return (decimal)t.PrevDay.C;
        return 0;
    }

    // --- JSON models ---

    private sealed record SnapshotResponse(
        [property: JsonPropertyName("status")] string? Status,
        [property: JsonPropertyName("tickers")] List<SnapshotTicker>? Tickers);

    private sealed record SnapshotTicker(
        [property: JsonPropertyName("ticker")] string? Ticker,
        [property: JsonPropertyName("lastTrade")] SnapshotTrade? LastTrade,
        [property: JsonPropertyName("day")] SnapshotAgg? Day,
        [property: JsonPropertyName("prevDay")] SnapshotAgg? PrevDay);

    private sealed record SnapshotTrade(
        [property: JsonPropertyName("p")] double P);

    private sealed record SnapshotAgg(
        [property: JsonPropertyName("c")] double C);

    private sealed record PrevCloseResponse(
        [property: JsonPropertyName("resultsCount")] int ResultsCount,
        [property: JsonPropertyName("results")] List<PrevCloseResult>? Results);

    private sealed record PrevCloseResult(
        [property: JsonPropertyName("c")] double C);
}
