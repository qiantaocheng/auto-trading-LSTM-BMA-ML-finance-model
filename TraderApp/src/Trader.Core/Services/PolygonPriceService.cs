using System.Collections.Concurrent;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.Core.Services;

/// <summary>
/// Fetches delayed stock prices from Polygon API (no IBKR dependency).
/// Uses batch snapshot endpoint with individual prev-close fallback.
/// Includes TTL cache (5s) and rate-limit protection (max 4 calls/min).
/// </summary>
public sealed class PolygonPriceService
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    // ── Price cache (5s TTL) ───────────────────────────────────────────
    private readonly ConcurrentDictionary<string, CachedPrice> _cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly TimeSpan CacheTtl = TimeSpan.FromSeconds(5);

    // ── Rate limiting (max 4 calls/min = 15s gap) ─────────────────────
    private DateTime _lastHttpCallUtc = DateTime.MinValue;
    private static readonly TimeSpan MinCallGap = TimeSpan.FromSeconds(15);
    private readonly SemaphoreSlim _httpLock = new(1, 1);

    // ── Stale tracking ────────────────────────────────────────────────
    private int _consecutiveFailures;
    public bool IsStale => _consecutiveFailures >= 3;
    public int ConsecutiveFailures => _consecutiveFailures;

    public PolygonPriceService(HttpClient httpClient, string apiKey)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _httpClient.BaseAddress ??= new Uri("https://api.polygon.io/");
        _httpClient.Timeout = TimeSpan.FromSeconds(10);
    }

    /// <summary>
    /// Get delayed prices for multiple symbols in one call.
    /// Returns cached results when fresh; only hits Polygon for stale/missing symbols.
    /// </summary>
    public async Task<Dictionary<string, decimal>> GetPricesAsync(IEnumerable<string> symbols, CancellationToken ct = default)
    {
        var symbolList = symbols.ToList();
        if (symbolList.Count == 0 || string.IsNullOrEmpty(_apiKey))
            return new Dictionary<string, decimal>();

        var result = new Dictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);
        var staleSymbols = new List<string>();
        var now = DateTime.UtcNow;

        // Step 1: Return cached prices for fresh symbols
        foreach (var symbol in symbolList)
        {
            if (_cache.TryGetValue(symbol, out var cached) && (now - cached.FetchedAt) < CacheTtl)
            {
                result[symbol] = cached.Price;
            }
            else
            {
                staleSymbols.Add(symbol);
            }
        }

        // Step 2: If all cached, return immediately
        if (staleSymbols.Count == 0)
            return result;

        // Step 3: Rate-limit check — if too soon since last call, return stale cache
        await _httpLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var elapsed = now - _lastHttpCallUtc;
            if (elapsed < MinCallGap)
            {
                // Too soon — return stale cache for remaining symbols
                foreach (var symbol in staleSymbols)
                {
                    if (_cache.TryGetValue(symbol, out var stale))
                        result[symbol] = stale.Price;
                }
                return result;
            }

            // Step 4: Fetch from Polygon
            var freshPrices = await FetchFromPolygonAsync(staleSymbols, ct).ConfigureAwait(false);
            _lastHttpCallUtc = DateTime.UtcNow;

            if (freshPrices.Count > 0)
            {
                Interlocked.Exchange(ref _consecutiveFailures, 0);
                foreach (var (symbol, price) in freshPrices)
                {
                    result[symbol] = price;
                    _cache[symbol] = new CachedPrice(price, DateTime.UtcNow);
                }
            }
            else
            {
                Interlocked.Increment(ref _consecutiveFailures);
            }

            // Fill remaining from stale cache
            foreach (var symbol in staleSymbols)
            {
                if (!result.ContainsKey(symbol) && _cache.TryGetValue(symbol, out var stale))
                    result[symbol] = stale.Price;
            }
        }
        finally
        {
            _httpLock.Release();
        }

        return result;
    }

    /// <summary>
    /// Get delayed price for a single symbol.
    /// </summary>
    public async Task<decimal?> GetPriceAsync(string symbol, CancellationToken ct = default)
    {
        var prices = await GetPricesAsync(new[] { symbol }, ct).ConfigureAwait(false);
        return prices.TryGetValue(symbol, out var price) ? price : null;
    }

    private async Task<Dictionary<string, decimal>> FetchFromPolygonAsync(List<string> symbols, CancellationToken ct)
    {
        var prices = new Dictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);

        // Try batch snapshot first
        try
        {
            var tickers = string.Join(",", symbols);
            var uri = $"v2/snapshot/locale/us/markets/stocks/tickers?tickers={tickers}&apiKey={_apiKey}";
            using var response = await _httpClient.GetAsync(uri, ct).ConfigureAwait(false);

            if ((int)response.StatusCode == 429)
            {
                // Rate limited — increment failures and return empty
                Interlocked.Increment(ref _consecutiveFailures);
                return prices;
            }

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
        foreach (var symbol in symbols)
        {
            if (prices.ContainsKey(symbol)) continue;

            try
            {
                var uri = $"v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={_apiKey}";
                using var response = await _httpClient.GetAsync(uri, ct).ConfigureAwait(false);

                if ((int)response.StatusCode == 429)
                    break; // Stop all individual calls if rate limited

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

    private static decimal ExtractPrice(SnapshotTicker t)
    {
        if (t.LastTrade?.P is > 0)
            return (decimal)t.LastTrade.P;
        if (t.Day?.C is > 0)
            return (decimal)t.Day.C;
        if (t.PrevDay?.C is > 0)
            return (decimal)t.PrevDay.C;
        return 0;
    }

    // ── SPY MA200 exposure cap ────────────────────────────────────────
    // Thresholds: SPY >= MA200 → 1.0, SPY < MA200 → 0.60, SPY < 0.95*MA200 → 0.30
    private double _cachedSpyMa200Cap = 1.0;
    private double _cachedSpyPrice;
    private double _cachedSpyMa200;
    private DateTime _spyMa200CacheDate;

    /// <summary>
    /// Compute SPY MA200 exposure cap using Polygon daily aggregates.
    /// Cached once per calendar day. Returns (cap, spyPrice, spyMa200).
    /// </summary>
    public async Task<(double Cap, double SpyPrice, double SpyMa200)> GetSpyMa200CapAsync(CancellationToken ct = default)
    {
        var today = DateTime.UtcNow.Date;
        if (_spyMa200CacheDate == today)
            return (_cachedSpyMa200Cap, _cachedSpyPrice, _cachedSpyMa200);

        try
        {
            // Fetch 250 daily bars for SPY (need 200 for MA + some buffer)
            var end = today.ToString("yyyy-MM-dd");
            var start = today.AddDays(-370).ToString("yyyy-MM-dd");
            var uri = $"v2/aggs/ticker/SPY/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=300&apiKey={_apiKey}";

            using var response = await _httpClient.GetAsync(uri, ct).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
                return (_cachedSpyMa200Cap, _cachedSpyPrice, _cachedSpyMa200);

            var payload = await response.Content.ReadFromJsonAsync<AggResponse>(JsonOptions, ct).ConfigureAwait(false);
            if (payload?.Results is null || payload.Results.Count < 200)
                return (_cachedSpyMa200Cap, _cachedSpyPrice, _cachedSpyMa200);

            // Compute MA200 from last 200 closes
            var closes = payload.Results.Select(r => r.C).ToList();
            var spyPrice = closes[^1];
            var ma200 = closes.Skip(closes.Count - 200).Take(200).Average();

            double cap;
            if (spyPrice < ma200 * 0.95)
                cap = 0.30;
            else if (spyPrice < ma200)
                cap = 0.60;
            else
                cap = 1.0;

            _cachedSpyMa200Cap = cap;
            _cachedSpyPrice = spyPrice;
            _cachedSpyMa200 = ma200;
            _spyMa200CacheDate = today;

            return (cap, spyPrice, ma200);
        }
        catch
        {
            return (_cachedSpyMa200Cap, _cachedSpyPrice, _cachedSpyMa200);
        }
    }

    private sealed record AggResponse(
        [property: JsonPropertyName("results")] List<AggBar>? Results);

    private sealed record AggBar(
        [property: JsonPropertyName("c")] double C);

    private sealed record CachedPrice(decimal Price, DateTime FetchedAt);

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
