using System.Data;
using System.Data.SQLite;

namespace Trader.Core.Repositories;

public sealed record TickerRecord(string Symbol, int Tag, string Source, DateTime AddedAt);

public sealed class TraderDatabase : IDisposable
{
    private string _dbPath;
    private SQLiteConnection? _activeConnection;
    private readonly object _connLock = new();

    public TraderDatabase(string dbPath)
    {
        _dbPath = dbPath;
    }

    public string CurrentDbPath => _dbPath;

    private SQLiteConnection Connection
    {
        get
        {
            lock (_connLock)
            {
                if (_activeConnection is null)
                {
                    _activeConnection = OpenConnection(_dbPath);
                }
                return _activeConnection;
            }
        }
    }

    private static SQLiteConnection OpenConnection(string dbPath)
    {
        var connection = new SQLiteConnection($"Data Source={dbPath};Version=3;");
        connection.Open();
        using (var pragma = connection.CreateCommand())
        {
            pragma.CommandText = "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;";
            pragma.ExecuteNonQuery();
        }
        return connection;
    }

    /// <summary>
    /// Switch to a different database file (e.g., when toggling Paper/Live mode).
    /// Closes current connection and opens the new one. Caller must ensure no concurrent DB access.
    /// </summary>
    public void SwitchDatabase(string newDbPath)
    {
        lock (_connLock)
        {
            if (_activeConnection is not null)
            {
                try { _activeConnection.Close(); } catch { }
                _activeConnection.Dispose();
                _activeConnection = null;
            }
            _dbPath = newDbPath;
            // Force open + schema creation on new DB
            _activeConnection = OpenConnection(newDbPath);
        }
        EnsureSchema();
    }

    public void EnsureSchema()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"
CREATE TABLE IF NOT EXISTS tickers (
    symbol TEXT PRIMARY KEY,
    tag INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    strategy TEXT NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    entered_at TEXT NOT NULL,
    scheduled_exit TEXT,
    target_weight REAL,
    last_rebalanced TEXT,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_fill_price REAL,
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS direct_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    ticker TEXT NOT NULL,
    score REAL NOT NULL,
    ema4 REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS capital_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    net_liq REAL NOT NULL,
    cash REAL NOT NULL,
    stock_value REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_capital_history_timestamp ON capital_history(timestamp DESC);

CREATE TABLE IF NOT EXISTS etf_rotation_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS broker_positions (
    symbol TEXT PRIMARY KEY,
    quantity INTEGER NOT NULL,
    avg_cost REAL,
    market_value REAL,
    last_synced TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_intents (
    intent_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    order_id INTEGER,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    executed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_trade_intents_lookup ON trade_intents(strategy, symbol, side);

CREATE TABLE IF NOT EXISTS pending_buys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
";
        cmd.ExecuteNonQuery();

        // Migrate old tables into positions if they exist
        MigrateOldTables();

        // Auto-seed positions from broker_positions if positions table is empty
        SeedPositionsFromBrokerIfEmpty();
    }

    private void MigrateOldTables()
    {
        try
        {
            // Check if old auto_positions table exists
            using var check = Connection.CreateCommand();
            check.CommandText = "SELECT name FROM sqlite_master WHERE type='table' AND name='auto_positions'";
            if (check.ExecuteScalar() is string)
            {
                using var migrate = Connection.CreateCommand();
                migrate.CommandText = @"
INSERT OR IGNORE INTO positions (symbol, strategy, shares, entry_price, entered_at, scheduled_exit, note)
SELECT symbol, 'AutoBuy', shares, COALESCE(entry_price, 0), entered_at, scheduled_exit, note FROM auto_positions;
DROP TABLE IF EXISTS auto_positions;";
                migrate.ExecuteNonQuery();
            }
        }
        catch { /* ignore migration errors */ }

        try
        {
            using var check = Connection.CreateCommand();
            check.CommandText = "SELECT name FROM sqlite_master WHERE type='table' AND name='etf_positions'";
            if (check.ExecuteScalar() is string)
            {
                using var migrate = Connection.CreateCommand();
                migrate.CommandText = @"
INSERT OR IGNORE INTO positions (symbol, strategy, shares, entry_price, entered_at, scheduled_exit, target_weight, last_rebalanced)
SELECT symbol, 'ETF', shares, entry_price, entered_at, NULL, target_weight, last_rebalanced FROM etf_positions;
DROP TABLE IF EXISTS etf_positions;";
                migrate.ExecuteNonQuery();
            }
        }
        catch { /* ignore migration errors */ }
    }

    /// <summary>
    /// One-time bootstrap: if positions table is empty but broker_positions has data,
    /// seed positions from broker_positions. ETF Portfolio B tickers get strategy='ETF',
    /// others get strategy='Manual'. Safe to call repeatedly — only seeds when positions empty.
    /// </summary>
    public void SeedPositionsFromBrokerIfEmpty()
    {
        try
        {
            using var countCmd = Connection.CreateCommand();
            countCmd.CommandText = "SELECT COUNT(*) FROM positions";
            var posCount = Convert.ToInt32(countCmd.ExecuteScalar());
            if (posCount > 0) return; // Already have positions, skip

            using var brokerCmd = Connection.CreateCommand();
            brokerCmd.CommandText = "SELECT COUNT(*) FROM broker_positions WHERE quantity > 0";
            var brokerCount = Convert.ToInt32(brokerCmd.ExecuteScalar());
            if (brokerCount == 0) return; // No broker data to seed from

            // Portfolio B ETF tickers and their target weights
            var etfWeights = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase)
            {
                ["QQQ"] = 0.25, ["USMV"] = 0.25, ["QUAL"] = 0.20,
                ["PDBC"] = 0.15, ["COPX"] = 0.05, ["URA"] = 0.05, ["DBA"] = 0.05
            };

            using var readCmd = Connection.CreateCommand();
            readCmd.CommandText = "SELECT symbol, quantity, avg_cost FROM broker_positions WHERE quantity > 0";
            using var reader = readCmd.ExecuteReader();
            var now = DateTime.UtcNow.ToString("o");

            while (reader.Read())
            {
                var symbol = reader.GetString(0).Trim().ToUpperInvariant();
                var qty = reader.GetInt32(1);
                var avgCost = reader.GetDouble(2);

                var isEtf = etfWeights.TryGetValue(symbol, out var weight);
                using var ins = Connection.CreateCommand();
                ins.CommandText = @"INSERT OR IGNORE INTO positions (symbol, strategy, shares, entry_price, entered_at, scheduled_exit, target_weight, last_rebalanced)
                    VALUES (@s, @strat, @shares, @price, @entered, @exit, @weight, @rebal)";
                ins.Parameters.Add(new SQLiteParameter("@s", DbType.String) { Value = symbol });
                ins.Parameters.Add(new SQLiteParameter("@strat", DbType.String) { Value = isEtf ? "ETF" : "Manual" });
                ins.Parameters.Add(new SQLiteParameter("@shares", DbType.Int32) { Value = qty });
                ins.Parameters.Add(new SQLiteParameter("@price", DbType.Double) { Value = avgCost });
                ins.Parameters.Add(new SQLiteParameter("@entered", DbType.String) { Value = now });
                ins.Parameters.Add(new SQLiteParameter("@exit", DbType.String) { Value = isEtf ? (object)DBNull.Value : "2099-12-31T00:00:00Z" });
                ins.Parameters.Add(new SQLiteParameter("@weight", DbType.Double) { Value = isEtf ? weight : (object)DBNull.Value });
                ins.Parameters.Add(new SQLiteParameter("@rebal", DbType.String) { Value = isEtf ? now : (object)DBNull.Value });
                ins.ExecuteNonQuery();
            }
        }
        catch { /* ignore seed errors */ }
    }

    // --- Tickers CRUD ---

    public IReadOnlyList<TickerRecord> GetTickerRecords()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT symbol, tag, COALESCE(source,''), COALESCE(added_at, CURRENT_TIMESTAMP) FROM tickers ORDER BY added_at DESC";
        using var reader = cmd.ExecuteReader();
        var list = new List<TickerRecord>();
        while (reader.Read())
        {
            var symbol = reader.GetString(0);
            var tag = reader.GetInt32(1);
            var source = reader.GetString(2);
            var addedAtRaw = reader.GetString(3);
            DateTime.TryParse(addedAtRaw, out var addedAt);
            list.Add(new TickerRecord(symbol, tag, source, addedAt == default ? DateTime.UtcNow : addedAt));
        }
        return list;
    }

    public void ReplaceAutoTickers(IEnumerable<string> symbols, string source)
    {
        using var tx = Connection.BeginTransaction();
        using (var delete = Connection.CreateCommand())
        {
            delete.CommandText = "DELETE FROM tickers WHERE tag = 0";
            delete.ExecuteNonQuery();
        }
        foreach (var symbol in symbols)
        {
            if (string.IsNullOrWhiteSpace(symbol)) continue;
            using var insert = Connection.CreateCommand();
            insert.CommandText = "INSERT OR REPLACE INTO tickers(symbol, tag, source, added_at) VALUES(@symbol, 0, @source, @ts)";
            insert.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
            insert.Parameters.Add(new SQLiteParameter("@source", DbType.String) { Value = source });
            insert.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
            insert.ExecuteNonQuery();
        }
        tx.Commit();
    }

    public void AddManualTicker(string symbol, string source)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT OR REPLACE INTO tickers(symbol, tag, source, added_at) VALUES(@symbol, 1, @source, @ts)";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@source", DbType.String) { Value = source });
        cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
        cmd.ExecuteNonQuery();
    }

    public void InsertDirectPredictions(IEnumerable<(string Ticker, double Score, double Ema4)> predictions, string snapshotId, string asOf)
    {
        var ts = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        using var tx = Connection.BeginTransaction();
        foreach (var (ticker, score, ema4) in predictions)
        {
            if (string.IsNullOrWhiteSpace(ticker)) continue;
            using var cmd = Connection.CreateCommand();
            cmd.CommandText = "INSERT INTO direct_predictions (ts, snapshot_id, as_of, ticker, score, ema4, created_at) VALUES (@ts, @sid, @asOf, @ticker, @score, @ema4, @cat)";
            cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.Int64) { Value = ts });
            cmd.Parameters.Add(new SQLiteParameter("@sid", DbType.String) { Value = snapshotId });
            cmd.Parameters.Add(new SQLiteParameter("@asOf", DbType.String) { Value = asOf });
            cmd.Parameters.Add(new SQLiteParameter("@ticker", DbType.String) { Value = ticker.Trim().ToUpperInvariant() });
            cmd.Parameters.Add(new SQLiteParameter("@score", DbType.Double) { Value = score });
            cmd.Parameters.Add(new SQLiteParameter("@ema4", DbType.Double) { Value = ema4 });
            cmd.Parameters.Add(new SQLiteParameter("@cat", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
            cmd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    public void DeleteTicker(string symbol)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM tickers WHERE symbol = @symbol";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.ExecuteNonQuery();
    }

    // ── Unified Positions CRUD ──────────────────────────────────────────

    public void InsertPosition(string symbol, string strategy, int shares, decimal entryPrice, int holdDays = 5, double? targetWeight = null, string? note = null)
    {
        var now = DateTime.UtcNow;
        var scheduledExit = strategy == "ETF" ? (DateTime?)null : AddTradingDays(now, holdDays);
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"INSERT OR REPLACE INTO positions (symbol, strategy, shares, entry_price, entered_at, scheduled_exit, target_weight, note)
            VALUES (@symbol, @strategy, @shares, @price, @entered, @exit, @weight, @note)";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@strategy", DbType.String) { Value = strategy });
        cmd.Parameters.Add(new SQLiteParameter("@shares", DbType.Int32) { Value = shares });
        cmd.Parameters.Add(new SQLiteParameter("@price", DbType.Double) { Value = (double)entryPrice });
        cmd.Parameters.Add(new SQLiteParameter("@entered", DbType.String) { Value = now.ToString("o") });
        cmd.Parameters.Add(new SQLiteParameter("@exit", DbType.String) { Value = scheduledExit.HasValue ? scheduledExit.Value.ToString("o") : (object)DBNull.Value });
        cmd.Parameters.Add(new SQLiteParameter("@weight", DbType.Double) { Value = targetWeight.HasValue ? targetWeight.Value : (object)DBNull.Value });
        cmd.Parameters.Add(new SQLiteParameter("@note", DbType.String) { Value = (object?)note ?? DBNull.Value });
        cmd.ExecuteNonQuery();
    }

    /// <summary>Upsert for ETF rotation: update shares/price/weight if exists, insert if new.</summary>
    public void InsertOrUpdatePosition(string symbol, string strategy, int shares, decimal entryPrice, double? targetWeight = null)
    {
        var now = DateTime.UtcNow;
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"INSERT INTO positions (symbol, strategy, shares, entry_price, entered_at, target_weight, last_rebalanced)
            VALUES (@symbol, @strategy, @shares, @price, @entered, @weight, @rebal)
            ON CONFLICT(symbol) DO UPDATE SET shares=@shares, entry_price=@price, target_weight=@weight, last_rebalanced=@rebal, strategy=@strategy";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@strategy", DbType.String) { Value = strategy });
        cmd.Parameters.Add(new SQLiteParameter("@shares", DbType.Int32) { Value = shares });
        cmd.Parameters.Add(new SQLiteParameter("@price", DbType.Double) { Value = (double)entryPrice });
        cmd.Parameters.Add(new SQLiteParameter("@entered", DbType.String) { Value = now.ToString("o") });
        cmd.Parameters.Add(new SQLiteParameter("@weight", DbType.Double) { Value = targetWeight.HasValue ? targetWeight.Value : (object)DBNull.Value });
        cmd.Parameters.Add(new SQLiteParameter("@rebal", DbType.String) { Value = now.ToString("o") });
        cmd.ExecuteNonQuery();
    }

    /// <summary>Get all positions, optionally filtered by strategy.</summary>
    public IReadOnlyList<PositionRecord> GetPositions(string? strategy = null)
    {
        using var cmd = Connection.CreateCommand();
        if (strategy is not null)
        {
            cmd.CommandText = "SELECT symbol, strategy, shares, entry_price, COALESCE(current_price,0), entered_at, scheduled_exit, target_weight, last_rebalanced, note FROM positions WHERE strategy = @s ORDER BY entered_at DESC";
            cmd.Parameters.Add(new SQLiteParameter("@s", DbType.String) { Value = strategy });
        }
        else
        {
            cmd.CommandText = "SELECT symbol, strategy, shares, entry_price, COALESCE(current_price,0), entered_at, scheduled_exit, target_weight, last_rebalanced, note FROM positions ORDER BY strategy, entered_at DESC";
        }
        using var reader = cmd.ExecuteReader();
        var list = new List<PositionRecord>();
        while (reader.Read())
        {
            list.Add(ReadPositionRecord(reader));
        }
        return list;
    }

    public PositionRecord? GetPosition(string symbol)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT symbol, strategy, shares, entry_price, COALESCE(current_price,0), entered_at, scheduled_exit, target_weight, last_rebalanced, note FROM positions WHERE symbol = @s";
        cmd.Parameters.Add(new SQLiteParameter("@s", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        using var reader = cmd.ExecuteReader();
        return reader.Read() ? ReadPositionRecord(reader) : null;
    }

    private static PositionRecord ReadPositionRecord(SQLiteDataReader reader)
    {
        var symbol = reader.GetString(0);
        var strategy = reader.GetString(1);
        var shares = reader.GetInt32(2);
        var entryPrice = (decimal)reader.GetDouble(3);
        var currentPrice = (decimal)reader.GetDouble(4);
        DateTime.TryParse(reader.IsDBNull(5) ? null : reader.GetString(5), out var enteredAt);
        DateTime? scheduledExit = null;
        if (!reader.IsDBNull(6) && DateTime.TryParse(reader.GetString(6), out var se)) scheduledExit = se;
        double? targetWeight = reader.IsDBNull(7) ? null : reader.GetDouble(7);
        DateTime? lastRebalanced = null;
        if (!reader.IsDBNull(8) && DateTime.TryParse(reader.GetString(8), out var lr)) lastRebalanced = lr;
        var note = reader.IsDBNull(9) ? null : reader.GetString(9);
        return new PositionRecord(symbol, strategy, shares, entryPrice, currentPrice, enteredAt, scheduledExit, targetWeight, lastRebalanced, note);
    }

    public void UpdatePositionShares(string symbol, int shares)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "UPDATE positions SET shares = @shares WHERE symbol = @symbol";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@shares", DbType.Int32) { Value = shares });
        cmd.ExecuteNonQuery();
    }

    public void UpdatePositionPrices(IReadOnlyDictionary<string, decimal> prices)
    {
        if (prices.Count == 0) return;
        using var tx = Connection.BeginTransaction();
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "UPDATE positions SET current_price = @p WHERE symbol = @s";
        var pS = new SQLiteParameter("@s", DbType.String);
        var pP = new SQLiteParameter("@p", DbType.Double);
        cmd.Parameters.Add(pS);
        cmd.Parameters.Add(pP);
        foreach (var (symbol, price) in prices)
        {
            pS.Value = symbol; pP.Value = (double)price;
            cmd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    /// <summary>Sync entry prices from broker_positions avg_cost.</summary>
    public void SyncEntryPricesFromBroker()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"UPDATE positions SET entry_price = (
            SELECT avg_cost FROM broker_positions WHERE broker_positions.symbol = positions.symbol
        ) WHERE EXISTS (
            SELECT 1 FROM broker_positions WHERE broker_positions.symbol = positions.symbol AND avg_cost > 0
        )";
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Sync shares AND entry_price from broker_positions.
    /// Fixes stock splits (broker auto-adjusts qty+avg_cost) and other discrepancies.
    /// Returns the number of positions updated.
    /// </summary>
    public int SyncPositionsFromBroker()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"UPDATE positions SET
            shares = (SELECT quantity FROM broker_positions WHERE broker_positions.symbol = positions.symbol),
            entry_price = (SELECT avg_cost FROM broker_positions WHERE broker_positions.symbol = positions.symbol)
        WHERE EXISTS (
            SELECT 1 FROM broker_positions WHERE broker_positions.symbol = positions.symbol AND quantity > 0 AND avg_cost > 0
        ) AND (
            shares != (SELECT quantity FROM broker_positions WHERE broker_positions.symbol = positions.symbol)
            OR ABS(entry_price - (SELECT avg_cost FROM broker_positions WHERE broker_positions.symbol = positions.symbol)) > 0.01
        )";
        return cmd.ExecuteNonQuery();
    }

    public void DeletePosition(string symbol)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM positions WHERE symbol = @symbol";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.ExecuteNonQuery();
    }

    public void DeleteAllPositions(string strategy)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM positions WHERE strategy = @s";
        cmd.Parameters.Add(new SQLiteParameter("@s", DbType.String) { Value = strategy });
        cmd.ExecuteNonQuery();
    }

    public DateTime? GetLastEtfRebalanceDate()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT MAX(last_rebalanced) FROM positions WHERE strategy = 'ETF'";
        var result = cmd.ExecuteScalar();
        if (result is string s && DateTime.TryParse(s, out var dt))
            return dt;
        return null;
    }

    // --- Trades CRUD ---

    public void InsertTrade(string symbol, string side, int quantity, decimal avgPrice, string? note = null)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT INTO trades (symbol, side, quantity, avg_fill_price, note, created_at) VALUES (@symbol, @side, @qty, @price, @note, @ts)";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@side", DbType.String) { Value = side });
        cmd.Parameters.Add(new SQLiteParameter("@qty", DbType.Int32) { Value = quantity });
        cmd.Parameters.Add(new SQLiteParameter("@price", DbType.Double) { Value = (double)avgPrice });
        cmd.Parameters.Add(new SQLiteParameter("@note", DbType.String) { Value = (object?)note ?? DBNull.Value });
        cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
        cmd.ExecuteNonQuery();
    }

    public IReadOnlyList<TradeRecord> GetRecentTrades(int limit = 50)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT id, symbol, side, quantity, COALESCE(avg_fill_price, 0), COALESCE(created_at, CURRENT_TIMESTAMP), note FROM trades ORDER BY id DESC LIMIT @limit";
        cmd.Parameters.Add(new SQLiteParameter("@limit", DbType.Int32) { Value = limit });
        using var reader = cmd.ExecuteReader();
        var list = new List<TradeRecord>();
        while (reader.Read())
        {
            var id = reader.GetInt32(0);
            var symbol = reader.GetString(1);
            var side = reader.GetString(2);
            var qty = reader.GetInt32(3);
            var price = (decimal)reader.GetDouble(4);
            DateTime.TryParse(reader.GetString(5), out var createdAt);
            var note = reader.IsDBNull(6) ? null : reader.GetString(6);
            list.Add(new TradeRecord(id, symbol, side, qty, price, createdAt, note));
        }
        return list;
    }

    public IReadOnlyList<TradeRecord> GetRecentTrades(string symbol, int hours = 24)
    {
        using var cmd = Connection.CreateCommand();
        var cutoff = DateTime.UtcNow.AddHours(-hours).ToString("o");
        cmd.CommandText = "SELECT id, symbol, side, quantity, COALESCE(avg_fill_price, 0), COALESCE(created_at, CURRENT_TIMESTAMP), note FROM trades WHERE symbol = @sym AND created_at >= @cutoff ORDER BY id DESC";
        cmd.Parameters.Add(new SQLiteParameter("@sym", DbType.String) { Value = symbol });
        cmd.Parameters.Add(new SQLiteParameter("@cutoff", DbType.String) { Value = cutoff });
        using var reader = cmd.ExecuteReader();
        var list = new List<TradeRecord>();
        while (reader.Read())
        {
            var id = reader.GetInt32(0);
            var sym = reader.GetString(1);
            var side = reader.GetString(2);
            var qty = reader.GetInt32(3);
            var price = (decimal)reader.GetDouble(4);
            DateTime.TryParse(reader.GetString(5), out var createdAt);
            var note = reader.IsDBNull(6) ? null : reader.GetString(6);
            list.Add(new TradeRecord(id, sym, side, qty, price, createdAt, note));
        }
        return list;
    }

    private static DateTime AddTradingDays(DateTime start, int days)
    {
        var current = start;
        var added = 0;
        while (added < days)
        {
            current = current.AddDays(1);
            if (IsTradingDay(current))
                added++;
        }
        return current;
    }

    // --- Capital History CRUD ---

    public void InsertCapitalHistory(decimal netLiq, decimal cash, decimal stockValue)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT INTO capital_history (timestamp, net_liq, cash, stock_value) VALUES (@ts, @netliq, @cash, @stockval)";
        cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
        cmd.Parameters.Add(new SQLiteParameter("@netliq", DbType.Double) { Value = (double)netLiq });
        cmd.Parameters.Add(new SQLiteParameter("@cash", DbType.Double) { Value = (double)cash });
        cmd.Parameters.Add(new SQLiteParameter("@stockval", DbType.Double) { Value = (double)stockValue });
        cmd.ExecuteNonQuery();
    }

    public IReadOnlyList<CapitalHistoryRecord> GetCapitalHistory(int limit = 500)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT id, timestamp, net_liq, cash, stock_value FROM capital_history ORDER BY timestamp DESC LIMIT @limit";
        cmd.Parameters.Add(new SQLiteParameter("@limit", DbType.Int32) { Value = limit });
        using var reader = cmd.ExecuteReader();
        var list = new List<CapitalHistoryRecord>();
        while (reader.Read())
        {
            var id = reader.GetInt32(0);
            var timestampStr = reader.GetString(1);
            var netLiq = (decimal)reader.GetDouble(2);
            var cash = (decimal)reader.GetDouble(3);
            var stockValue = (decimal)reader.GetDouble(4);
            DateTime.TryParse(timestampStr, out var timestamp);
            list.Add(new CapitalHistoryRecord(id, timestamp, netLiq, cash, stockValue));
        }
        list.Reverse();
        return list;
    }

    public IReadOnlyList<CapitalHistoryRecord> GetCapitalHistorySince(DateTime since)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT id, timestamp, net_liq, cash, stock_value FROM capital_history WHERE timestamp >= @ts ORDER BY timestamp ASC";
        cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = since.ToString("o") });
        using var reader = cmd.ExecuteReader();
        var list = new List<CapitalHistoryRecord>();
        while (reader.Read())
        {
            var id = reader.GetInt32(0);
            var timestampStr = reader.GetString(1);
            var netLiq = (decimal)reader.GetDouble(2);
            var cash = (decimal)reader.GetDouble(3);
            var stockValue = (decimal)reader.GetDouble(4);
            DateTime.TryParse(timestampStr, out var timestamp);
            list.Add(new CapitalHistoryRecord(id, timestamp, netLiq, cash, stockValue));
        }
        return list;
    }

    public CapitalHistoryRecord? GetFirstCapitalRecordSince(DateTime since)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT id, timestamp, net_liq, cash, stock_value FROM capital_history WHERE timestamp >= @ts ORDER BY timestamp ASC LIMIT 1";
        cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = since.ToString("o") });
        using var reader = cmd.ExecuteReader();
        if (reader.Read())
        {
            var id = reader.GetInt32(0);
            var timestampStr = reader.GetString(1);
            var netLiq = (decimal)reader.GetDouble(2);
            var cash = (decimal)reader.GetDouble(3);
            var stockValue = (decimal)reader.GetDouble(4);
            DateTime.TryParse(timestampStr, out var timestamp);
            return new CapitalHistoryRecord(id, timestamp, netLiq, cash, stockValue);
        }
        return null;
    }

    /// <summary>Delete capital history records older than the specified number of days.</summary>
    public int TrimCapitalHistory(int keepDays = 90)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM capital_history WHERE timestamp < @cutoff";
        cmd.Parameters.Add(new SQLiteParameter("@cutoff", DbType.String) { Value = DateTime.UtcNow.AddDays(-keepDays).ToString("o") });
        return cmd.ExecuteNonQuery();
    }

    // --- ETF Rotation State ---

    public string? GetEtfRotationState(string key)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT value FROM etf_rotation_state WHERE key = @key";
        cmd.Parameters.Add(new SQLiteParameter("@key", DbType.String) { Value = key });
        var result = cmd.ExecuteScalar();
        return result as string;
    }

    public void SetEtfRotationState(string key, string value)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT OR REPLACE INTO etf_rotation_state (key, value) VALUES (@key, @value)";
        cmd.Parameters.Add(new SQLiteParameter("@key", DbType.String) { Value = key });
        cmd.Parameters.Add(new SQLiteParameter("@value", DbType.String) { Value = value });
        cmd.ExecuteNonQuery();
    }

    public void SetEtfRotationStateBatch(IEnumerable<(string Key, string Value)> pairs)
    {
        using var tx = Connection.BeginTransaction();
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT OR REPLACE INTO etf_rotation_state (key, value) VALUES (@key, @value)";
        var pKey = new SQLiteParameter("@key", DbType.String);
        var pValue = new SQLiteParameter("@value", DbType.String);
        cmd.Parameters.Add(pKey);
        cmd.Parameters.Add(pValue);
        foreach (var (key, value) in pairs)
        {
            pKey.Value = key;
            pValue.Value = value;
            cmd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    // --- NYSE Trading Day Utilities ---

    public static int CountTradingDaysBetween(DateTime startDate, DateTime endDate)
    {
        if (endDate <= startDate) return 0;
        var count = 0;
        var current = startDate.Date.AddDays(1);
        var end = endDate.Date;
        while (current <= end)
        {
            if (IsTradingDay(current)) count++;
            current = current.AddDays(1);
        }
        return count;
    }

    public static bool IsTradingDay(DateTime date)
    {
        if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
            return false;
        return !IsNyseHoliday(date);
    }

    // ── Dynamic NYSE holiday calendar (loaded from pandas_market_calendars) ──
    private static HashSet<DateTime>? _externalHolidays;
    private static readonly object _holidayLock = new();

    /// <summary>
    /// Load NYSE holidays from pandas_market_calendars (called once at startup).
    /// Falls back to hardcoded calculator if not loaded.
    /// </summary>
    public static void LoadNyseHolidays(IEnumerable<DateTime> holidays)
    {
        lock (_holidayLock)
        {
            _externalHolidays = new HashSet<DateTime>(holidays.Select(d => d.Date));
        }
    }

    public static bool IsNyseHoliday(DateTime date)
    {
        var d = date.Date;

        // Use external calendar if loaded
        lock (_holidayLock)
        {
            if (_externalHolidays is not null)
                return _externalHolidays.Contains(d);
        }

        // Fallback: hardcoded calculator
        return IsNyseHolidayHardcoded(d);
    }

    private static bool IsNyseHolidayHardcoded(DateTime d)
    {
        var year = d.Year;
        var holidays = new List<DateTime>
        {
            AdjustForWeekend(new DateTime(year, 1, 1)),
            AdjustForWeekend(new DateTime(year, 6, 19)),
            AdjustForWeekend(new DateTime(year, 7, 4)),
            AdjustForWeekend(new DateTime(year, 12, 25)),
        };
        holidays.Add(NthWeekday(year, 1, DayOfWeek.Monday, 3));
        holidays.Add(NthWeekday(year, 2, DayOfWeek.Monday, 3));
        holidays.Add(LastWeekday(year, 5, DayOfWeek.Monday));
        holidays.Add(NthWeekday(year, 9, DayOfWeek.Monday, 1));
        holidays.Add(NthWeekday(year, 11, DayOfWeek.Thursday, 4));
        holidays.Add(EasterSunday(year).AddDays(-2));
        return holidays.Any(h => h.Date == d);
    }

    private static DateTime AdjustForWeekend(DateTime holiday)
    {
        if (holiday.DayOfWeek == DayOfWeek.Saturday) return holiday.AddDays(-1);
        if (holiday.DayOfWeek == DayOfWeek.Sunday) return holiday.AddDays(1);
        return holiday;
    }

    private static DateTime NthWeekday(int year, int month, DayOfWeek dow, int n)
    {
        var first = new DateTime(year, month, 1);
        var diff = ((int)dow - (int)first.DayOfWeek + 7) % 7;
        return first.AddDays(diff + 7 * (n - 1));
    }

    private static DateTime LastWeekday(int year, int month, DayOfWeek dow)
    {
        var last = new DateTime(year, month, DateTime.DaysInMonth(year, month));
        var diff = ((int)last.DayOfWeek - (int)dow + 7) % 7;
        return last.AddDays(-diff);
    }

    private static DateTime EasterSunday(int year)
    {
        int a = year % 19, b = year / 100, c = year % 100;
        int d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
        int h = (19 * a + b - d - g + 15) % 30;
        int i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7;
        int m = (a + 11 * h + 22 * l) / 451;
        int month = (h + l - 7 * m + 114) / 31;
        int day = ((h + l - 7 * m + 114) % 31) + 1;
        return new DateTime(year, month, day);
    }

    public static DateTime NextTradingDayOpenUtc(DateTime fromUtc)
    {
        var et = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
        var etNow = TimeZoneInfo.ConvertTimeFromUtc(fromUtc, et);
        var candidate = etNow.Date;
        var marketOpen = candidate.AddHours(9).AddMinutes(30);
        if (etNow >= marketOpen) candidate = candidate.AddDays(1);
        while (!IsTradingDay(candidate)) candidate = candidate.AddDays(1);
        var nextOpen = candidate.AddHours(9).AddMinutes(30);
        return TimeZoneInfo.ConvertTimeToUtc(nextOpen, et);
    }

    // --- Broker Positions CRUD (IBKR mirror) ---

    public void ReplaceBrokerPositions(IEnumerable<(string Symbol, int Quantity, decimal AvgCost, decimal MarketValue)> positions)
    {
        using var tx = Connection.BeginTransaction();
        using (var del = Connection.CreateCommand()) { del.CommandText = "DELETE FROM broker_positions"; del.ExecuteNonQuery(); }
        var now = DateTime.UtcNow.ToString("o");
        foreach (var (symbol, qty, avgCost, marketValue) in positions)
        {
            if (qty == 0) continue;
            using var cmd = Connection.CreateCommand();
            cmd.CommandText = @"INSERT INTO broker_positions (symbol, quantity, avg_cost, market_value, last_synced)
                VALUES (@symbol, @qty, @cost, @value, @synced)";
            cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
            cmd.Parameters.Add(new SQLiteParameter("@qty", DbType.Int32) { Value = qty });
            cmd.Parameters.Add(new SQLiteParameter("@cost", DbType.Double) { Value = (double)avgCost });
            cmd.Parameters.Add(new SQLiteParameter("@value", DbType.Double) { Value = (double)marketValue });
            cmd.Parameters.Add(new SQLiteParameter("@synced", DbType.String) { Value = now });
            cmd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    public IReadOnlyList<BrokerPositionRecord> GetBrokerPositions()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT symbol, quantity, avg_cost, market_value, last_synced FROM broker_positions ORDER BY symbol";
        using var reader = cmd.ExecuteReader();
        var list = new List<BrokerPositionRecord>();
        while (reader.Read())
        {
            var symbol = reader.GetString(0);
            var qty = reader.GetInt32(1);
            var avgCost = (decimal)reader.GetDouble(2);
            var marketValue = (decimal)reader.GetDouble(3);
            DateTime.TryParse(reader.GetString(4), out var lastSynced);
            list.Add(new BrokerPositionRecord(symbol, qty, avgCost, marketValue, lastSynced));
        }
        return list;
    }

    // --- Trade Intents CRUD ---

    public static string GenerateIntentId(string strategy, string symbol, string side, DateTime? dateBucket = null)
    {
        var date = (dateBucket ?? DateTime.UtcNow).ToString("yyyy-MM-dd");
        var input = $"{strategy}|{symbol.Trim().ToUpperInvariant()}|{side}|{date}";
        var hash = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    public bool IsIntentExecuted(string intentId)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT status FROM trade_intents WHERE intent_id = @id";
        cmd.Parameters.Add(new SQLiteParameter("@id", DbType.String) { Value = intentId });
        var result = cmd.ExecuteScalar();
        return result is string s && s == "executed";
    }

    public bool TryInsertTradeIntent(string intentId, string strategy, string symbol, string side, int quantity)
    {
        try
        {
            using var cmd = Connection.CreateCommand();
            cmd.CommandText = @"INSERT INTO trade_intents (intent_id, strategy, symbol, side, quantity, status, created_at)
                VALUES (@id, @strat, @symbol, @side, @qty, 'pending', @ts)";
            cmd.Parameters.Add(new SQLiteParameter("@id", DbType.String) { Value = intentId });
            cmd.Parameters.Add(new SQLiteParameter("@strat", DbType.String) { Value = strategy });
            cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
            cmd.Parameters.Add(new SQLiteParameter("@side", DbType.String) { Value = side });
            cmd.Parameters.Add(new SQLiteParameter("@qty", DbType.Int32) { Value = quantity });
            cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
            cmd.ExecuteNonQuery();
            return true;
        }
        catch { return false; }
    }

    public void MarkIntentExecuted(string intentId, int orderId)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"UPDATE trade_intents SET status = 'executed', order_id = @oid, executed_at = @now WHERE intent_id = @id";
        cmd.Parameters.Add(new SQLiteParameter("@id", DbType.String) { Value = intentId });
        cmd.Parameters.Add(new SQLiteParameter("@oid", DbType.Int32) { Value = orderId });
        cmd.Parameters.Add(new SQLiteParameter("@now", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
        cmd.ExecuteNonQuery();
    }

    public void MarkIntentFailed(string intentId, string error)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"UPDATE trade_intents SET status = 'failed', error = @err, executed_at = @now WHERE intent_id = @id";
        cmd.Parameters.Add(new SQLiteParameter("@id", DbType.String) { Value = intentId });
        cmd.Parameters.Add(new SQLiteParameter("@err", DbType.String) { Value = error });
        cmd.Parameters.Add(new SQLiteParameter("@now", DbType.String) { Value = DateTime.UtcNow.ToString("o") });
        cmd.ExecuteNonQuery();
    }

    // ── pending_buys ────────────────────────────────────────────────────

    public void ReplacePendingBuys(IEnumerable<(string Symbol, int Rank, double Score)> tickers)
    {
        using var tx = Connection.BeginTransaction();
        using (var del = Connection.CreateCommand()) { del.CommandText = "DELETE FROM pending_buys"; del.ExecuteNonQuery(); }
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT INTO pending_buys (symbol, rank, score, created_at) VALUES (@s, @r, @sc, @ts)";
        var pS = new SQLiteParameter("@s", DbType.String);
        var pR = new SQLiteParameter("@r", DbType.Int32);
        var pSc = new SQLiteParameter("@sc", DbType.Double);
        var pTs = new SQLiteParameter("@ts", DbType.String);
        cmd.Parameters.Add(pS);
        cmd.Parameters.Add(pR);
        cmd.Parameters.Add(pSc);
        cmd.Parameters.Add(pTs);
        var now = DateTime.UtcNow.ToString("o");
        foreach (var (symbol, rank, score) in tickers)
        {
            pS.Value = symbol; pR.Value = rank; pSc.Value = score; pTs.Value = now;
            cmd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    public IReadOnlyList<PendingBuyRecord> GetPendingBuys()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT symbol, rank, score, created_at FROM pending_buys ORDER BY rank ASC";
        using var reader = cmd.ExecuteReader();
        var list = new List<PendingBuyRecord>();
        while (reader.Read())
        {
            var symbol = reader.GetString(0);
            var rank = reader.GetInt32(1);
            var score = reader.GetDouble(2);
            DateTime.TryParse(reader.GetString(3), out var createdAt);
            list.Add(new PendingBuyRecord(symbol, rank, score, createdAt));
        }
        return list;
    }

    public void ClearPendingBuys()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM pending_buys";
        cmd.ExecuteNonQuery();
    }

    // ── Integrity Check & Backup (FIX G) ──────────────────────────────

    public bool RunIntegrityCheck()
    {
        try
        {
            using var cmd = Connection.CreateCommand();
            cmd.CommandText = "PRAGMA integrity_check";
            var result = cmd.ExecuteScalar()?.ToString();
            return result == "ok";
        }
        catch
        {
            return false;
        }
    }

    public void BackupTo(string destPath)
    {
        // Force WAL checkpoint before backup to ensure all data is in main DB file
        using (var cmd = Connection.CreateCommand())
        {
            cmd.CommandText = "PRAGMA wal_checkpoint(TRUNCATE)";
            cmd.ExecuteNonQuery();
        }
        File.Copy(_dbPath, destPath, overwrite: true);
    }

    public void Dispose()
    {
        lock (_connLock)
        {
            if (_activeConnection is not null)
            {
                try { _activeConnection.Close(); } catch { }
                _activeConnection.Dispose();
                _activeConnection = null;
            }
        }
    }
}

public sealed record PositionRecord(string Symbol, string Strategy, int Shares, decimal EntryPrice, decimal CurrentPrice, DateTime EnteredAt, DateTime? ScheduledExit, double? TargetWeight, DateTime? LastRebalanced, string? Note);

public sealed record PendingBuyRecord(string Symbol, int Rank, double Score, DateTime CreatedAt);

public sealed record BrokerPositionRecord(string Symbol, int Quantity, decimal AvgCost, decimal MarketValue, DateTime LastSynced);

public sealed record TradeRecord(int Id, string Symbol, string Side, int Quantity, decimal AvgFillPrice, DateTime CreatedAt, string? Note = null)
{
    public string Action => Side;
}

public sealed record CapitalHistoryRecord(int Id, DateTime Timestamp, decimal NetLiq, decimal Cash, decimal StockValue);
