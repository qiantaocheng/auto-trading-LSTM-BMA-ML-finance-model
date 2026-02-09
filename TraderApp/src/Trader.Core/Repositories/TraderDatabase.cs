using System.Data;
using System.Data.SQLite;

namespace Trader.Core.Repositories;

public sealed record TickerRecord(string Symbol, int Tag, string Source, DateTime AddedAt);

public sealed class TraderDatabase : IDisposable
{
    private readonly string _dbPath;
    private readonly Lazy<SQLiteConnection> _connection;

    public TraderDatabase(string dbPath)
    {
        _dbPath = dbPath;
        _connection = new Lazy<SQLiteConnection>(() =>
        {
            var connection = new SQLiteConnection($"Data Source={_dbPath};Version=3;");
            connection.Open();
            return connection;
        });
    }

    private SQLiteConnection Connection => _connection.Value;

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

CREATE TABLE IF NOT EXISTS auto_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entered_at TEXT NOT NULL,
    scheduled_exit TEXT NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL,
    UNIQUE(symbol)
);

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
";
        cmd.ExecuteNonQuery();

        // Migrations: add columns to existing tables
        TryAddColumn("trades", "note", "TEXT");
        TryAddColumn("auto_positions", "note", "TEXT");
    }

    private void TryAddColumn(string table, string column, string type)
    {
        try
        {
            using var cmd = Connection.CreateCommand();
            cmd.CommandText = $"ALTER TABLE {table} ADD COLUMN {column} {type}";
            cmd.ExecuteNonQuery();
        }
        catch
        {
            // Column already exists — ignore
        }
    }

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
            if (string.IsNullOrWhiteSpace(symbol))
            {
                continue;
            }
            using var insert = Connection.CreateCommand();
            insert.CommandText = "INSERT OR REPLACE INTO tickers(symbol, tag, source, added_at) VALUES(@symbol, 0, @source, CURRENT_TIMESTAMP)";
            insert.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
            insert.Parameters.Add(new SQLiteParameter("@source", DbType.String) { Value = source });
            insert.ExecuteNonQuery();
        }

        tx.Commit();
    }

    public void AddManualTicker(string symbol, string source)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT OR REPLACE INTO tickers(symbol, tag, source, added_at) VALUES(@symbol, 1, @source, CURRENT_TIMESTAMP)";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@source", DbType.String) { Value = source });
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
            cmd.CommandText = "INSERT INTO direct_predictions (ts, snapshot_id, as_of, ticker, score, ema4) VALUES (@ts, @sid, @asOf, @ticker, @score, @ema4)";
            cmd.Parameters.Add(new SQLiteParameter("@ts", DbType.Int64) { Value = ts });
            cmd.Parameters.Add(new SQLiteParameter("@sid", DbType.String) { Value = snapshotId });
            cmd.Parameters.Add(new SQLiteParameter("@asOf", DbType.String) { Value = asOf });
            cmd.Parameters.Add(new SQLiteParameter("@ticker", DbType.String) { Value = ticker.Trim().ToUpperInvariant() });
            cmd.Parameters.Add(new SQLiteParameter("@score", DbType.Double) { Value = score });
            cmd.Parameters.Add(new SQLiteParameter("@ema4", DbType.Double) { Value = ema4 });
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

    // --- Auto Positions CRUD ---

    public void InsertAutoPosition(string symbol, int shares, decimal entryPrice, int holdDays = 5, string? note = null)
    {
        var now = DateTime.UtcNow;
        var scheduledExit = AddTradingDays(now, holdDays);
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = @"INSERT OR REPLACE INTO auto_positions (symbol, entered_at, scheduled_exit, shares, entry_price, note)
            VALUES (@symbol, @entered, @exit, @shares, @price, @note)";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@entered", DbType.String) { Value = now.ToString("o") });
        cmd.Parameters.Add(new SQLiteParameter("@exit", DbType.String) { Value = scheduledExit.ToString("o") });
        cmd.Parameters.Add(new SQLiteParameter("@shares", DbType.Int32) { Value = shares });
        cmd.Parameters.Add(new SQLiteParameter("@price", DbType.Double) { Value = (double)entryPrice });
        cmd.Parameters.Add(new SQLiteParameter("@note", DbType.String) { Value = (object?)note ?? DBNull.Value });
        cmd.ExecuteNonQuery();
    }

    public IReadOnlyList<AutoPositionRecord> GetAutoPositions()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT symbol, shares, entry_price, entered_at, scheduled_exit FROM auto_positions ORDER BY entered_at DESC";
        using var reader = cmd.ExecuteReader();
        var list = new List<AutoPositionRecord>();
        while (reader.Read())
        {
            var symbol = reader.GetString(0);
            var shares = reader.GetInt32(1);
            var entryPrice = (decimal)reader.GetDouble(2);
            DateTime.TryParse(reader.GetString(3), out var enteredAt);
            DateTime.TryParse(reader.GetString(4), out var scheduledExit);
            list.Add(new AutoPositionRecord(symbol, shares, entryPrice, enteredAt, scheduledExit));
        }
        return list;
    }

    public AutoPositionRecord? GetAutoPosition(string symbol)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT symbol, shares, entry_price, entered_at, scheduled_exit FROM auto_positions WHERE symbol = @symbol";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        using var reader = cmd.ExecuteReader();
        if (reader.Read())
        {
            var sym = reader.GetString(0);
            var shares = reader.GetInt32(1);
            var entryPrice = (decimal)reader.GetDouble(2);
            DateTime.TryParse(reader.GetString(3), out var enteredAt);
            DateTime.TryParse(reader.GetString(4), out var scheduledExit);
            return new AutoPositionRecord(sym, shares, entryPrice, enteredAt, scheduledExit);
        }
        return null;
    }

    public void DeleteAutoPosition(string symbol)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM auto_positions WHERE symbol = @symbol";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.ExecuteNonQuery();
    }

    public void DeleteAllAutoPositions()
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "DELETE FROM auto_positions";
        cmd.ExecuteNonQuery();
    }

    // --- Trades CRUD ---

    public void InsertTrade(string symbol, string side, int quantity, decimal avgPrice, string? note = null)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "INSERT INTO trades (symbol, side, quantity, avg_fill_price, note) VALUES (@symbol, @side, @qty, @price, @note)";
        cmd.Parameters.Add(new SQLiteParameter("@symbol", DbType.String) { Value = symbol.Trim().ToUpperInvariant() });
        cmd.Parameters.Add(new SQLiteParameter("@side", DbType.String) { Value = side });
        cmd.Parameters.Add(new SQLiteParameter("@qty", DbType.Int32) { Value = quantity });
        cmd.Parameters.Add(new SQLiteParameter("@price", DbType.Double) { Value = (double)avgPrice });
        cmd.Parameters.Add(new SQLiteParameter("@note", DbType.String) { Value = (object?)note ?? DBNull.Value });
        cmd.ExecuteNonQuery();
    }

    public IReadOnlyList<TradeRecord> GetRecentTrades(int limit = 50)
    {
        using var cmd = Connection.CreateCommand();
        cmd.CommandText = "SELECT id, symbol, side, quantity, COALESCE(avg_fill_price, 0), COALESCE(created_at, CURRENT_TIMESTAMP) FROM trades ORDER BY id DESC LIMIT @limit";
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
            list.Add(new TradeRecord(id, symbol, side, qty, price, createdAt));
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
            if (current.DayOfWeek != DayOfWeek.Saturday && current.DayOfWeek != DayOfWeek.Sunday)
            {
                added++;
            }
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
        list.Reverse(); // Return in chronological order (oldest first)
        return list;
    }

    public void Dispose()
    {
        if (_connection.IsValueCreated)
        {
            _connection.Value.Dispose();
        }
    }
}

public sealed record AutoPositionRecord(string Symbol, int Shares, decimal EntryPrice, DateTime EnteredAt, DateTime ScheduledExit);

public sealed record TradeRecord(int Id, string Symbol, string Side, int Quantity, decimal AvgFillPrice, DateTime CreatedAt);

public sealed record CapitalHistoryRecord(int Id, DateTime Timestamp, decimal NetLiq, decimal Cash, decimal StockValue);
