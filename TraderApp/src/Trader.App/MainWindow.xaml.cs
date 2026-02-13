using System;
using System.ComponentModel;
using System.Drawing;
using System.Windows;
using System.Windows.Threading;
using Trader.App.ViewModels;
using WinForms = System.Windows.Forms;

namespace Trader.App;

public partial class MainWindow : Window
{
    private WinForms.NotifyIcon? _trayIcon;
    private bool _isExiting;
    private readonly DispatcherTimer _tradingHoursTimer;

    public MainWindow(ShellViewModel shellViewModel)
    {
        InitializeComponent();
        DataContext = shellViewModel;

        CreateTrayIcon();

        // Timer to auto-show window during US trading hours
        _tradingHoursTimer = new DispatcherTimer { Interval = TimeSpan.FromMinutes(1) };
        _tradingHoursTimer.Tick += TradingHoursTimer_Tick;
        _tradingHoursTimer.Start();
    }

    private void CreateTrayIcon()
    {
        var contextMenu = new WinForms.ContextMenuStrip();
        contextMenu.Items.Add("Show Trader", null, (_, _) => ShowFromTray());
        contextMenu.Items.Add(new WinForms.ToolStripSeparator());
        contextMenu.Items.Add("Exit", null, (_, _) => ExitApplication());

        _trayIcon = new WinForms.NotifyIcon
        {
            Text = "Trader AutoPilot",
            Icon = CreateDefaultIcon(),
            Visible = true,
            ContextMenuStrip = contextMenu,
        };

        _trayIcon.DoubleClick += (_, _) => ShowFromTray();
    }

    private static System.Drawing.Icon CreateDefaultIcon()
    {
        // Create a simple green "$" icon
        var bmp = new System.Drawing.Bitmap(16, 16);
        using (var g = System.Drawing.Graphics.FromImage(bmp))
        {
            g.Clear(System.Drawing.Color.FromArgb(46, 204, 113));
            using var font = new System.Drawing.Font("Arial", 9, System.Drawing.FontStyle.Bold);
            using var brush = new System.Drawing.SolidBrush(System.Drawing.Color.White);
            g.DrawString("$", font, brush, 1, 0);
        }
        return System.Drawing.Icon.FromHandle(bmp.GetHicon());
    }

    internal void ShowFromTray()
    {
        Show();
        WindowState = WindowState.Normal;
        Activate();
    }

    private void ExitButton_Click(object sender, System.Windows.RoutedEventArgs e) => ExitApplication();

    private void ExitApplication()
    {
        _isExiting = true;
        _trayIcon?.Dispose();
        _trayIcon = null;
        _tradingHoursTimer.Stop();
        System.Windows.Application.Current.Shutdown();
    }

    protected override void OnClosing(CancelEventArgs e)
    {
        if (!_isExiting)
        {
            // Minimize to tray instead of closing
            e.Cancel = true;
            Hide();
            _trayIcon?.ShowBalloonTip(2000, "Trader AutoPilot",
                "Running in background. Double-click tray icon to show.",
                WinForms.ToolTipIcon.Info);
            return;
        }
        base.OnClosing(e);
    }

    private void TradingHoursTimer_Tick(object? sender, EventArgs e)
    {
        // Auto-show during US trading hours (9:25 AM - 4:05 PM ET, Mon-Fri)
        // Slightly wider window to ensure app is visible before market opens
        try
        {
            var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));

            if (et.DayOfWeek == DayOfWeek.Saturday || et.DayOfWeek == DayOfWeek.Sunday)
                return;

            var preOpen = new TimeSpan(9, 25, 0);
            var postClose = new TimeSpan(16, 5, 0);
            var inTradingWindow = et.TimeOfDay >= preOpen && et.TimeOfDay <= postClose;

            if (inTradingWindow && !IsVisible)
            {
                ShowFromTray();
            }
        }
        catch
        {
            // Timezone not found — skip auto-show
        }
    }

    protected override void OnClosed(EventArgs e)
    {
        _trayIcon?.Dispose();
        _tradingHoursTimer.Stop();
        base.OnClosed(e);
    }
}
