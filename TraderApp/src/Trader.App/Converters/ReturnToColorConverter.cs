using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Trader.App.Converters;

public sealed class ReturnToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double d)
        {
            if (d > 0) return new SolidColorBrush(Color.FromRgb(0x2E, 0xCC, 0x71)); // green
            if (d < 0) return new SolidColorBrush(Color.FromRgb(0xE7, 0x4C, 0x3C)); // red
        }
        return new SolidColorBrush(Color.FromRgb(0x71, 0x71, 0x82)); // muted
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
