using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using Trader.App.ViewModels.Pages;

namespace Trader.App.Converters;

public sealed class CapitalHistoryGeometryConverter : IMultiValueConverter
{
    public bool IncludeBaseline { get; set; }

    public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
    {
        if (values.Length < 3)
        {
            return Geometry.Empty;
        }

        if (values[0] is not IEnumerable<CapitalPoint> pointsEnumerable)
        {
            return Geometry.Empty;
        }

        if (values[1] is not double width || values[2] is not double height)
        {
            return Geometry.Empty;
        }

        if (width <= 0 || height <= 0)
        {
            return Geometry.Empty;
        }

        var points = pointsEnumerable.ToList();
        if (points.Count < 2)
        {
            return Geometry.Empty;
        }

        var min = points.Min(p => p.NetLiq);
        var max = points.Max(p => p.NetLiq);
        var range = Math.Max(1e-6m, max - min);

        double MapY(decimal net) => height - (double)((net - min) / range) * height;

        var geometry = new PathGeometry();
        var figure = new PathFigure();

        if (IncludeBaseline)
        {
            figure.StartPoint = new Point(0, height);
            var firstY = MapY(points.First().NetLiq);
            figure.Segments.Add(new LineSegment(new Point(0, firstY), true));
        }
        else
        {
            figure.StartPoint = new Point(0, MapY(points.First().NetLiq));
        }

        for (int i = 1; i < points.Count; i++)
        {
            var x = width * i / (points.Count - 1);
            var y = MapY(points[i].NetLiq);
            figure.Segments.Add(new LineSegment(new Point(x, y), true));
        }

        if (IncludeBaseline)
        {
            figure.Segments.Add(new LineSegment(new Point(width, height), true));
            figure.IsClosed = true;
        }

        geometry.Figures.Add(figure);
        return geometry;
    }

    public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
