#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Quantitative Trading System Launcher
一键启动专业量化交易系统
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

def main():
    print("🚀 Professional Quantitative Trading System V5")
    print("=" * 60)
    print("顶级金融机构级别的量化交易系统")
    print("集成: Multi-factor Risk Model + Dynamic Alpha + Regime-Aware BMA + Professional Portfolio Optimization")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='Professional Quantitative Trading System')
    parser.add_argument('--mode', choices=['professional', 'ultra', 'original'], 
                       default='professional', help='选择运行模式')
    parser.add_argument('--start-date', type=str, 
                       default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                       help='开始日期 (默认: 2年前)')
    parser.add_argument('--end-date', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='结束日期 (默认: 今天)')
    parser.add_argument('--stocks', type=str, nargs='+', 
                       help='自定义股票列表 (例如: AAPL MSFT GOOGL)')
    parser.add_argument('--top-n', type=int, default=10, help='返回推荐数量')
    
    args = parser.parse_args()
    
    print(f"📋 运行配置:")
    print(f"  模式: {args.mode}")
    print(f"  时间范围: {args.start_date} 至 {args.end_date}")
    print(f"  推荐数量: {args.top_n}")
    
    try:
        if args.mode == 'professional':
            # 运行专业版引擎
            print("\n🎯 启动专业量化引擎...")
            from quant_engine_professional import ProfessionalQuantEngine
            
            engine = ProfessionalQuantEngine()
            
            # 确定股票池
            if args.stocks:
                tickers = [s.upper() for s in args.stocks]
                print(f"  使用自定义股票池: {tickers}")
            else:
                # 使用专业股票池
                tickers = [
                    # 科技巨头
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                    # 金融
                    'JPM', 'BAC', 'GS', 'BRK-B', 'WFC',
                    # 医疗保健  
                    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
                    # 消费品
                    'PG', 'KO', 'WMT', 'HD', 'MCD',
                    # 工业
                    'BA', 'CAT', 'GE', 'MMM',
                    # 高成长科技
                    'CRM', 'ADBE', 'PYPL', 'AMD', 'QCOM'
                ]
                print(f"  使用专业股票池: {len(tickers)}只股票")
            
            # 运行完整分析
            results = engine.run_complete_analysis(
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            display_professional_results(results, args.top_n)
            
        elif args.mode == 'ultra':
            # 运行Ultra Enhanced版本
            print("\n🚀 启动Ultra Enhanced BMA引擎...")
            from 量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
            
            model = UltraEnhancedQuantitativeModel()
            
            tickers = args.stocks if args.stocks else [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
                'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO'
            ]
            
            results = model.run_complete_analysis(
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=args.top_n
            )
            
            display_ultra_results(results)
            
        elif args.mode == 'original':
            # 运行原始BMA Enhanced版本
            print("\n📊 启动原始BMA Enhanced引擎...")
            from 量化模型_bma_enhanced import QuantitativeModel
            
            model = QuantitativeModel()
            
            tickers = args.stocks if args.stocks else [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'
            ]
            
            # 下载数据
            all_data = model.download_data(tickers, args.start_date, args.end_date)
            
            if not all_data:
                print("❌ 数据下载失败")
                return
            
            # 准备ML数据
            X, y, ticker_series, dates = model.prepare_ml_data_with_time_series(all_data, target_period=5)
            
            # 训练模型
            model_scores = model.train_models_with_bma(X, y, enable_hyperopt=True, 
                                                     apply_preprocessing=True, 
                                                     dates=dates, tickers=ticker_series)
            
            # 生成推荐
            recommendations = model.generate_recommendations(all_data, top_n=args.top_n)
            
            display_original_results(recommendations)
            
    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        print("💡 建议:")
        print("  1. 检查网络连接")
        print("  2. 确保安装了所有依赖包")
        print("  3. 尝试使用更小的股票池")
        return 1
    
    return 0

def display_professional_results(results: dict, top_n: int):
    """显示专业版结果"""
    print("\n" + "="*60)
    print("🎯 PROFESSIONAL QUANTITATIVE ANALYSIS RESULTS")
    print("="*60)
    
    summary = results.get('analysis_summary', {})
    
    # 状态和质量指标
    status_icon = "✅" if summary.get('status') == 'SUCCESS' else "❌"
    print(f"{status_icon} 分析状态: {summary.get('status', 'UNKNOWN')}")
    print(f"⏱️  总耗时: {summary.get('total_time_seconds', 0):.1f}秒")
    
    quality_indicators = []
    if summary.get('data_quality') == 'HIGH':
        quality_indicators.append("📊 数据质量: 优秀")
    if summary.get('model_quality') == 'HIGH':
        quality_indicators.append("🔬 模型质量: 优秀")
    if summary.get('signal_quality') == 'HIGH':
        quality_indicators.append("📡 信号质量: 优秀")
    
    if quality_indicators:
        print("🏆 质量评估:")
        for indicator in quality_indicators:
            print(f"   {indicator}")
    
    # 数据和模型统计
    if 'data_loading' in results:
        data_info = results['data_loading']
        print(f"\n📈 数据概览:")
        print(f"   成功加载: {data_info.get('securities_loaded', 0)}只股票")
        print(f"   时间区间: {data_info.get('date_range', 'Unknown')}")
    
    if 'risk_model' in results:
        risk_info = results['risk_model']
        print(f"\n🛡️  风险模型:")
        print(f"   模型R²: {risk_info.get('model_r2', 0):.3f}")
        print(f"   因子数量: {risk_info.get('factor_count', 0)}")
        print(f"   覆盖率: {risk_info.get('coverage', 0):.2%}")
    
    if 'market_regime' in results:
        regime_info = results['market_regime']
        print(f"\n🌊 市场状态:")
        print(f"   当前状态: {regime_info.get('regime_name', 'Unknown')}")
        print(f"   置信度: {regime_info.get('probability', 0):.2%}")
        
        chars = regime_info.get('characteristics', {})
        if chars:
            print(f"   波动率: {chars.get('volatility', 0):.3f}")
            print(f"   趋势: {chars.get('trend', 0):.3f}")
    
    if 'alpha_signals' in results:
        signal_info = results['alpha_signals']
        print(f"\n🎯 Alpha信号:")
        print(f"   总信号数: {signal_info.get('total_signals', 0)}")
        print(f"   有效信号: {signal_info.get('active_signals', 0)}")
        
        strength = signal_info.get('signal_strength', {})
        if strength:
            print(f"   信号强度: {strength.get('mean', 0):.3f} ± {strength.get('std', 0):.3f}")
    
    # 投资组合结果
    if 'portfolio' in results and results['portfolio'].get('success', False):
        port_info = results['portfolio']['metrics']
        print(f"\n💼 投资组合:")
        print(f"   预期收益: {port_info.get('expected_return', 0):.3%}")
        print(f"   信息比率: {port_info.get('information_ratio', 0):.3f}")
        print(f"   夏普比率: {port_info.get('sharpe_ratio', 0):.3f}")
        print(f"   换手率: {port_info.get('turnover', 0):.2%}")
        print(f"   集中度HHI: {port_info.get('concentration_hhi', 0):.4f}")
        
        # 风险归因
        risk_attr = results['portfolio'].get('risk_attribution', {})
        if risk_attr:
            print(f"\n📊 风险归因 (前3因子):")
            sorted_risks = sorted(risk_attr.items(), key=lambda x: abs(x[1]), reverse=True)
            for factor, contribution in sorted_risks[:3]:
                print(f"   {factor}: {contribution:.4f}")
    
    # 投资建议
    if 'recommendations' in results and results['recommendations']:
        recommendations = results['recommendations']
        print(f"\n💡 投资建议 (Top {min(top_n, len(recommendations))}):")
        print(f"{'排名':<4} {'股票':<6} {'权重':<8} {'信号':<8} {'理由':<30}")
        print("-" * 60)
        
        for rec in recommendations[:top_n]:
            print(f"{rec['rank']:<4} {rec['ticker']:<6} "
                  f"{rec['weight']:.3f}  {rec['signal_strength']:.3f}  "
                  f"{rec['recommendation_reason'][:28]:<30}")
    
    # 输出文件
    if 'output_file' in results:
        print(f"\n📁 详细结果已保存至: {results['output_file']}")
    
    print("="*60)
    print("✨ Professional Quantitative Analysis Complete!")

def display_ultra_results(results: dict):
    """显示Ultra Enhanced结果"""
    print("\n" + "="*60)
    print("🚀 ULTRA ENHANCED BMA ANALYSIS RESULTS")
    print("="*60)
    
    if results.get('success', False):
        print("✅ Ultra Enhanced分析成功完成")
        print(f"⏱️  总耗时: {results.get('total_time', 0):.1f}秒")
        
        if 'recommendations' in results:
            recommendations = results['recommendations']
            print(f"\n💎 Ultra Enhanced投资建议:")
            for rec in recommendations[:5]:
                print(f"  {rec['rank']}. {rec['ticker']}: 权重{rec['weight']:.3f} "
                      f"(信号强度: {rec['prediction_signal']:.3f})")
        
        if 'result_file' in results:
            print(f"\n📁 结果文件: {results['result_file']}")
    else:
        print(f"❌ Ultra Enhanced分析失败: {results.get('error', '未知错误')}")

def display_original_results(recommendations: list):
    """显示原始版本结果"""
    print("\n" + "="*60)
    print("📊 ORIGINAL BMA ENHANCED RESULTS")
    print("="*60)
    
    if recommendations:
        print(f"✅ BMA Enhanced分析完成")
        print(f"\n📈 投资建议:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec.get('ticker', 'N/A')}: "
                  f"评分 {rec.get('ml_score', 0):.3f}")
    else:
        print("❌ 未生成有效推荐")

if __name__ == "__main__":
    exit(main())
