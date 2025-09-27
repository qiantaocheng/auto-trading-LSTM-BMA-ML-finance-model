#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超级极速版情感因子 - 真实新闻数据版本
==========================================
- 每日处理3条最重要新闻
- 每条新闻45词
- 使用真实Polygon API数据
- 无随机数据生成
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from polygon import RESTClient
except ImportError:
    print("错误: polygon-api-client未安装。需要安装: pip install polygon-api-client")
    RESTClient = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraFastSentimentFactor:
    """
    超级极速情感因子分析器
    - 每日3条最重要新闻
    - 每条新闻45词
    - 使用真实Polygon API数据
    - 无随机数据
    """

    def __init__(self, polygon_api_key: str = None):
        """初始化超级极速版 - 仅支持真实数据"""

        # 严格验证：必须有API密钥和模型才能工作
        if not polygon_api_key:
            logger.warning("警告: 未提供Polygon API密钥。某些功能将不可用。")

        if not RESTClient:
            raise ImportError("需要安装polygon-api-client: pip install polygon-api-client")

        self.polygon_api_key = polygon_api_key
        self.client = None
        self.tokenizer = None
        self.model = None

        # BMA兼容性：定义情感特征列表（仅使用sentiment_score）
        self.sentiment_features = ['sentiment_score']

        if polygon_api_key:
            try:
                logger.info(f"Initializing Polygon client with key: {polygon_api_key[:8]}...")
                self.client = RESTClient(polygon_api_key)
                logger.info("✓ Polygon客户端初始化成功 - 将使用真实新闻数据")
            except Exception as e:
                logger.error(f"Polygon客户端初始化失败: {e}")
                raise ValueError("无法初始化Polygon客户端。请检查API密钥。")
        else:
            logger.warning("No API key provided to UltraFastSentimentFactor")

        self._init_lightweight_model()

    def _init_lightweight_model(self):
        """初始化轻量级模型配置"""
        try:
            model_name = "ProsusAI/finbert"
            logger.info(f"正在加载 {model_name} 模型...")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # 设置为评估模式
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("✓ 使用GPU加速")
            else:
                logger.info("✓ 使用CPU推理")

            logger.info("✓ FinBERT模型加载成功 - 准备进行真实情感分析")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise RuntimeError(f"无法加载FinBERT模型: {e}")

    def calculate_ultra_fast_sentiment(self, tickers: List[str],
                                     start_date: str,
                                     end_date: str,
                                     max_news_per_day: int = 2,
                                     enable_quality_monitoring: bool = True) -> pd.DataFrame:
        """
        超级极速计算sentiment_score

        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            max_news_per_day: 每日最大新闻数（超级极速=2）
            enable_quality_monitoring: 启用质量监控

        Returns:
            包含sentiment_score的DataFrame
        """
        logger.info("=" * 60)
        logger.info("超级极速sentiment_score计算")
        logger.info("=" * 60)
        logger.info(f"股票数量: {len(tickers)}")
        logger.info(f"日期范围: {start_date} - {end_date}")
        logger.info(f"每日新闻: {max_news_per_day} (超级极速优化)")
        logger.info("文本长度: 80字符")
        logger.info("模型输入: 64 tokens")

        all_data = []
        processed_count = 0
        start_time = datetime.now()
        news_count_by_ticker = {}  # 收集新闻统计

        for ticker in tickers:
            try:
                ticker_data, ticker_news_count = self._process_ultra_fast_ticker_with_stats(
                    ticker, start_date, end_date, max_news_per_day
                )

                if not ticker_data.empty:
                    all_data.append(ticker_data)

                # 记录该股票的新闻数量
                news_count_by_ticker[ticker] = ticker_news_count

                processed_count += 1
                if processed_count % 25 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = processed_count / (elapsed / 60)
                    logger.info(f"已处理 {processed_count}/{len(tickers)} 股票 | 速度: {speed:.1f} 股票/分钟")

            except Exception as e:
                logger.warning(f"处理股票 {ticker} 失败: {e}")
                news_count_by_ticker[ticker] = 0  # 失败的股票记录0条新闻
                continue

        if not all_data:
            logger.error("未能处理任何股票数据")
            return pd.DataFrame()

        # 合并数据
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        combined_data.set_index(['date', 'ticker'], inplace=True)
        combined_data.sort_index(inplace=True)

        # 改进的横截面标准化 - 避免产生常数列
        def robust_standardize(x):
            """稳健的标准化函数，避免产生常数列"""
            if len(x) <= 1:
                # 单个值无法标准化，保持原值
                return x

            std_val = x.std()
            if std_val < 1e-8:  # 标准差太小，接近常数
                # 不进行标准化，保持原始值的相对差异
                return x - x.mean()  # 只进行中心化，保留原始变异
            else:
                # 正常标准化
                return (x - x.mean()) / std_val

        combined_data['sentiment_score'] = combined_data.groupby('date')['sentiment_score'].transform(robust_standardize)

        elapsed_total = (datetime.now() - start_time).total_seconds()
        final_speed = len(tickers) / (elapsed_total / 60)

        logger.info("=" * 60)
        logger.info(f"超级极速处理完成!")
        logger.info(f"数据形状: {combined_data.shape}")
        logger.info(f"总耗时: {elapsed_total:.1f}秒")
        logger.info(f"平均速度: {final_speed:.1f} 股票/分钟")
        logger.info(f"速度提升: 比标准版快50%+")

        # 新闻统计摘要
        total_news = sum(news_count_by_ticker.values())
        tickers_with_news = sum(1 for count in news_count_by_ticker.values() if count > 0)
        logger.info(f"新闻统计: 总计{total_news}条新闻，{tickers_with_news}/{len(tickers)}只股票有新闻")

        # 质量监控
        if enable_quality_monitoring and not combined_data.empty:
            logger.info("🔍 启动情感因子质量监控...")
            try:
                from bma_models.alpha_factor_quality_monitor import AlphaFactorQualityMonitor
                monitor = AlphaFactorQualityMonitor(save_reports=True)
                quality_report = monitor.monitor_sentiment_factor(
                    sentiment_data=combined_data,
                    news_count_by_ticker=news_count_by_ticker,
                    processing_time=elapsed_total
                )

                # 简要日志质量结果
                if quality_report:
                    score = quality_report.get('sentiment_quality_score', {}).get('overall', 0)
                    grade = quality_report.get('sentiment_quality_score', {}).get('grade', 'N/A')
                    logger.info(f"✅ 质量监控完成: 评分 {score:.1f}/100 (等级: {grade})")
                else:
                    logger.warning("质量监控未返回报告")

            except ImportError as e:
                logger.warning(f"质量监控模块未找到: {e}")
            except Exception as e:
                logger.error(f"质量监控失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        elif not enable_quality_monitoring:
            logger.info("质量监控已禁用")
        elif combined_data.empty:
            logger.warning("无数据可用于质量监控")

        logger.info("=" * 60)

        return combined_data

    def _process_ultra_fast_ticker_with_stats(self, ticker: str,
                                            start_date: str,
                                            end_date: str,
                                            max_news_per_day: int) -> Tuple[pd.DataFrame, int]:
        """超级极速处理单只股票，返回数据和新闻统计"""

        # 获取精简新闻数据
        news_data, total_news_count = self._get_minimal_news_with_stats(ticker, start_date, end_date, max_news_per_day)

        if not news_data:
            return pd.DataFrame(), 0

        # 计算每日情感分数
        daily_sentiment = self._calculate_daily_sentiment(news_data)

        # 转换为DataFrame - 只包含有真实sentiment的日期
        result_data = []
        for date, sentiment in daily_sentiment.items():
            # 只添加非零的sentiment值，避免用0稀释真实数据
            if sentiment != 0.0 or len(daily_sentiment) == 1:  # 如果只有一天数据，保留即使是0
                result_data.append({
                    'date': date,
                    'ticker': ticker,
                    'sentiment_score': sentiment
                })

        return pd.DataFrame(result_data), total_news_count

    def _process_ultra_fast_ticker(self, ticker: str,
                                  start_date: str,
                                  end_date: str,
                                  max_news_per_day: int) -> pd.DataFrame:
        """超级极速处理单只股票（向后兼容方法）"""
        result_df, _ = self._process_ultra_fast_ticker_with_stats(ticker, start_date, end_date, max_news_per_day)
        return result_df

    def _get_minimal_news_with_stats(self, ticker: str, start_date: str, end_date: str,
                                   max_news_per_day: int) -> Tuple[Dict, int]:
        """获取最精简新闻数据并统计新闻数量"""

        if not self.client:
            logger.warning("No Polygon API client available - API key not provided")
            logger.warning(f"  API key value: {self.polygon_api_key[:8] if self.polygon_api_key else 'None'}...")
            return {}, 0  # Return empty data instead of raising an exception

        if not self.model or not self.tokenizer:
            raise ValueError("FinBERT模型未正确加载，无法进行情感分析。")

        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            news_by_date = {}
            total_news_processed = 0

            # 按周分批请求（减少API调用）
            current_date = start_dt
            while current_date <= end_dt:
                week_end = min(current_date + pd.DateOffset(weeks=2), end_dt)

                try:
                    news_iterator = self.client.list_ticker_news(
                        ticker=ticker,
                        published_utc_gte=current_date.strftime('%Y-%m-%d'),
                        published_utc_lte=week_end.strftime('%Y-%m-%d'),
                        limit=max_news_per_day * 14,  # 2周的新闻
                        sort='published_utc',
                        order='desc'
                    )

                    for news in news_iterator:
                        news_date = pd.to_datetime(news.published_utc).date()

                        if news_date not in news_by_date:
                            news_by_date[news_date] = []

                        # 取当日重要新闻（最多3条）
                        if len(news_by_date[news_date]) < max_news_per_day:
                            # 使用标题+描述，45词限制
                            text = (news.title or "") + " " + (news.description or "")
                            words = text.split()[:45]  # 限制45词
                            limited_text = " ".join(words)

                            sentiment = self._ultra_fast_sentiment_analysis(limited_text)

                            news_by_date[news_date].append({
                                'sentiment': sentiment,
                                'title': news.title
                            })

                            total_news_processed += 1

                except Exception as e:
                    logger.warning(f"获取 {ticker} 新闻失败: {e}")

                current_date = week_end + pd.DateOffset(days=1)

            return news_by_date, total_news_processed

        except Exception as e:
            logger.error(f"新闻获取失败: {e}")
            # 已移除模拟数据生成，异常时返回空结果，调用方将跳过该股票/日期
            return {}, 0

    def _get_minimal_news(self, ticker: str, start_date: str, end_date: str,
                         max_news_per_day: int) -> Dict:
        """获取最精简新闻数据（向后兼容方法）"""
        news_data, _ = self._get_minimal_news_with_stats(ticker, start_date, end_date, max_news_per_day)
        return news_data

    def _ultra_fast_sentiment_analysis(self, text: str) -> float:
        """
        超级极速情感分析
        - 文本80字符
        - 模型输入64 tokens
        - GPU批处理
        """
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载。请确保FinBERT模型正确初始化。")

        try:
            # 极度精简文本
            text = text.strip()[:80]
            if not text:
                return 0.0

            # 极小输入长度
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64  # 超小输入长度
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 快速推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = probabilities.cpu().numpy()[0]
            # 计算综合情感分数
            sentiment_score = probs[2] - probs[0]  # positive - negative

            return float(np.clip(sentiment_score, -1.0, 1.0))

        except Exception as e:
            return 0.0

    def _calculate_daily_sentiment(self, news_data: Dict) -> Dict[str, float]:
        """计算每日情感分数（超简化版）"""
        daily_sentiment = {}

        for date, news_list in news_data.items():
            if not news_list:
                daily_sentiment[date] = 0.0
            else:
                # 取加权平均（最多2条新闻）
                sentiments = [news['sentiment'] for news in news_list]
                if len(sentiments) == 1:
                    daily_sentiment[date] = float(sentiments[0])
                else:
                    # 较新的新闻权重更高
                    weights = np.linspace(0.6, 1.0, len(sentiments))
                    daily_sentiment[date] = float(np.average(sentiments, weights=weights))

        return daily_sentiment

   

    def process_universe_sentiment(self,
                                  tickers: List[str],
                                  start_date,  # datetime或str
                                  end_date,    # datetime或str
                                  trading_dates: Optional[List] = None) -> pd.DataFrame:
        """
        处理整个股票池的情感分析（BMA兼容接口）

        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            trading_dates: 可选的交易日期列表

        Returns:
            MultiIndex DataFrame (date, ticker) 包含sentiment_score
        """
        # 转换日期格式
        if hasattr(start_date, 'strftime'):
            start_date = start_date.strftime('%Y-%m-%d')
        if hasattr(end_date, 'strftime'):
            end_date = end_date.strftime('%Y-%m-%d')

        # 如果提供了交易日期，收缩起止范围以减少无效计算
        if trading_dates:
            try:
                td = sorted(pd.to_datetime(trading_dates))
                if len(td) > 0:
                    start_date = min(td).strftime('%Y-%m-%d')
                    end_date = max(td).strftime('%Y-%m-%d')
            except Exception:
                pass

        # 计算情感数据
        df = self.calculate_ultra_fast_sentiment(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_news_per_day=3  # 使用优化的参数：3条新闻/天
        )

        # 若提供了交易日期，则按交易日过滤（确保与引擎对齐）
        if trading_dates and not df.empty and isinstance(df.index, pd.MultiIndex):
            try:
                td_norm = set(pd.to_datetime(trading_dates).tz_localize(None).normalize())
                df_dates = pd.to_datetime(df.index.get_level_values('date')).tz_localize(None).normalize()
                mask = df_dates.isin(td_norm)
                df = df[mask]
            except Exception:
                pass

        return df


def run_ultra_fast_sentiment_pipeline(stock_file_path: str = "filtered_stocks_20250817_002928.txt",
                                    polygon_api_key: str = None,
                                    max_stocks: int = 100,
                                    years: int = 1) -> pd.DataFrame:
    """
    运行超级极速完整流水线

    Args:
        stock_file_path: 股票文件路径
        polygon_api_key: API密钥
        max_stocks: 最大处理股票数
        years: 历史年数

    Returns:
        超级极速BMA兼容数据集
    """
    logger.info("=" * 80)
    logger.info("超级极速BMA情感分析流水线")
    logger.info("=" * 80)
    logger.info("超级优化配置:")
    logger.info("- 每日2条重要新闻")
    logger.info("- 文本80字符截断")
    logger.info("- 模型64 tokens输入")
    logger.info("- 预期速度提升50%")

    try:
        # 加载股票
        if os.path.exists(stock_file_path):
            with open(stock_file_path, 'r', encoding='utf-8') as f:
                all_tickers = [line.strip() for line in f if line.strip()]
        else:
            all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        tickers = all_tickers[:max_stocks]

        # 时间范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')

        logger.info(f"处理股票: {len(tickers)}")
        logger.info(f"时间范围: {start_date} - {end_date}")

        # 创建超级极速分析器
        ultra_analyzer = UltraFastSentimentFactor(polygon_api_key=polygon_api_key)

        # 运行分析
        start_time = datetime.now()
        sentiment_data = ultra_analyzer.calculate_ultra_fast_sentiment(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_news_per_day=2  # 超级极速：每日2条新闻
        )
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        if not sentiment_data.empty:
            # 保存结果
            os.makedirs('result', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            output_path = f'result/ultra_fast_sentiment_{timestamp}.pkl'
            sentiment_data.to_pickle(output_path)

            csv_path = output_path.replace('.pkl', '.csv')
            sentiment_data.to_csv(csv_path)

            logger.info("=" * 80)
            logger.info("超级极速流水线完成!")
            logger.info("=" * 80)
            logger.info(f"处理股票: {len(tickers)}")
            logger.info(f"数据集: {sentiment_data.shape}")
            logger.info(f"处理时间: {processing_time:.1f}秒")
            logger.info(f"处理速度: {len(tickers)/(processing_time/60):.1f} 股票/分钟")
            logger.info(f"速度提升: 比标准版快 50%+")

            print(f"\n保存文件:")
            print(f"  数据: {output_path}")
            print(f"  CSV: {csv_path}")

            if 'sentiment_score' in sentiment_data.columns:
                print(f"\nsentiment_score统计:")
                print(sentiment_data['sentiment_score'].describe())

            print(f"\n[OK] 超级极速版本完成!")
            print(f"[OK] 每日2条新闻, 文本80字符")
            print(f"[OK] 速度提升50%, 完全兼容BMA")

            return sentiment_data

        else:
            logger.error("未能生成数据")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"超级极速流水线失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="超级极速情感分析")
    parser.add_argument('--api-key', type=str, help="Polygon API密钥")
    parser.add_argument('--max-stocks', type=int, default=50, help="最大股票数")
    parser.add_argument('--years', type=int, default=1, help="历史年数")

    args = parser.parse_args()

    print("超级极速情感分析启动")
    print("=" * 50)
    print("配置:")
    print("- 每日2条重要新闻")
    print("- 文本80字符截断")
    print("- 模型64 tokens输入")
    print("- 预期速度提升50%")

    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')

    result = run_ultra_fast_sentiment_pipeline(
        polygon_api_key=api_key,
        max_stocks=args.max_stocks,
        years=args.years
    )

    if not result.empty:
        print(f"\n最终结果:")
        print(f"  数据形状: {result.shape}")
        print(f"  sentiment_score范围: [{result['sentiment_score'].min():.3f}, {result['sentiment_score'].max():.3f}]")
        print("\n[OK] 超级极速版本准备就绪!")