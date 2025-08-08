#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证器 - 确保BMA和LSTM模型返回正确的JSON和Excel格式
每周一自动运行，结合两个模型的top10股票
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import schedule
import time
import threading

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 文件路径配置
        self.result_dir = Path("result")
        self.bma_pattern = "bma_quantitative_analysis_*.xlsx"
        self.lstm_pattern = "*lstm_analysis_*.xlsx"
        
        # 输出路径
        self.output_dir = Path("model_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 验证标准
        self.required_bma_columns = ['ticker', 'rating', 'score', 'recommendation']
        self.required_lstm_columns = ['ticker', 'rating', 'weighted_prediction', 'confidence_score']
        
        # 统计信息
        self.validation_stats = {
            'bma': {'found': False, 'valid': False, 'top10': []},
            'lstm': {'found': False, 'valid': False, 'top10': []},
            'combined_top10': [],
            'last_validation': None
        }
    
    def find_latest_model_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """查找最新的模型文件"""
        try:
            # 查找BMA文件
            bma_files = list(self.result_dir.glob(self.bma_pattern))
            latest_bma = max(bma_files, key=os.path.getmtime) if bma_files else None
            
            # 查找LSTM文件
            lstm_files = list(self.result_dir.glob(self.lstm_pattern))
            latest_lstm = max(lstm_files, key=os.path.getmtime) if lstm_files else None
            
            if latest_bma:
                self.logger.info(f"✅ 找到BMA文件: {latest_bma}")
            else:
                self.logger.warning("⚠️ 未找到BMA文件")
            
            if latest_lstm:
                self.logger.info(f"✅ 找到LSTM文件: {latest_lstm}")
            else:
                self.logger.warning("⚠️ 未找到LSTM文件")
            
            return latest_bma, latest_lstm
            
        except Exception as e:
            self.logger.error(f"❌ 查找模型文件失败: {e}")
            return None, None
    
    def validate_bma_file(self, file_path: Path) -> Dict:
        """验证BMA文件格式"""
        validation_result = {
            'valid': False,
            'file_path': str(file_path),
            'file_size': 0,
            'sheets': [],
            'data_rows': 0,
            'top10': [],
            'errors': []
        }
        
        try:
            if not file_path.exists():
                validation_result['errors'].append("文件不存在")
                return validation_result
            
            validation_result['file_size'] = file_path.stat().st_size
            
            # 读取Excel文件
            excel_file = pd.ExcelFile(file_path)
            validation_result['sheets'] = excel_file.sheet_names
            
            self.logger.info(f"📊 BMA文件包含工作表: {validation_result['sheets']}")
            
            # 查找主要数据工作表
            main_sheet = None
            for sheet_name in excel_file.sheet_names:
                if any(keyword in sheet_name.lower() for keyword in ['analysis', 'result', 'recommend', 'main']):
                    main_sheet = sheet_name
                    break
            
            if not main_sheet:
                main_sheet = excel_file.sheet_names[0]  # 使用第一个工作表
            
            # 读取数据
            df = pd.read_excel(file_path, sheet_name=main_sheet)
            validation_result['data_rows'] = len(df)
            
            self.logger.info(f"📈 BMA数据行数: {len(df)}")
            self.logger.info(f"📋 BMA列名: {list(df.columns)}")
            
            # 检查必要列
            missing_columns = []
            for col in self.required_bma_columns:
                # 模糊匹配列名
                found = False
                for df_col in df.columns:
                    if col.lower() in str(df_col).lower():
                        found = True
                        break
                if not found:
                    missing_columns.append(col)
            
            if missing_columns:
                validation_result['errors'].append(f"缺少必要列: {missing_columns}")
            
            # 尝试提取top10
            try:
                # 尝试不同的评分列名
                score_columns = ['score', 'rating', 'recommendation_score', 'bma_score']
                score_col = None
                
                for col in score_columns:
                    for df_col in df.columns:
                        if col.lower() in str(df_col).lower():
                            score_col = df_col
                            break
                    if score_col:
                        break
                
                if score_col:
                    # 排序并获取top10
                    top_df = df.nlargest(10, score_col)
                    
                    # 查找ticker列
                    ticker_col = None
                    for col in ['ticker', 'symbol', 'stock', 'code']:
                        for df_col in df.columns:
                            if col.lower() in str(df_col).lower():
                                ticker_col = df_col
                                break
                        if ticker_col:
                            break
                    
                    if ticker_col:
                        validation_result['top10'] = [
                            {
                                'ticker': row[ticker_col],
                                'score': row[score_col],
                                'rank': i + 1
                            }
                            for i, (_, row) in enumerate(top_df.iterrows())
                        ]
                        
                        self.logger.info(f"🎯 BMA Top10: {[item['ticker'] for item in validation_result['top10']]}")
                    else:
                        validation_result['errors'].append("找不到股票代码列")
                else:
                    validation_result['errors'].append("找不到评分列")
                    
            except Exception as e:
                validation_result['errors'].append(f"提取top10失败: {str(e)}")
            
            # 验证通过条件
            validation_result['valid'] = (
                len(validation_result['errors']) == 0 and
                validation_result['data_rows'] > 0 and
                len(validation_result['top10']) > 0
            )
            
            if validation_result['valid']:
                self.logger.info("✅ BMA文件验证通过")
            else:
                self.logger.warning(f"⚠️ BMA文件验证失败: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"读取文件异常: {str(e)}")
            self.logger.error(f"❌ BMA文件验证异常: {e}")
            return validation_result
    
    def validate_lstm_file(self, file_path: Path) -> Dict:
        """验证LSTM文件格式"""
        validation_result = {
            'valid': False,
            'file_path': str(file_path),
            'file_size': 0,
            'sheets': [],
            'data_rows': 0,
            'top10': [],
            'errors': []
        }
        
        try:
            if not file_path.exists():
                validation_result['errors'].append("文件不存在")
                return validation_result
            
            validation_result['file_size'] = file_path.stat().st_size
            
            # 读取Excel文件
            excel_file = pd.ExcelFile(file_path)
            validation_result['sheets'] = excel_file.sheet_names
            
            self.logger.info(f"📊 LSTM文件包含工作表: {validation_result['sheets']}")
            
            # 查找主要数据工作表
            main_sheet = None
            for sheet_name in excel_file.sheet_names:
                if any(keyword in sheet_name.lower() for keyword in ['prediction', 'result', 'analysis', 'main']):
                    main_sheet = sheet_name
                    break
            
            if not main_sheet:
                main_sheet = excel_file.sheet_names[0]
            
            # 读取数据
            df = pd.read_excel(file_path, sheet_name=main_sheet)
            validation_result['data_rows'] = len(df)
            
            self.logger.info(f"📈 LSTM数据行数: {len(df)}")
            self.logger.info(f"📋 LSTM列名: {list(df.columns)}")
            
            # 检查必要列
            missing_columns = []
            for col in self.required_lstm_columns:
                found = False
                for df_col in df.columns:
                    if col.lower() in str(df_col).lower():
                        found = True
                        break
                if not found:
                    missing_columns.append(col)
            
            if missing_columns:
                validation_result['errors'].append(f"缺少必要列: {missing_columns}")
            
            # 尝试提取top10
            try:
                # 尝试不同的预测列名
                pred_columns = ['weighted_prediction', 'prediction', 'confidence_score', 'lstm_score']
                pred_col = None
                
                for col in pred_columns:
                    for df_col in df.columns:
                        if col.lower() in str(df_col).lower():
                            pred_col = df_col
                            break
                    if pred_col:
                        break
                
                if pred_col:
                    # 排序并获取top10
                    top_df = df.nlargest(10, pred_col)
                    
                    # 查找ticker列
                    ticker_col = None
                    for col in ['ticker', 'symbol', 'stock', 'code']:
                        for df_col in df.columns:
                            if col.lower() in str(df_col).lower():
                                ticker_col = df_col
                                break
                        if ticker_col:
                            break
                    
                    if ticker_col:
                        validation_result['top10'] = [
                            {
                                'ticker': row[ticker_col],
                                'score': row[pred_col],
                                'rank': i + 1
                            }
                            for i, (_, row) in enumerate(top_df.iterrows())
                        ]
                        
                        self.logger.info(f"🎯 LSTM Top10: {[item['ticker'] for item in validation_result['top10']]}")
                    else:
                        validation_result['errors'].append("找不到股票代码列")
                else:
                    validation_result['errors'].append("找不到预测评分列")
                    
            except Exception as e:
                validation_result['errors'].append(f"提取top10失败: {str(e)}")
            
            # 验证通过条件
            validation_result['valid'] = (
                len(validation_result['errors']) == 0 and
                validation_result['data_rows'] > 0 and
                len(validation_result['top10']) > 0
            )
            
            if validation_result['valid']:
                self.logger.info("✅ LSTM文件验证通过")
            else:
                self.logger.warning(f"⚠️ LSTM文件验证失败: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"读取文件异常: {str(e)}")
            self.logger.error(f"❌ LSTM文件验证异常: {e}")
            return validation_result
    
    def combine_top10_recommendations(self, bma_top10: List[Dict], lstm_top10: List[Dict]) -> List[Dict]:
        """结合两个模型的top10推荐"""
        try:
            # 创建股票评分字典
            stock_scores = {}
            
            # 添加BMA评分 (权重0.5)
            for item in bma_top10:
                ticker = item['ticker']
                score = item['score']
                rank = item['rank']
                
                if ticker not in stock_scores:
                    stock_scores[ticker] = {'bma_score': 0, 'lstm_score': 0, 'bma_rank': 999, 'lstm_rank': 999}
                
                stock_scores[ticker]['bma_score'] = float(score) if score is not None else 0
                stock_scores[ticker]['bma_rank'] = rank
            
            # 添加LSTM评分 (权重0.5)
            for item in lstm_top10:
                ticker = item['ticker']
                score = item['score']
                rank = item['rank']
                
                if ticker not in stock_scores:
                    stock_scores[ticker] = {'bma_score': 0, 'lstm_score': 0, 'bma_rank': 999, 'lstm_rank': 999}
                
                stock_scores[ticker]['lstm_score'] = float(score) if score is not None else 0
                stock_scores[ticker]['lstm_rank'] = rank
            
            # 计算综合评分
            combined_scores = []
            for ticker, scores in stock_scores.items():
                # 综合评分 = 0.5 * BMA评分 + 0.5 * LSTM评分 - 排名惩罚
                bma_norm = scores['bma_score'] / max([s['bma_score'] for s in stock_scores.values()]) if max([s['bma_score'] for s in stock_scores.values()]) > 0 else 0
                lstm_norm = scores['lstm_score'] / max([s['lstm_score'] for s in stock_scores.values()]) if max([s['lstm_score'] for s in stock_scores.values()]) > 0 else 0
                
                # 排名加分 (排名越高加分越多)
                rank_bonus = (20 - scores['bma_rank']) / 20 + (20 - scores['lstm_rank']) / 20
                
                combined_score = (bma_norm * 0.5 + lstm_norm * 0.5) + rank_bonus * 0.1
                
                combined_scores.append({
                    'ticker': ticker,
                    'combined_score': combined_score,
                    'bma_score': scores['bma_score'],
                    'lstm_score': scores['lstm_score'],
                    'bma_rank': scores['bma_rank'],
                    'lstm_rank': scores['lstm_rank'],
                    'in_both': scores['bma_rank'] <= 10 and scores['lstm_rank'] <= 10
                })
            
            # 按综合评分排序，取top10
            combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            top10_combined = combined_scores[:10]
            
            # 添加最终排名
            for i, item in enumerate(top10_combined):
                item['final_rank'] = i + 1
            
            self.logger.info(f"🏆 综合Top10: {[item['ticker'] for item in top10_combined]}")
            
            return top10_combined
            
        except Exception as e:
            self.logger.error(f"❌ 结合top10失败: {e}")
            return []
    
    def save_combined_results(self, combined_top10: List[Dict]) -> bool:
        """保存综合结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存JSON格式
            json_file = self.output_dir / f"combined_top10_{timestamp}.json"
            json_data = {
                'generated_at': datetime.now().isoformat(),
                'model_combination': 'BMA + LSTM',
                'top10_stocks': combined_top10,
                'summary': {
                    'total_stocks': len(combined_top10),
                    'stocks_in_both_models': len([s for s in combined_top10 if s.get('in_both', False)]),
                    'bma_only': len([s for s in combined_top10 if s.get('bma_rank', 999) <= 10 and s.get('lstm_rank', 999) > 10]),
                    'lstm_only': len([s for s in combined_top10 if s.get('lstm_rank', 999) <= 10 and s.get('bma_rank', 999) > 10])
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # 保存Excel格式
            excel_file = self.output_dir / f"combined_top10_{timestamp}.xlsx"
            df = pd.DataFrame(combined_top10)
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Combined_Top10', index=False)
                
                # 添加汇总信息
                summary_df = pd.DataFrame([json_data['summary']])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 保存简单的股票列表
            symbols_file = self.output_dir / f"top10_symbols_{timestamp}.txt"
            with open(symbols_file, 'w', encoding='utf-8') as f:
                for item in combined_top10:
                    f.write(f"{item['ticker']}\n")
            
            self.logger.info(f"✅ 综合结果已保存:")
            self.logger.info(f"   JSON: {json_file}")
            self.logger.info(f"   Excel: {excel_file}")
            self.logger.info(f"   符号列表: {symbols_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 保存综合结果失败: {e}")
            return False
    
    def run_validation(self) -> Dict:
        """运行完整验证流程"""
        self.logger.info("🚀 开始模型验证...")
        
        # 查找最新文件
        bma_file, lstm_file = self.find_latest_model_files()
        
        # 验证BMA文件
        bma_result = None
        if bma_file:
            self.validation_stats['bma']['found'] = True
            bma_result = self.validate_bma_file(bma_file)
            self.validation_stats['bma']['valid'] = bma_result['valid']
            self.validation_stats['bma']['top10'] = bma_result['top10']
        
        # 验证LSTM文件
        lstm_result = None
        if lstm_file:
            self.validation_stats['lstm']['found'] = True
            lstm_result = self.validate_lstm_file(lstm_file)
            self.validation_stats['lstm']['valid'] = lstm_result['valid']
            self.validation_stats['lstm']['top10'] = lstm_result['top10']
        
        # 结合两个模型的结果
        if (bma_result and bma_result['valid'] and 
            lstm_result and lstm_result['valid']):
            
            combined_top10 = self.combine_top10_recommendations(
                bma_result['top10'], 
                lstm_result['top10']
            )
            
            if combined_top10:
                self.validation_stats['combined_top10'] = combined_top10
                self.save_combined_results(combined_top10)
        
        self.validation_stats['last_validation'] = datetime.now().isoformat()
        
        # 生成验证报告
        report = {
            'validation_time': self.validation_stats['last_validation'],
            'bma': {
                'file_found': self.validation_stats['bma']['found'],
                'validation_passed': self.validation_stats['bma']['valid'],
                'file_path': str(bma_file) if bma_file else None,
                'top10_count': len(self.validation_stats['bma']['top10']),
                'details': bma_result
            },
            'lstm': {
                'file_found': self.validation_stats['lstm']['found'],
                'validation_passed': self.validation_stats['lstm']['valid'],
                'file_path': str(lstm_file) if lstm_file else None,
                'top10_count': len(self.validation_stats['lstm']['top10']),
                'details': lstm_result
            },
            'combined': {
                'success': len(self.validation_stats['combined_top10']) > 0,
                'top10_count': len(self.validation_stats['combined_top10']),
                'recommendations': self.validation_stats['combined_top10']
            }
        }
        
        # 保存验证报告
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📊 验证报告已保存: {report_file}")
        
        # 打印汇总
        self.logger.info("📋 验证汇总:")
        self.logger.info(f"   BMA模型: {'✅' if report['bma']['validation_passed'] else '❌'}")
        self.logger.info(f"   LSTM模型: {'✅' if report['lstm']['validation_passed'] else '❌'}")
        self.logger.info(f"   综合推荐: {'✅' if report['combined']['success'] else '❌'}")
        
        if report['combined']['success']:
            top_symbols = [item['ticker'] for item in self.validation_stats['combined_top10'][:5]]
            self.logger.info(f"   Top5推荐: {top_symbols}")
        
        return report


class WeeklyScheduler:
    """每周一自动运行调度器"""
    
    def __init__(self, validator: ModelValidator):
        self.validator = validator
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.scheduler_thread = None
    
    def setup_schedule(self):
        """设置每周一运行的调度"""
        # 每周一上午9点运行
        schedule.every().monday.at("09:00").do(self._run_weekly_validation)
        
        # 也可以每天检查一次是否有新的模型文件
        schedule.every().day.at("10:00").do(self._check_for_new_models)
        
        self.logger.info("📅 已设置每周一自动验证调度")
    
    def _run_weekly_validation(self):
        """每周验证任务"""
        self.logger.info("📅 执行每周一模型验证...")
        try:
            report = self.validator.run_validation()
            
            # 可以在这里添加邮件通知等
            if report['combined']['success']:
                self.logger.info("✅ 每周验证成功完成")
            else:
                self.logger.warning("⚠️ 每周验证完成但存在问题")
                
        except Exception as e:
            self.logger.error(f"❌ 每周验证失败: {e}")
    
    def _check_for_new_models(self):
        """检查是否有新的模型文件"""
        try:
            bma_file, lstm_file = self.validator.find_latest_model_files()
            
            # 检查文件是否是今天生成的
            today = datetime.now().date()
            
            new_files = []
            if bma_file and datetime.fromtimestamp(bma_file.stat().st_mtime).date() == today:
                new_files.append(f"BMA: {bma_file.name}")
            
            if lstm_file and datetime.fromtimestamp(lstm_file.stat().st_mtime).date() == today:
                new_files.append(f"LSTM: {lstm_file.name}")
            
            if new_files:
                self.logger.info(f"🆕 发现新模型文件: {', '.join(new_files)}")
                # 自动运行验证
                self.validator.run_validation()
                
        except Exception as e:
            self.logger.error(f"❌ 检查新模型文件失败: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        if self.is_running:
            return
        
        self.setup_schedule()
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("🚀 调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        schedule.clear()
        self.logger.info("🛑 调度器已停止")


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"model_validator_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 模型验证器启动")
    
    try:
        # 创建验证器
        validator = ModelValidator()
        
        # 运行立即验证
        logger.info("📊 执行立即验证...")
        report = validator.run_validation()
        
        # 创建调度器
        scheduler = WeeklyScheduler(validator)
        scheduler.start_scheduler()
        
        logger.info("✅ 模型验证器运行中...")
        logger.info("   - 立即验证已完成")
        logger.info("   - 每周一09:00自动验证")
        logger.info("   - 每天10:00检查新模型文件")
        
        # 保持运行
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("👋 用户中断")
            scheduler.stop_scheduler()
    
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)