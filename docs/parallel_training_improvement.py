#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行训练改进方案 - Ridge和LambdaRank并行执行
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class ParallelStackerTrainer:
    """
    并行训练Ridge和LambdaRank Stacker

    核心改进：
    1. 真正的并行执行，减少训练时间
    2. 统一的异常处理
    3. 性能监控和时间统计
    """

    def train_stackers_parallel(self,
                               ridge_stacker,
                               lambda_rank_stacker,
                               stacker_data: pd.DataFrame,
                               use_rank_aware_blending: bool = True):
        """
        并行训练两个Stacker模型

        Args:
            ridge_stacker: Ridge模型实例
            lambda_rank_stacker: LambdaRank模型实例
            stacker_data: 训练数据
            use_rank_aware_blending: 是否使用Rank-aware融合

        Returns:
            dict: 训练结果和统计信息
        """

        results = {
            'ridge_success': False,
            'lambda_success': False,
            'ridge_time': 0,
            'lambda_time': 0,
            'total_time': 0,
            'time_saved': 0
        }

        start_time = time.time()

        if not use_rank_aware_blending:
            # 如果不使用rank-aware blending，只训练Ridge
            logger.info("🎯 单独训练Ridge Stacker（无并行）")
            try:
                ridge_start = time.time()
                ridge_stacker.fit(stacker_data, max_train_to_today=True)
                results['ridge_time'] = time.time() - ridge_start
                results['ridge_success'] = True
                logger.info(f"✅ Ridge训练完成，耗时: {results['ridge_time']:.2f}秒")
            except Exception as e:
                logger.error(f"❌ Ridge训练失败: {e}")

            results['total_time'] = time.time() - start_time
            return results

        # 并行训练Ridge和LambdaRank
        logger.info("🚀 开始并行训练Ridge和LambdaRank...")

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="Stacker") as executor:
            # 提交两个训练任务
            futures = {}

            # Ridge训练任务
            ridge_future = executor.submit(
                self._train_ridge_wrapper,
                ridge_stacker,
                stacker_data
            )
            futures[ridge_future] = 'ridge'

            # LambdaRank训练任务
            lambda_future = executor.submit(
                self._train_lambda_wrapper,
                lambda_rank_stacker,
                stacker_data
            )
            futures[lambda_future] = 'lambda'

            # 等待完成并收集结果
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result, elapsed_time = future.result(timeout=600)  # 10分钟超时

                    if model_name == 'ridge':
                        results['ridge_success'] = result
                        results['ridge_time'] = elapsed_time
                        logger.info(f"✅ Ridge训练完成，耗时: {elapsed_time:.2f}秒")
                    else:
                        results['lambda_success'] = result
                        results['lambda_time'] = elapsed_time
                        logger.info(f"✅ LambdaRank训练完成，耗时: {elapsed_time:.2f}秒")

                except TimeoutError:
                    logger.error(f"❌ {model_name}训练超时")
                except Exception as e:
                    logger.error(f"❌ {model_name}训练失败: {e}")

        # 计算统计信息
        results['total_time'] = time.time() - start_time
        sequential_time = results['ridge_time'] + results['lambda_time']
        results['time_saved'] = sequential_time - results['total_time']

        # 打印性能报告
        logger.info("=" * 50)
        logger.info("📊 并行训练性能报告:")
        logger.info(f"   Ridge训练时间: {results['ridge_time']:.2f}秒")
        logger.info(f"   LambdaRank训练时间: {results['lambda_time']:.2f}秒")
        logger.info(f"   总耗时: {results['total_time']:.2f}秒")
        logger.info(f"   节省时间: {results['time_saved']:.2f}秒")
        logger.info(f"   加速比: {sequential_time/results['total_time']:.2f}x")
        logger.info("=" * 50)

        return results

    def _train_ridge_wrapper(self, ridge_stacker, data):
        """Ridge训练包装器"""
        start = time.time()
        try:
            ridge_stacker.fit(data, max_train_to_today=True)
            return True, time.time() - start
        except Exception as e:
            logger.error(f"Ridge训练异常: {e}")
            return False, time.time() - start

    def _train_lambda_wrapper(self, lambda_rank_stacker, data):
        """LambdaRank训练包装器"""
        start = time.time()
        try:
            lambda_rank_stacker.fit(data)
            return True, time.time() - start
        except Exception as e:
            logger.error(f"LambdaRank训练异常: {e}")
            return False, time.time() - start


class ImprovedPredictionPipeline:
    """
    改进的预测管道 - 并行化Ridge和LambdaRank预测
    """

    def parallel_predict(self,
                         ridge_stacker,
                         lambda_rank_stacker,
                         rank_aware_blender,
                         prediction_data: pd.DataFrame):
        """
        并行生成Ridge和LambdaRank预测，然后融合

        Args:
            ridge_stacker: 已训练的Ridge模型
            lambda_rank_stacker: 已训练的LambdaRank模型
            rank_aware_blender: Rank-aware融合器
            prediction_data: 预测数据

        Returns:
            pd.DataFrame: 融合后的预测结果
        """

        logger.info("🔮 开始并行预测...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个预测任务
            ridge_future = executor.submit(ridge_stacker.predict, prediction_data)
            lambda_future = executor.submit(lambda_rank_stacker.predict, prediction_data)

            # 等待两个预测完成
            ridge_predictions = ridge_future.result()
            lambda_predictions = lambda_future.result()

        # 使用Rank-aware Blender融合
        blended_results = rank_aware_blender.blend_predictions(
            ridge_predictions,
            lambda_predictions
        )

        elapsed = time.time() - start_time
        logger.info(f"✅ 并行预测完成，耗时: {elapsed:.2f}秒")

        return blended_results


# 使用示例
if __name__ == "__main__":
    # 初始化并行训练器
    trainer = ParallelStackerTrainer()

    # 模拟数据和模型
    # stacker_data = pd.DataFrame(...)  # 你的训练数据
    # ridge_stacker = RidgeStacker(...)
    # lambda_rank_stacker = LambdaRankStacker(...)

    # 执行并行训练
    # results = trainer.train_stackers_parallel(
    #     ridge_stacker,
    #     lambda_rank_stacker,
    #     stacker_data,
    #     use_rank_aware_blending=True
    # )

    print("并行训练改进方案已准备就绪")