"""
回归平均学习引擎 v8.0 - Mean Reversion Learning Engine

基于用户观察：
在足够大的样本（365期、600期）中，所有号码、大小、单双、波色都趋向于平均水平。
这个引擎利用"回归平均"（Regression to the Mean）现象进行预测。

核心思想：
1. 偏离平均较多的属性，更可能在未来"回归"到平均
2. 长期来看，随机过程会趋向于理论概率
3. 当前偏离越大，回归压力越大

⚠️ 重要警告：
虽然回归平均是真实统计现象，但它不能预测下一期结果！
这仍然是过拟合，对实际投注无效。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter


class MeanReversionAnalyzer:
    """回归平均分析器"""
    
    @staticmethod
    def analyze_number_deviation(data: pd.DataFrame, analysis_periods: int = 365) -> Dict:
        """
        分析号码相对于理论平均的偏离
        
        理论：每个号码在足够大样本中应该出现约 analysis_periods / 49 次
        """
        recent_data = data['特码'].iloc[-analysis_periods:]
        
        # 理论期望次数
        expected_count = analysis_periods / 49
        
        # 实际出现次数
        actual_counts = Counter(recent_data)
        
        # 计算偏离度
        deviations = {}
        for num in range(1, 50):
            actual = actual_counts.get(num, 0)
            deviation = actual - expected_count
            deviation_ratio = deviation / expected_count  # 相对偏离
            
            deviations[num] = {
                'actual_count': actual,
                'expected_count': expected_count,
                'deviation': deviation,
                'deviation_ratio': deviation_ratio,
                'reversion_pressure': -deviation_ratio  # 负偏离越大，回归压力越大
            }
        
        return deviations
    
    @staticmethod
    def analyze_attribute_deviation(data: pd.DataFrame, analysis_periods: int = 365) -> Dict:
        """
        分析大小、单双、波色的偏离
        
        理论比例：
        - 大小：各50%
        - 单双：各50%
        - 波色：各33.33%
        """
        recent_data = data.iloc[-analysis_periods:]
        
        # 大小统计
        big_count = (recent_data['特码'] >= 25).sum()
        small_count = (recent_data['特码'] < 25).sum()
        big_ratio = big_count / analysis_periods
        small_ratio = small_count / analysis_periods
        
        # 单双统计
        odd_count = (recent_data['特码'] % 2 == 1).sum()
        even_count = (recent_data['特码'] % 2 == 0).sum()
        odd_ratio = odd_count / analysis_periods
        even_ratio = even_count / analysis_periods
        
        # 波色统计
        red_nums = {1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46}
        blue_nums = {3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48}
        green_nums = {5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49}
        
        red_count = recent_data['特码'].apply(lambda x: x in red_nums).sum()
        blue_count = recent_data['特码'].apply(lambda x: x in blue_nums).sum()
        green_count = recent_data['特码'].apply(lambda x: x in green_nums).sum()
        
        red_ratio = red_count / analysis_periods
        blue_ratio = blue_count / analysis_periods
        green_ratio = green_count / analysis_periods
        
        return {
            'big_small': {
                'big': {'count': big_count, 'ratio': big_ratio, 'deviation': big_ratio - 0.5, 'reversion_pressure': 0.5 - big_ratio},
                'small': {'count': small_count, 'ratio': small_ratio, 'deviation': small_ratio - 0.5, 'reversion_pressure': 0.5 - small_ratio}
            },
            'odd_even': {
                'odd': {'count': odd_count, 'ratio': odd_ratio, 'deviation': odd_ratio - 0.5, 'reversion_pressure': 0.5 - odd_ratio},
                'even': {'count': even_count, 'ratio': even_ratio, 'deviation': even_ratio - 0.5, 'reversion_pressure': 0.5 - even_ratio}
            },
            'color': {
                'red': {'count': red_count, 'ratio': red_ratio, 'deviation': red_ratio - 0.333, 'reversion_pressure': 0.333 - red_ratio},
                'blue': {'count': blue_count, 'ratio': blue_ratio, 'deviation': blue_ratio - 0.333, 'reversion_pressure': 0.333 - blue_ratio},
                'green': {'count': green_count, 'ratio': green_ratio, 'deviation': green_ratio - 0.333, 'reversion_pressure': 0.333 - green_ratio}
            }
        }


class MeanReversionPredictor:
    """基于回归平均的预测器"""
    
    def __init__(self, analysis_periods: int = 365):
        self.analysis_periods = analysis_periods
    
    def predict(self, data: pd.DataFrame, top_k: int = 38) -> List[Dict]:
        """
        基于回归平均预测
        
        策略：
        1. 出现次数低于平均的号码，回归压力大，更可能出现
        2. 结合属性（大小、单双、波色）的回归压力
        3. 综合评分，选择最可能回归的号码
        """
        # 分析号码偏离
        number_deviations = MeanReversionAnalyzer.analyze_number_deviation(
            data, self.analysis_periods
        )
        
        # 分析属性偏离
        attribute_deviations = MeanReversionAnalyzer.analyze_attribute_deviation(
            data, self.analysis_periods
        )
        
        # 计算每个号码的综合回归分数
        scores = {}
        
        for num in range(1, 50):
            # 基础回归分数（出现次数低的得分高）
            base_score = number_deviations[num]['reversion_pressure']
            
            # 属性加成
            big_small_bonus = 0
            if num >= 25:  # 大数
                big_small_bonus = attribute_deviations['big_small']['big']['reversion_pressure']
            else:  # 小数
                big_small_bonus = attribute_deviations['big_small']['small']['reversion_pressure']
            
            odd_even_bonus = 0
            if num % 2 == 1:  # 单数
                odd_even_bonus = attribute_deviations['odd_even']['odd']['reversion_pressure']
            else:  # 双数
                odd_even_bonus = attribute_deviations['odd_even']['even']['reversion_pressure']
            
            color_bonus = 0
            red_nums = {1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46}
            blue_nums = {3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48}
            green_nums = {5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49}
            
            if num in red_nums:
                color_bonus = attribute_deviations['color']['red']['reversion_pressure']
            elif num in blue_nums:
                color_bonus = attribute_deviations['color']['blue']['reversion_pressure']
            elif num in green_nums:
                color_bonus = attribute_deviations['color']['green']['reversion_pressure']
            
            # 综合得分（权重可调）
            total_score = (
                base_score * 10.0 +  # 号码本身的回归压力
                big_small_bonus * 3.0 +  # 大小回归压力
                odd_even_bonus * 3.0 +  # 单双回归压力
                color_bonus * 2.0  # 波色回归压力
            )
            
            scores[num] = total_score
        
        # 排序并返回top_k
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        predictions = []
        for rank, (num, score) in enumerate(sorted_nums[:top_k], 1):
            predictions.append({
                '号码': num,
                '回归分数': f"{score:.3f}",
                '实际次数': number_deviations[num]['actual_count'],
                '期望次数': f"{number_deviations[num]['expected_count']:.1f}",
                '偏离度': f"{number_deviations[num]['deviation_ratio']*100:.1f}%"
            })
        
        return predictions
    
    def backtest(self, data: pd.DataFrame, test_periods: int = 100) -> Dict:
        """回测回归平均策略"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]['特码']
            
            # 预测
            predictions = self.predict(train_data, top_k=38)
            predicted_nums = [p['号码'] for p in predictions]
            
            # 判断命中
            hit = actual in predicted_nums
            results.append(1 if hit else 0)
        
        accuracy = np.mean(results)
        
        return {
            'accuracy': accuracy,
            'test_periods': test_periods,
            'hits': sum(results),
            'total': len(results)
        }


class MeanReversionLearningEngine:
    """回归平均学习引擎 v8.0"""
    
    def __init__(self):
        self.predictor = None
        self.best_result = None
        self.analysis_periods = None
    
    def auto_learn(self, data: pd.DataFrame, 
                   min_analysis_periods: int = 200,
                   max_analysis_periods: int = 800,
                   test_periods: int = 100,
                   verbose: bool = True) -> Dict:
        """
        自动学习最优分析期数
        
        尝试不同的分析期数（200-800期），找到回测准确率最高的配置
        """
        if verbose:
            print("\n" + "="*70)
            print("🔄 回归平均学习引擎 v8.0")
            print("="*70)
            print(f"数据总期数: {len(data)}")
            print(f"测试不同分析期数: {min_analysis_periods}-{max_analysis_periods}")
            print(f"回测期数: {test_periods}")
            print("="*70)
            print()
        
        best_accuracy = 0
        best_periods = None
        results_history = []
        
        # 尝试不同的分析期数
        test_ranges = [200, 300, 365, 400, 500, 600, 700, 800]
        test_ranges = [p for p in test_ranges if min_analysis_periods <= p <= max_analysis_periods]
        
        for analysis_periods in test_ranges:
            if len(data) < analysis_periods + test_periods + 50:
                continue
            
            predictor = MeanReversionPredictor(analysis_periods=analysis_periods)
            result = predictor.backtest(data, test_periods=test_periods)
            
            results_history.append({
                'analysis_periods': analysis_periods,
                'accuracy': result['accuracy']
            })
            
            if verbose:
                print(f"分析期数 {analysis_periods:3d} → 回测准确率: {result['accuracy']*100:.2f}%")
            
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_periods = analysis_periods
        
        if verbose:
            print()
            print("="*70)
            print(f"🎯 最优配置找到！")
            print(f"最佳分析期数: {best_periods}期")
            print(f"最高准确率: {best_accuracy*100:.2f}%")
            print("="*70)
        
        # 使用最优配置创建预测器
        self.predictor = MeanReversionPredictor(analysis_periods=best_periods)
        self.analysis_periods = best_periods
        
        self.best_result = {
            'success': True,
            'best_analysis_periods': best_periods,
            'accuracy': best_accuracy,
            'results_history': results_history
        }
        
        return self.best_result
    
    def predict(self, data: pd.DataFrame, top_k: int = 38) -> List[Dict]:
        """使用学习到的最优配置进行预测"""
        if self.predictor is None:
            # 如果没有学习，使用默认配置
            self.predictor = MeanReversionPredictor(analysis_periods=365)
        
        return self.predictor.predict(data, top_k)
    
    def get_deviation_analysis(self, data: pd.DataFrame) -> Dict:
        """获取当前的偏离分析"""
        if self.analysis_periods is None:
            analysis_periods = 365
        else:
            analysis_periods = self.analysis_periods
        
        number_dev = MeanReversionAnalyzer.analyze_number_deviation(data, analysis_periods)
        attribute_dev = MeanReversionAnalyzer.analyze_attribute_deviation(data, analysis_periods)
        
        return {
            'analysis_periods': analysis_periods,
            'number_deviations': number_dev,
            'attribute_deviations': attribute_dev
        }
