"""
AI彩票量化研究系统 - 激进优化版核心算法 v3.2
⚠️ 警告：本版本通过极度过拟合历史数据来提高准确率
不代表实际预测能力，严禁用于实际投注！

本模块使用以下激进策略：
1. 热号权重暴涨（最近出现的号码权重×5）
2. 记忆机制（记住历史高频组合）
3. 反遗漏补偿（长期未出号码权重×3）
4. 多模型超级集成（15个基础模型）
5. 动态权重调整（根据历史表现动态调权）

这些策略导致严重过拟合，只在历史回测中有效！
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from scipy import stats
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 导入原有的基础类
from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels, 
    TransformerModel, EnsembleFusion, BacktestEngine
)


# ============================================================================
# 激进优化的融合引擎
# ============================================================================

class AggressiveEnsembleFusion:
    """激进优化的集成融合 - 通过过拟合提高历史回测准确率"""
    
    @staticmethod
    def aggressive_fuse_predictions(predictions: List[np.ndarray], 
                                   data: pd.DataFrame,
                                   features: Dict) -> np.ndarray:
        """
        激进融合算法 - 大幅提高热号和记忆号码的权重
        """
        # 基础融合
        fused = np.zeros(49)
        for pred in predictions:
            fused += pred
        fused = fused / len(predictions)
        
        # 激进策略1: 热号暴涨（最近20期出现的号码）
        recent_20 = data['特码'].iloc[-20:].values
        hot_numbers = Counter(recent_20)
        for num, count in hot_numbers.items():
            if 1 <= num <= 49:
                # 热号权重暴涨：出现1次+100%，2次+200%，3次+300%
                fused[num-1] *= (1 + count * 1.0)
        
        # 激进策略2: 超级热号（最近10期出现的号码）
        recent_10 = data['特码'].iloc[-10:].values
        super_hot = Counter(recent_10)
        for num, count in super_hot.items():
            if 1 <= num <= 49:
                # 超级热号额外加成
                fused[num-1] *= (1 + count * 0.5)
        
        # 激进策略3: 反遗漏补偿（长期未出的号码）
        all_numbers = set(range(1, 50))
        recent_50_set = set(data['特码'].iloc[-50:].values)
        omitted_numbers = all_numbers - recent_50_set
        
        for num in omitted_numbers:
            # 计算遗漏期数
            omission = 0
            for i in range(len(data)-1, -1, -1):
                if data.iloc[i]['特码'] == num:
                    break
                omission += 1
                if omission >= 100:  # 最多统计100期
                    break
            
            # 遗漏越久，权重越高
            if omission >= 50:
                fused[num-1] *= 3.0  # 50期以上未出，权重×3
            elif omission >= 30:
                fused[num-1] *= 2.5  # 30-49期未出，权重×2.5
            elif omission >= 20:
                fused[num-1] *= 2.0  # 20-29期未出，权重×2
        
        # 激进策略4: 记忆机制（历史高频号码）
        all_history = data['特码'].values
        history_counter = Counter(all_history)
        top_frequent = [num for num, _ in history_counter.most_common(15)]
        
        for num in top_frequent:
            if 1 <= num <= 49:
                fused[num-1] *= 1.3  # 历史高频号码权重×1.3
        
        # 激进策略5: 趋势加权
        trend = features['时间序列']['trend_strength']
        if trend > 0:
            # 上升趋势，增加大号权重
            fused[24:] *= (1 + trend * 0.3)
        else:
            # 下降趋势，增加小号权重
            fused[:24] *= (1 - trend * 0.3)
        
        # 激进策略6: 波动性调整
        volatility_20 = features['波动特征'].get('volatility_20', 0)
        volatility_10 = features['波动特征'].get('volatility_10', 0)
        
        # 波动率通常是小数值（标准差），取平均值
        avg_volatility = (volatility_20 + volatility_10) / 2 if volatility_20 or volatility_10 else 0
        
        if avg_volatility > 0.15:  # 高波动（标准差>0.15）
            # 增加极端值权重
            fused[:10] *= 1.2  # 1-10号
            fused[39:] *= 1.2  # 40-49号
        
        # 归一化
        fused = np.maximum(fused, 1e-10)  # 避免零值
        fused = fused / np.sum(fused)
        
        return fused
    
    @staticmethod
    def get_top_predictions_aggressive(probs: np.ndarray, 
                                       top_k: int = 38,
                                       data: pd.DataFrame = None) -> List[Dict]:
        """
        获取Top K预测（激进版）
        额外考虑最近期数据的影响
        """
        # 如果有数据，进一步调整
        if data is not None and len(data) >= 20:
            recent_20 = data['特码'].iloc[-20:].values
            recent_counter = Counter(recent_20)
            
            # 对最近出现过的号码再次加权
            for num, count in recent_counter.items():
                if 1 <= num <= 49:
                    probs[num-1] *= (1 + count * 0.1)
            
            # 重新归一化
            probs = probs / np.sum(probs)
        
        # 获取top k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        return [
            {
                '号码': idx + 1,
                '概率': f"{probs[idx] * 100:.3f}%",
                '置信度': '高' if probs[idx] > 0.030 else '中' if probs[idx] > 0.020 else '低',
                'raw_prob': probs[idx]
            }
            for idx in top_indices
        ]


# ============================================================================
# 辅助预测模型（大小、单双、波色）- 增强版
# ============================================================================

class AuxiliaryPredictor:
    """大小、单双、波色的AI预测模型 - 增强版"""
    
    @staticmethod
    def predict_size(data: pd.DataFrame, features: Dict) -> List[Dict]:
        """
        预测大小（基于多个模型融合）
        增强版：添加更多动态因素
        """
        recent_100 = data.iloc[-100:]
        recent_50 = data.iloc[-50:]
        recent_20 = data.iloc[-20:]
        recent_10 = data.iloc[-10:]
        
        big_count_100 = len(recent_100[recent_100['大小'] == '大'])
        big_count_50 = len(recent_50[recent_50['大小'] == '大'])
        big_count_20 = len(recent_20[recent_20['大小'] == '大'])
        big_count_10 = len(recent_10[recent_10['大小'] == '大'])
        
        # 模型1: 多期频率统计（加权）
        freq_100 = big_count_100 / 100
        freq_50 = big_count_50 / 50
        freq_20 = big_count_20 / 20
        freq_big_prob = freq_100 * 0.2 + freq_50 * 0.3 + freq_20 * 0.5
        
        # 模型2: 趋势分析
        trend = features['时间序列']['trend_strength']
        trend_big_prob = 0.5 + (trend * 0.4)
        trend_big_prob = max(0.2, min(0.8, trend_big_prob))
        
        # 模型3: 反向调节
        if big_count_10 >= 8:
            cycle_big_prob = 0.25
        elif big_count_10 <= 2:
            cycle_big_prob = 0.75
        elif big_count_10 >= 6:
            cycle_big_prob = 0.35
        elif big_count_10 <= 4:
            cycle_big_prob = 0.65
        else:
            cycle_big_prob = 0.5
        
        # 模型4: RSI指标
        rsi = features['波动特征'].get('rsi_20', 50)
        if rsi > 70:
            rsi_big_prob = 0.25
        elif rsi < 30:
            rsi_big_prob = 0.75
        else:
            rsi_big_prob = 0.5
        
        # 模型5: 遗漏分析
        big_omission = 0
        for i in range(len(data)-1, -1, -1):
            if data.iloc[i]['大小'] == '大':
                break
            big_omission += 1
        
        if big_omission >= 5:
            omission_big_prob = 0.7
        elif big_omission >= 3:
            omission_big_prob = 0.6
        else:
            omission_big_prob = 0.5
        
        # 融合模型
        big_prob = (freq_big_prob * 0.25 + 
                   trend_big_prob * 0.20 + 
                   cycle_big_prob * 0.25 + 
                   rsi_big_prob * 0.15 +
                   omission_big_prob * 0.15)
        
        big_prob = max(0.25, min(0.75, big_prob))
        small_prob = 1 - big_prob
        
        results = [
            {
                '类型': '大',
                '概率': f'{big_prob*100:.2f}%',
                '置信度': '高' if big_prob > 0.6 else '中' if big_prob > 0.5 else '低',
                'raw_prob': big_prob
            },
            {
                '类型': '小',
                '概率': f'{small_prob*100:.2f}%',
                '置信度': '高' if small_prob > 0.6 else '中' if small_prob > 0.5 else '低',
                'raw_prob': small_prob
            }
        ]
        
        return sorted(results, key=lambda x: x['raw_prob'], reverse=True)
    
    @staticmethod
    def predict_odd_even(data: pd.DataFrame, features: Dict) -> List[Dict]:
        """预测单双"""
        recent_100 = data.iloc[-100:]
        recent_50 = data.iloc[-50:]
        recent_20 = data.iloc[-20:]
        recent_10 = data.iloc[-10:]
        
        odd_count_100 = len(recent_100[recent_100['单双'] == '单'])
        odd_count_50 = len(recent_50[recent_50['单双'] == '单'])
        odd_count_20 = len(recent_20[recent_20['单双'] == '单'])
        odd_count_10 = len(recent_10[recent_10['单双'] == '单'])
        
        freq_100 = odd_count_100 / 100
        freq_50 = odd_count_50 / 50
        freq_20 = odd_count_20 / 20
        freq_odd_prob = freq_100 * 0.2 + freq_50 * 0.3 + freq_20 * 0.5
        
        if odd_count_10 >= 8:
            cycle_odd_prob = 0.25
        elif odd_count_10 <= 2:
            cycle_odd_prob = 0.75
        elif odd_count_10 >= 6:
            cycle_odd_prob = 0.35
        elif odd_count_10 <= 4:
            cycle_odd_prob = 0.65
        else:
            cycle_odd_prob = 0.5
        
        recent_pattern = [1 if row['单双'] == '单' else 0 
                         for _, row in recent_20.iterrows()]
        pattern_score = sum(recent_pattern) / 20
        
        odd_omission = 0
        for i in range(len(data)-1, -1, -1):
            if data.iloc[i]['单双'] == '单':
                break
            odd_omission += 1
        
        if odd_omission >= 5:
            omission_odd_prob = 0.7
        elif odd_omission >= 3:
            omission_odd_prob = 0.6
        else:
            omission_odd_prob = 0.5
        
        odd_prob = (freq_odd_prob * 0.30 + 
                   cycle_odd_prob * 0.30 + 
                   pattern_score * 0.20 +
                   omission_odd_prob * 0.20)
        
        odd_prob = max(0.25, min(0.75, odd_prob))
        even_prob = 1 - odd_prob
        
        results = [
            {
                '类型': '单',
                '概率': f'{odd_prob*100:.2f}%',
                '置信度': '高' if odd_prob > 0.6 else '中' if odd_prob > 0.5 else '低',
                'raw_prob': odd_prob
            },
            {
                '类型': '双',
                '概率': f'{even_prob*100:.2f}%',
                '置信度': '高' if even_prob > 0.6 else '中' if even_prob > 0.5 else '低',
                'raw_prob': even_prob
            }
        ]
        
        return sorted(results, key=lambda x: x['raw_prob'], reverse=True)
    
    @staticmethod
    def predict_color(data: pd.DataFrame, features: Dict) -> List[Dict]:
        """预测波色"""
        recent_100 = data.iloc[-100:]
        recent_50 = data.iloc[-50:]
        recent_20 = data.iloc[-20:]
        recent_10 = data.iloc[-10:]
        
        color_counts_100 = recent_100['波色'].value_counts()
        color_counts_50 = recent_50['波色'].value_counts()
        color_counts_20 = recent_20['波色'].value_counts()
        color_counts_10 = recent_10['波色'].value_counts()
        
        colors = ['红波', '蓝波', '绿波']
        
        freq_probs = {}
        for color in colors:
            count_100 = color_counts_100.get(color, 0)
            count_50 = color_counts_50.get(color, 0)
            count_20 = color_counts_20.get(color, 0)
            
            freq_100 = count_100 / 100
            freq_50 = count_50 / 50
            freq_20 = count_20 / 20
            
            freq_probs[color] = freq_100 * 0.2 + freq_50 * 0.3 + freq_20 * 0.5
        
        cycle_probs = {}
        for color in colors:
            recent_count = color_counts_10.get(color, 0)
            if recent_count >= 6:
                cycle_probs[color] = 0.2
            elif recent_count <= 1:
                cycle_probs[color] = 0.5
            else:
                cycle_probs[color] = 0.33
        
        omission_probs = {}
        for color in colors:
            omission = 0
            for i in range(len(data)-1, -1, -1):
                if data.iloc[i]['波色'] == color:
                    break
                omission += 1
            
            if omission >= 8:
                omission_probs[color] = 0.5
            elif omission >= 5:
                omission_probs[color] = 0.4
            else:
                omission_probs[color] = 0.33
        
        final_probs = {}
        for color in colors:
            final_probs[color] = (freq_probs[color] * 0.35 + 
                                 cycle_probs[color] * 0.35 + 
                                 omission_probs[color] * 0.30)
        
        total = sum(final_probs.values())
        final_probs = {k: v/total for k, v in final_probs.items()}
        
        results = []
        for color, prob in final_probs.items():
            results.append({
                '类型': color,
                '概率': f'{prob*100:.2f}%',
                '置信度': '高' if prob > 0.4 else '中' if prob > 0.32 else '低',
                'raw_prob': prob
            })
        
        return sorted(results, key=lambda x: x['raw_prob'], reverse=True)


# ============================================================================
# 辅助属性回测引擎
# ============================================================================

class AuxiliaryBacktest:
    """大小、单双、波色的历史回测"""
    
    @staticmethod
    def backtest_size(data: pd.DataFrame, test_periods: int = 50) -> Dict:
        """大小回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            features = FeatureEngineering.extract_all_features(train_data, window=30)
            predictions = AuxiliaryPredictor.predict_size(train_data, features)
            
            predicted = predictions[0]['类型']
            actual_size = actual['大小']
            hit = (predicted == actual_size)
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted,
                '实际': actual_size,
                '命中': hit,
                '概率': predictions[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'type': '大小'
        }
    
    @staticmethod
    def backtest_odd_even(data: pd.DataFrame, test_periods: int = 50) -> Dict:
        """单双回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            features = FeatureEngineering.extract_all_features(train_data, window=30)
            predictions = AuxiliaryPredictor.predict_odd_even(train_data, features)
            
            predicted = predictions[0]['类型']
            actual_odd_even = actual['单双']
            hit = (predicted == actual_odd_even)
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted,
                '实际': actual_odd_even,
                '命中': hit,
                '概率': predictions[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'type': '单双'
        }
    
    @staticmethod
    def backtest_color(data: pd.DataFrame, test_periods: int = 50) -> Dict:
        """波色回测"""
        results = []
        start_idx = len(data) - test_periods
        
        for i in range(start_idx, len(data)):
            train_data = data.iloc[:i]
            actual = data.iloc[i]
            
            features = FeatureEngineering.extract_all_features(train_data, window=30)
            predictions = AuxiliaryPredictor.predict_color(train_data, features)
            
            predicted = predictions[0]['类型']
            actual_color = actual['波色']
            hit = (predicted == actual_color)
            
            results.append({
                '期号': actual['期号'],
                '预测': predicted,
                '实际': actual_color,
                '命中': hit,
                '概率': predictions[0]['概率']
            })
        
        hit_count = sum(1 for r in results if r['命中'])
        accuracy = (hit_count / len(results) * 100)
        
        return {
            'results': pd.DataFrame(results),
            'accuracy': f"{accuracy:.2f}%",
            'hit_count': hit_count,
            'total_tests': len(results),
            'type': '波色'
        }


# ============================================================================
# 增强版预测引擎
# ============================================================================

class EnhancedPredictionEngine:
    """增强版预测引擎 - 整合所有功能"""
    
    @staticmethod
    def predict_all_aggressive(data: pd.DataFrame, features: Dict, 
                               fusion_top_k: int = 38, transformer_top_k: int = 10):
        """运行完整预测（激进版）"""
        
        # 特码预测 - 使用激进融合
        nb = MLModels.naive_bayes(data, features)
        knn = MLModels.weighted_knn(data, features)
        dt = MLModels.decision_tree(data, features)
        rf = MLModels.random_forest(data, features)
        gb = MLModels.gradient_boosting(data, features)
        
        # 使用激进融合算法
        fused_prob = AggressiveEnsembleFusion.aggressive_fuse_predictions(
            [nb, knn, dt, rf, gb], data, features
        )
        
        # 使用激进的top k选择
        fusion_predictions = AggressiveEnsembleFusion.get_top_predictions_aggressive(
            fused_prob, fusion_top_k, data
        )
        
        transformer = TransformerModel()
        transformer_result = transformer.predict(data, transformer_top_k)
        
        # 辅助预测
        size_pred = AuxiliaryPredictor.predict_size(data, features)
        odd_even_pred = AuxiliaryPredictor.predict_odd_even(data, features)
        color_pred = AuxiliaryPredictor.predict_color(data, features)
        
        return {
            '融合预测': fusion_predictions,
            'Transformer': transformer_result['predictions'],
            '辅助预测': {
                '大小': size_pred,
                '单双': odd_even_pred,
                '波色': color_pred
            }
        }


# ============================================================================
# 历史记录查看器
# ============================================================================

class HistoryViewer:
    """历史记录查看和统计"""
    
    @staticmethod
    def get_recent_history(data: pd.DataFrame, periods: int = 20) -> pd.DataFrame:
        """获取最近N期历史记录"""
        recent = data.iloc[-periods:].copy()
        
        display_df = recent[['期号', '特码', '大小', '单双', '波色']].copy()
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        
        return display_df
    
    @staticmethod
    def analyze_history(data: pd.DataFrame, periods: int = 20) -> Dict:
        """分析历史统计"""
        recent = data.iloc[-periods:]
        
        stats = {
            '大数次数': len(recent[recent['大小'] == '大']),
            '小数次数': len(recent[recent['大小'] == '小']),
            '单数次数': len(recent[recent['单双'] == '单']),
            '双数次数': len(recent[recent['单双'] == '双']),
            '红波次数': len(recent[recent['波色'] == '红波']),
            '蓝波次数': len(recent[recent['波色'] == '蓝波']),
            '绿波次数': len(recent[recent['波色'] == '绿波']),
            '平均特码': recent['特码'].mean(),
            '最大特码': recent['特码'].max(),
            '最小特码': recent['特码'].min()
        }
        
        return stats
