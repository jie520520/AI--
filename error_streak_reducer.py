"""
防连错预测引擎 v8.1 - Anti-Consecutive-Loss Prediction Engine

用户提出的改进机制：
1. 每期实际预测49个号码（全部排序）
2. 默认使用前1-38位作为预测号码
3. 如果上期预测失败，本期使用11-49位作为预测号码
4. 目标：降低连错次数

改进版本：
- 动态调整策略
- 根据连错次数灵活调整
- 多种防连错模式
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class AntiConsecutiveLossPredictor:
    """防连错预测器"""
    
    def __init__(self, base_predictor, mode='user_proposed'):
        """
        初始化
        
        Args:
            base_predictor: 基础预测器（任意学习引擎）
            mode: 防连错模式
                - 'user_proposed': 用户提出的方案（失败→跳过前10个）
                - 'dynamic': 动态调整（根据连错次数）
                - 'mixed': 混合策略（成功用模型，失败用反向）
                - 'adaptive': 自适应权重
        """
        self.base_predictor = base_predictor
        self.mode = mode
        self.history = []  # 记录预测历史 [(predicted, actual, hit), ...]
    
    def predict(self, data: pd.DataFrame, top_k: int = 38) -> Tuple[List[int], Dict]:
        """
        防连错预测
        
        Returns:
            (预测号码列表, 详细信息字典)
        """
        # 获取基础预测器的完整49个号码排序
        full_prediction = self._get_full_prediction(data)
        
        # 分析最近的连错情况
        consecutive_losses = self._count_consecutive_losses()
        
        # 根据模式选择预测号码
        if self.mode == 'user_proposed':
            predicted_nums, info = self._user_proposed_strategy(
                full_prediction, top_k, consecutive_losses
            )
        elif self.mode == 'dynamic':
            predicted_nums, info = self._dynamic_strategy(
                full_prediction, top_k, consecutive_losses
            )
        elif self.mode == 'mixed':
            predicted_nums, info = self._mixed_strategy(
                data, full_prediction, top_k, consecutive_losses
            )
        elif self.mode == 'adaptive':
            predicted_nums, info = self._adaptive_strategy(
                data, full_prediction, top_k, consecutive_losses
            )
        else:
            # 默认策略
            predicted_nums = full_prediction[:top_k]
            info = {'strategy': 'default', 'consecutive_losses': consecutive_losses}
        
        return predicted_nums, info
    
    def _get_full_prediction(self, data: pd.DataFrame) -> List[int]:
        """获取基础预测器的完整49个号码排序"""
        # 使用基础预测器预测49个号码
        try:
            if hasattr(self.base_predictor, 'predict'):
                result = self.base_predictor.predict(data, top_k=49)
                
                # 处理不同返回格式
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], dict):
                        # 格式：[{'号码': 1, ...}, ...]
                        return [item['号码'] for item in result]
                    else:
                        # 格式：[1, 2, 3, ...]
                        return result
                else:
                    # 未知格式，返回默认排序
                    return list(range(1, 50))
            else:
                return list(range(1, 50))
        except:
            # 如果预测失败，返回默认排序
            return list(range(1, 50))
    
    def _count_consecutive_losses(self) -> int:
        """计算最近的连续失败次数"""
        if not self.history:
            return 0
        
        consecutive = 0
        for predicted, actual, hit in reversed(self.history):
            if not hit:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _user_proposed_strategy(
        self, 
        full_prediction: List[int], 
        top_k: int, 
        consecutive_losses: int
    ) -> Tuple[List[int], Dict]:
        """
        用户提出的策略
        
        - 上期成功或首期：使用1-38位
        - 上期失败：使用11-49位（跳过前10个）
        """
        if consecutive_losses == 0:
            # 上期成功或首期，使用前1-38
            predicted_nums = full_prediction[:top_k]
            strategy = f"默认策略: TOP 1-{top_k}"
        else:
            # 上期失败，跳过前10个，使用11-49位
            predicted_nums = full_prediction[10:10+top_k]
            strategy = f"防连错策略: TOP 11-{10+top_k} (跳过前10个)"
        
        info = {
            'strategy': strategy,
            'consecutive_losses': consecutive_losses,
            'mode': 'user_proposed',
            'range_start': 0 if consecutive_losses == 0 else 10,
            'range_end': top_k if consecutive_losses == 0 else 10+top_k
        }
        
        return predicted_nums, info
    
    def _dynamic_strategy(
        self, 
        full_prediction: List[int], 
        top_k: int, 
        consecutive_losses: int
    ) -> Tuple[List[int], Dict]:
        """
        动态调整策略（改进版）
        
        根据连错次数动态调整预测范围：
        - 连错0次：1-38位
        - 连错1次：1-38位（保持）
        - 连错2次：6-43位（跳过前5个）
        - 连错3次：11-48位（跳过前10个）
        - 连错4次及以上：随机从全部49个中选38个
        """
        if consecutive_losses <= 1:
            # 0-1次连错：保持默认策略
            start = 0
            predicted_nums = full_prediction[start:start+top_k]
            strategy = f"默认策略: TOP 1-{top_k}"
        elif consecutive_losses == 2:
            # 2次连错：跳过前5个
            start = 5
            predicted_nums = full_prediction[start:start+top_k]
            strategy = f"轻度调整: TOP 6-{5+top_k} (跳过前5个)"
        elif consecutive_losses == 3:
            # 3次连错：跳过前10个
            start = 10
            predicted_nums = full_prediction[start:start+top_k]
            strategy = f"中度调整: TOP 11-{10+top_k} (跳过前10个)"
        else:
            # 4次及以上连错：随机策略
            predicted_nums = np.random.choice(full_prediction, top_k, replace=False).tolist()
            strategy = f"激进调整: 随机选择{top_k}个 (连错{consecutive_losses}次)"
        
        info = {
            'strategy': strategy,
            'consecutive_losses': consecutive_losses,
            'mode': 'dynamic',
            'adjustment_level': min(consecutive_losses, 4)
        }
        
        return predicted_nums, info
    
    def _mixed_strategy(
        self, 
        data: pd.DataFrame,
        full_prediction: List[int], 
        top_k: int, 
        consecutive_losses: int
    ) -> Tuple[List[int], Dict]:
        """
        混合策略
        
        - 成功：100%使用模型预测
        - 失败1次：70%模型 + 30%冷号
        - 失败2次：50%模型 + 50%冷号
        - 失败3次及以上：30%模型 + 70%冷号
        """
        if consecutive_losses == 0:
            # 全部使用模型
            predicted_nums = full_prediction[:top_k]
            strategy = "100% 模型预测"
        else:
            # 计算模型和冷号的比例
            if consecutive_losses == 1:
                model_ratio = 0.7
            elif consecutive_losses == 2:
                model_ratio = 0.5
            else:
                model_ratio = 0.3
            
            model_count = int(top_k * model_ratio)
            cold_count = top_k - model_count
            
            # 从模型预测中选择
            model_picks = full_prediction[:model_count]
            
            # 找出冷号（最近100期出现次数最少的）
            recent_100 = data['特码'].iloc[-100:] if len(data) >= 100 else data['特码']
            freq = recent_100.value_counts()
            all_nums = set(range(1, 50))
            existing_nums = set(freq.index)
            
            # 按出现频率排序（从少到多）
            cold_nums = []
            for num in range(1, 50):
                if num not in set(model_picks):  # 不与模型预测重复
                    cold_nums.append((num, freq.get(num, 0)))
            
            cold_nums.sort(key=lambda x: x[1])  # 按频率升序
            cold_picks = [num for num, freq in cold_nums[:cold_count]]
            
            # 合并
            predicted_nums = model_picks + cold_picks
            strategy = f"混合策略: {int(model_ratio*100)}% 模型 + {int((1-model_ratio)*100)}% 冷号"
        
        info = {
            'strategy': strategy,
            'consecutive_losses': consecutive_losses,
            'mode': 'mixed'
        }
        
        return predicted_nums, info
    
    def _adaptive_strategy(
        self, 
        data: pd.DataFrame,
        full_prediction: List[int], 
        top_k: int, 
        consecutive_losses: int
    ) -> Tuple[List[int], Dict]:
        """
        自适应策略
        
        根据连错情况调整预测窗口：
        - 成功：使用最近期数据
        - 失败：逐渐增加历史期数
        """
        # 这个策略需要重新预测，这里简化处理
        # 实际应该调整base_predictor的参数
        if consecutive_losses == 0:
            predicted_nums = full_prediction[:top_k]
            strategy = "自适应: 标准窗口"
        elif consecutive_losses <= 2:
            # 轻度调整：使用中间段
            start = 3
            predicted_nums = full_prediction[start:start+top_k]
            strategy = "自适应: 扩展窗口"
        else:
            # 重度调整：使用后段
            start = 8
            predicted_nums = full_prediction[start:start+top_k]
            strategy = "自适应: 长期窗口"
        
        info = {
            'strategy': strategy,
            'consecutive_losses': consecutive_losses,
            'mode': 'adaptive'
        }
        
        return predicted_nums, info
    
    def update_history(self, predicted: List[int], actual: int):
        """
        更新预测历史
        
        Args:
            predicted: 预测的号码列表
            actual: 实际开出的号码
        """
        hit = actual in predicted
        self.history.append((predicted, actual, hit))
        
        # 只保留最近100期的历史
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_statistics(self) -> Dict:
        """获取防连错效果统计"""
        if not self.history:
            return {}
        
        total = len(self.history)
        hits = sum(1 for _, _, hit in self.history if hit)
        accuracy = hits / total if total > 0 else 0
        
        # 计算最大连错次数
        max_consecutive_losses = 0
        current_consecutive = 0
        for _, _, hit in self.history:
            if not hit:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0
        
        # 计算平均连错次数
        consecutive_loss_lengths = []
        current_consecutive = 0
        for _, _, hit in self.history:
            if not hit:
                current_consecutive += 1
            else:
                if current_consecutive > 0:
                    consecutive_loss_lengths.append(current_consecutive)
                current_consecutive = 0
        
        if current_consecutive > 0:
            consecutive_loss_lengths.append(current_consecutive)
        
        avg_consecutive_losses = (
            np.mean(consecutive_loss_lengths) 
            if consecutive_loss_lengths 
            else 0
        )
        
        return {
            'total_tests': total,
            'hits': hits,
            'accuracy': accuracy,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_consecutive_losses': avg_consecutive_losses,
            'consecutive_loss_count': len(consecutive_loss_lengths)
        }
