"""
防连错预测引擎 v8.2 - Anti-Consecutive-Loss Prediction Engine（修复版）

核心规则（与原始需求一致）：
1. 每期实际预测49个号码（全部排序）
2. 默认使用前1-K位作为预测号码（K = top_k，例如38）
3. 仅当"上一期实际预测"未命中时，本期才切换为跳过前10位的预测号码
   （即 TOP (11) ~ (10+K)）
4. 上一期一旦命中，立刻恢复默认策略

本次修复的问题：
----------------------------------------------------------------------
旧版 _get_full_prediction() 只认 base_predictor.predict() 这一个方法名。
但"遗传算法"模式的引擎(SelfLearningEngine)实际方法名是
predict_with_learned_rules()，"超级学习"模式的引擎(SuperLearningEngine)
实际方法名是 predict_ultra()，两者都没有名为 predict 的方法。

结果是 hasattr(self.base_predictor, 'predict') 判定为 False，
代码静默走到 fallback 分支，把 full_prediction 退化成固定的
[1, 2, 3, ..., 49]，跟真实模型预测完全无关。这会导致：
  - 默认策略永远等价于"猜1~38"
  - 防连错策略永远等价于"猜11~48"
  - 命中与否变成纯随机，和"上一期是否真的命中"脱节
  - 表现出来就是：明明默认策略上一期命中了，这一期却莫名其妙
    又触发了防连错（其实是因为全排序数据源本身就是错的，状态机
    本身的"命中→默认，未命中→防连错"逻辑没有问题）

修复方式：按引擎类型依次尝试 predict / predict_with_learned_rules /
predict_ultra，找到第一个真实存在的方法再调用，仍然失败才退化为
默认顺序（并且只在这种极端情况下退化）。
----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Optional


class AntiConsecutiveLossPredictor:
    """防连错预测器"""

    # 不同学习引擎可能用到的"获取完整49码排序"的方法名，按优先级尝试
    _CANDIDATE_METHOD_NAMES = (
        'predict',                     # 回归平均 / 完全自动 / 极限优化
        'predict_with_learned_rules',  # 遗传算法 (SelfLearningEngine)
        'predict_ultra',               # 超级学习 (SuperLearningEngine)
    )

    def __init__(self, base_predictor, mode='user_proposed'):
        """
        初始化

        Args:
            base_predictor: 基础预测器（任意学习引擎实例）
            mode: 防连错模式
                - 'user_proposed': 用户提出的方案（上期失败→跳过前10个）
                - 'dynamic': 动态调整（根据连错次数）
                - 'mixed': 混合策略（成功用模型，失败用冷号混合）
                - 'adaptive': 自适应窗口
        """
        self.base_predictor = base_predictor
        self.mode = mode
        self.history = []  # [(predicted, actual, hit), ...]

        # 缓存这个 base_predictor 真正可用的预测方法，避免每期都重新探测
        self._resolved_predict_func: Optional[Callable] = self._resolve_predict_func(base_predictor)

    @classmethod
    def _resolve_predict_func(cls, base_predictor) -> Optional[Callable]:
        """按优先级找到 base_predictor 上真实存在的预测方法"""
        for name in cls._CANDIDATE_METHOD_NAMES:
            func = getattr(base_predictor, name, None)
            if callable(func):
                return func
        return None

    def predict(self, data: pd.DataFrame, top_k: int = 38) -> Tuple[List[int], Dict]:
        """
        防连错预测

        Returns:
            (预测号码列表, 详细信息字典)
        """
        full_prediction = self._get_full_prediction(data)
        consecutive_losses = self._count_consecutive_losses()

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
            predicted_nums = full_prediction[:top_k]
            info = {'strategy': 'default', 'consecutive_losses': consecutive_losses}

        return predicted_nums, info

    def _get_full_prediction(self, data: pd.DataFrame) -> List[int]:
        """
        获取基础预测器的完整49个号码排序。

        修复点：不再只认 'predict' 这一个方法名，而是用
        _resolved_predict_func（在 __init__ 时已按优先级探测好）。
        只有当 base_predictor 确实没有任何可用预测方法、或调用过程中
        真的抛异常时，才退化为 [1..49]。
        """
        if self._resolved_predict_func is None:
            return list(range(1, 50))

        try:
            result = self._resolved_predict_func(data, top_k=49)

            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    # 格式：[{'号码': 1, ...}, ...]（如回归平均引擎）
                    nums = [item['号码'] for item in result]
                else:
                    # 格式：[1, 2, 3, ...]（如遗传算法 / 超级学习 / 极限优化 / 自动学习）
                    nums = list(result)

                # 防御：确保拿到的是1~49的合法、去重排序，且凑满49个
                nums = [int(n) for n in nums if 1 <= int(n) <= 49]
                seen = set()
                deduped = []
                for n in nums:
                    if n not in seen:
                        seen.add(n)
                        deduped.append(n)
                if len(deduped) < 49:
                    # 补齐缺失号码（极少数引擎可能不会返回全部49个）
                    missing = [n for n in range(1, 50) if n not in seen]
                    deduped.extend(missing)
                return deduped[:49]

            return list(range(1, 50))
        except Exception:
            return list(range(1, 50))

    def _count_consecutive_losses(self) -> int:
        """计算最近的连续失败次数（仅看最近这一次起的连续未命中）"""
        if not self.history:
            return 0

        consecutive = 0
        for predicted, actual, hit in reversed(self.history):
            if not hit:
                consecutive += 1
            else:
                break

        return consecutive

    @staticmethod
    def _safe_slice(full_prediction: List[int], start: int, top_k: int) -> List[int]:
        """
        安全切片：取 full_prediction[start:start+top_k]，
        如果号码不够（比如 top_k 设得很大，start+top_k 超过49），
        用 full_prediction 中尚未用到的号码从前往后补齐，
        确保返回数量始终等于 top_k（除非 top_k > 49）。
        """
        top_k = min(top_k, len(full_prediction))
        end = start + top_k
        picked = full_prediction[start:end]

        if len(picked) < top_k:
            used = set(picked)
            remaining_needed = top_k - len(picked)
            pad = [n for n in full_prediction if n not in used][:remaining_needed]
            picked = picked + pad

        return picked

    def _user_proposed_strategy(
        self,
        full_prediction: List[int],
        top_k: int,
        consecutive_losses: int
    ) -> Tuple[List[int], Dict]:
        """
        用户提出的策略（核心规则，已确认行为）：

        - 上一期实际预测「命中」，或这是第一期（无历史）：
              本期使用默认策略 TOP 1~K
        - 上一期实际预测「未命中」：
              本期切换为防连错策略 TOP 11~(10+K)（跳过前10个）

        只要上一期命中（不论用的是默认策略还是防连错策略），
        本期立即恢复默认策略；不会出现"明明命中了还继续跳号"的情况。
        """
        if consecutive_losses == 0:
            predicted_nums = self._safe_slice(full_prediction, 0, top_k)
            strategy = f"默认策略: TOP 1-{top_k}"
            range_start, range_end = 0, top_k
        else:
            predicted_nums = self._safe_slice(full_prediction, 10, top_k)
            strategy = f"防连错策略: TOP 11-{10 + top_k} (跳过前10个)"
            range_start, range_end = 10, 10 + top_k

        info = {
            'strategy': strategy,
            'consecutive_losses': consecutive_losses,
            'mode': 'user_proposed',
            'range_start': range_start,
            'range_end': range_end,
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
        - 连错0-1次：1-K位
        - 连错2次：6-(5+K)位（跳过前5个）
        - 连错3次：11-(10+K)位（跳过前10个）
        - 连错4次及以上：从全部49个中随机选K个
        """
        if consecutive_losses <= 1:
            predicted_nums = self._safe_slice(full_prediction, 0, top_k)
            strategy = f"默认策略: TOP 1-{top_k}"
        elif consecutive_losses == 2:
            predicted_nums = self._safe_slice(full_prediction, 5, top_k)
            strategy = f"轻度调整: TOP 6-{5 + top_k} (跳过前5个)"
        elif consecutive_losses == 3:
            predicted_nums = self._safe_slice(full_prediction, 10, top_k)
            strategy = f"中度调整: TOP 11-{10 + top_k} (跳过前10个)"
        else:
            k = min(top_k, len(full_prediction))
            predicted_nums = np.random.choice(full_prediction, k, replace=False).tolist()
            strategy = f"激进调整: 随机选择{k}个 (连错{consecutive_losses}次)"

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

        - 命中：100%使用模型预测
        - 失败1次：70%模型 + 30%冷号
        - 失败2次：50%模型 + 50%冷号
        - 失败3次及以上：30%模型 + 70%冷号
        """
        if consecutive_losses == 0:
            predicted_nums = self._safe_slice(full_prediction, 0, top_k)
            strategy = "100% 模型预测"
        else:
            if consecutive_losses == 1:
                model_ratio = 0.7
            elif consecutive_losses == 2:
                model_ratio = 0.5
            else:
                model_ratio = 0.3

            model_count = int(top_k * model_ratio)
            cold_count = top_k - model_count

            model_picks = full_prediction[:model_count]

            recent_100 = data['特码'].iloc[-100:] if len(data) >= 100 else data['特码']
            freq = recent_100.value_counts()

            cold_nums = []
            for num in range(1, 50):
                if num not in set(model_picks):
                    cold_nums.append((num, freq.get(num, 0)))

            cold_nums.sort(key=lambda x: x[1])
            cold_picks = [num for num, _ in cold_nums[:cold_count]]

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
        - 命中：标准窗口
        - 轻度连错：扩展窗口
        - 重度连错：长期窗口
        """
        if consecutive_losses == 0:
            predicted_nums = self._safe_slice(full_prediction, 0, top_k)
            strategy = "自适应: 标准窗口"
        elif consecutive_losses <= 2:
            predicted_nums = self._safe_slice(full_prediction, 3, top_k)
            strategy = "自适应: 扩展窗口"
        else:
            predicted_nums = self._safe_slice(full_prediction, 8, top_k)
            strategy = "自适应: 长期窗口"

        info = {
            'strategy': strategy,
            'consecutive_losses': consecutive_losses,
            'mode': 'adaptive'
        }

        return predicted_nums, info

    def update_history(self, predicted: List[int], actual: int):
        """
        更新预测历史（必须用"本期实际采用的预测列表"调用，
        即 predict() 返回的 predicted_nums，而不是无防连错的对照预测）

        Args:
            predicted: 本期实际使用的预测号码列表
            actual: 本期实际开出的号码
        """
        hit = actual in predicted
        self.history.append((predicted, actual, hit))

        if len(self.history) > 100:
            self.history = self.history[-100:]

    def get_statistics(self) -> Dict:
        """获取防连错效果统计"""
        if not self.history:
            return {}

        total = len(self.history)
        hits = sum(1 for _, _, hit in self.history if hit)
        accuracy = hits / total if total > 0 else 0

        max_consecutive_losses = 0
        current_consecutive = 0
        for _, _, hit in self.history:
            if not hit:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0

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
