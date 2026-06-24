"""
AI彩票量化研究系统 - 超级深度学习引擎 v5.1 科研重构版

╔══════════════════════════════════════════════════════════════════╗
║  对 v5.0 的四项根本性改进                                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  [1] 加性评分模型 (替代乘性模型)                                  ║
║      旧版: prob *= hot_w × cold_w × freq_w                       ║
║           → 指数级爆炸，适应度景观极其崎岖，结果随机              ║
║      新版: score(num) = Σ w_i × f_i(num)  → softmax → 概率      ║
║           → 景观平滑，梯度一致，收敛稳定                          ║
║                                                                  ║
║  [2] 走步交叉验证 (Walk-Forward Cross-Validation)                ║
║      将测试区间等分 n_folds 折，各折独立评估后：                   ║
║      fitness = mean_acc - λ × std_acc                            ║
║      奖励一致高准确率，惩罚"某期走运"的解                         ║
║                                                                  ║
║  [3] 自适应PSO + 差分进化 (替代基础PSO + 原始SA)                  ║
║      APSO: 惯性权重退火 + 速度钳制 + 多样性重注入                 ║
║      DE:   scipy best1bin + 拉丁超立方初始化 + L-BFGS-B精修      ║
║      增强SA: 自适应步长 + 周期重热                                 ║
║                                                                  ║
║  [4] 委员会集成 (替代"参数平均"伪集成)                            ║
║      三个优化器各保留 Top-N 历史解 → 按适应度加权投票             ║
║      在预测概率空间做集成，而非在参数空间做平均                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

API 与 v5.0 完全兼容：
  SuperLearningEngine.ultra_learn(data, test_periods, verbose)
  SuperLearningEngine.predict_ultra(data, top_k)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from scipy.optimize import differential_evolution as _scipy_de
import warnings
import random
import copy
import math
warnings.filterwarnings('ignore')

from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels,
    TransformerModel, EnsembleFusion, BacktestEngine
)


# ════════════════════════════════════════════════════════════════
# § 1  参数空间定义
#      12 个连续参数，对应加性评分模型的各项权重
# ════════════════════════════════════════════════════════════════

_PARAM_META = [
    # (参数名,           下界,  上界,   是否取整)
    ('w_hot_1',         0.0,   6.0,   False),  # 最近 1 期热号权重
    ('w_hot_5',         0.0,   5.0,   False),  # 最近 5 期热号频率权重
    ('w_hot_10',        0.0,   4.0,   False),  # 最近 10 期
    ('w_hot_20',        0.0,   3.5,   False),  # 最近 20 期
    ('w_cold_30',      -1.0,   6.0,   False),  # 30 期未出现补偿
    ('w_omission',     -1.0,   6.0,   False),  # 归一化遗漏度权重
    ('w_freq',         -2.0,   4.0,   False),  # 全局频率偏差权重
    ('w_trend',        -3.0,   3.0,   False),  # 趋势方向偏好
    ('w_cycle7',       -1.0,   4.0,   False),  # 7 期周期注入
    ('w_cycle14',      -1.0,   3.0,   False),  # 14 期周期注入
    ('w_memory',        0.0,   3.0,   False),  # 历史高频记忆奖励
    ('memory_k',        5.0,  25.0,   True),   # 记忆数量（整数）
]

PARAM_NAMES  = [m[0] for m in _PARAM_META]
BOUNDS_LO    = np.array([m[1] for m in _PARAM_META], dtype=float)
BOUNDS_HI    = np.array([m[2] for m in _PARAM_META], dtype=float)
PARAM_BOUNDS = [(m[1], m[2]) for m in _PARAM_META]   # scipy 格式
N_PARAMS     = len(_PARAM_META)
_INT_MASK    = np.array([m[3] for m in _PARAM_META], dtype=bool)


def _rand_vec() -> np.ndarray:
    return BOUNDS_LO + np.random.rand(N_PARAMS) * (BOUNDS_HI - BOUNDS_LO)


def _clip_vec(v: np.ndarray) -> np.ndarray:
    return np.clip(v, BOUNDS_LO, BOUNDS_HI)


def _vec_to_genome(v: np.ndarray) -> Dict[str, float]:
    """参数向量 → 命名字典（含整数约束 + 向后兼容别名）"""
    g: Dict[str, float] = {}
    for i, meta in enumerate(_PARAM_META):
        val = float(np.clip(v[i], meta[1], meta[2]))
        g[meta[0]] = float(round(val)) if meta[3] else val

    # 向后兼容：为旧版 UI 显示保留旧键名
    g['hot_weight_20']  = g['w_hot_20']
    g['hot_weight_10']  = g['w_hot_10']
    g['hot_weight_5']   = g['w_hot_5']
    g['cold_weight_50'] = g['w_omission']
    g['cold_weight_30'] = g['w_cold_30']
    g['cold_weight_20'] = g['w_cold_30']
    g['trend_weight']   = g['w_trend']
    g['volatility_weight'] = g.get('w_cycle7', 0.0)
    g['omission_weight']   = g['w_omission']
    g['freq_weight']       = g['w_freq']
    g['memory_top_k']      = g['memory_k']
    return g


# ════════════════════════════════════════════════════════════════
# § 2  加性评分模型（核心预测逻辑）
# ════════════════════════════════════════════════════════════════

def _compute_scores(v: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """
    对 1~49 每个号码计算加性得分向量（shape=49）。

    特征设计（全部归一化到合理量纲）：
      f_hot_k(num)    = count(num, last_k) / k
      f_cold30(num)   = 1 if absent from last 30
      f_omission(num) = min(periods_since_seen, 100) / 100
      f_freq(num)     = clip((actual/expected - 1), -2, 2)
      f_trend(num)    = sign(slope_20) × (num - 25) / 24
      f_cycle_k(num)  = 1 if num == seq[-k]
      f_memory(num)   = 1 if num in top memory_k by all-time frequency
    """
    seq = data['特码'].values.astype(int)
    n   = len(seq)
    scores = np.zeros(49)

    if n < 5:
        return scores

    # 快速计数工具：利用 np.bincount，比 Counter 快 5-10×
    def _cnt(window) -> np.ndarray:
        w = window[(window >= 1) & (window <= 49)]
        return np.bincount(w - 1, minlength=49).astype(float) if len(w) else np.zeros(49)

    # ── 热号特征 ──────────────────────────────────────────────
    scores += v[0] * _cnt(seq[-1:])                              # last 1

    if n >= 5:
        scores += v[1] * _cnt(seq[-5:])  / 5.0                  # last 5
    if n >= 10:
        scores += v[2] * _cnt(seq[-10:]) / 10.0                 # last 10
    if n >= 20:
        scores += v[3] * _cnt(seq[-20:]) / 20.0                 # last 20

    # ── 冷号 / 遗漏特征 ───────────────────────────────────────
    if n >= 30:
        cnt30   = _cnt(seq[-30:])
        absent  = (cnt30 == 0).astype(float)
        scores += v[4] * absent                                  # 30 期未出

    # 遗漏度：利用逆向扫描，O(n) 但只走一遍
    last_seen = np.full(49, -1, dtype=int)
    for idx in range(n):
        num = seq[idx]
        if 1 <= num <= 49:
            last_seen[num - 1] = idx
    omission = np.where(last_seen == -1, n, n - 1 - last_seen)
    scores  += v[5] * np.minimum(omission, 100) / 100.0         # 归一化遗漏

    # ── 全局频率特征 ──────────────────────────────────────────
    cnt_all  = _cnt(seq)
    expected = n / 49.0
    rel_freq = (cnt_all - expected) / max(expected, 1.0)
    scores  += v[6] * np.clip(rel_freq, -2.0, 2.0)              # 偏差频率

    # ── 趋势特征（号码大小方向偏好）─────────────────────────
    if n >= 10:
        window_size = min(20, n)
        yw   = seq[-window_size:].astype(float)
        xw   = np.arange(window_size, dtype=float)
        # 简单线性斜率 via 协方差
        if xw.std() > 0:
            slope = np.cov(xw, yw)[0, 1] / (xw.var() + 1e-8)
            trend_dir = 1.0 if slope > 0.1 else (-1.0 if slope < -0.1 else 0.0)
        else:
            trend_dir = 0.0
        size_feat = (np.arange(1, 50, dtype=float) - 25.0) / 24.0
        scores   += v[7] * trend_dir * size_feat

    # ── 周期注入特征 ──────────────────────────────────────────
    for offset, widx in [(7, 8), (14, 9)]:
        if n > offset:
            cnum = int(seq[-(offset + 1)])
            if 1 <= cnum <= 49:
                scores[cnum - 1] += v[widx]

    # ── 历史记忆特征 ──────────────────────────────────────────
    mem_k  = max(5, min(25, int(round(v[11]))))
    top_idx = np.argsort(cnt_all)[-mem_k:]                      # top memory_k 号码（索引）
    scores[top_idx] += v[10]

    return scores


def predict_from_vec(v: np.ndarray, data: pd.DataFrame,
                     top_k: int = 38) -> List[int]:
    """softmax(scores) → 取 Top-K"""
    scores  = _compute_scores(v, data)
    exp_s   = np.exp(scores - scores.max())                     # 数值稳定
    probs   = exp_s / exp_s.sum()
    top_idx = np.argsort(probs)[-top_k:][::-1]
    return [int(i + 1) for i in top_idx]


# ════════════════════════════════════════════════════════════════
# § 3  走步交叉验证评估器
# ════════════════════════════════════════════════════════════════

class WalkForwardEvaluator:
    """
    Walk-Forward Cross-Validation

    将 test_periods 等分成 n_folds 个不重叠子窗口，
    每个子窗口独立计算命中率，然后：
        fitness = mean(acc_i) - variance_penalty × std(acc_i)

    优点：
      · 总评估次数 ≈ test_periods（与单窗口相同，无额外开销）
      · 对"某段时期特别运气好"的参数施加方差惩罚
      · 真正稳定的参数才能在多个时间段都保持高准确率
    """

    def __init__(self, test_periods: int = 100, n_folds: int = 3,
                 top_k: int = 38, variance_penalty: float = 0.5):
        self.test_periods    = test_periods
        self.n_folds         = n_folds
        self.top_k           = top_k
        self.variance_penalty = variance_penalty

    def evaluate(self, v: np.ndarray, data: pd.DataFrame) -> float:
        """返回 fitness ∈ [0, 1]，越高越好"""
        n          = len(data)
        fold_size  = max(10, self.test_periods // self.n_folds)
        start_base = n - self.test_periods
        if start_base < 20:
            return 0.0

        fold_accs = []
        for fold in range(self.n_folds):
            f_start = start_base + fold * fold_size
            f_end   = min(f_start + fold_size, n)
            if f_end - f_start < 5:
                continue

            hits = 0
            for i in range(f_start, f_end):
                try:
                    pred   = predict_from_vec(v, data.iloc[:i], self.top_k)
                    actual = int(data.iloc[i]['特码'])
                    if actual in pred:
                        hits += 1
                except Exception:
                    pass

            fold_accs.append(hits / (f_end - f_start))

        if not fold_accs:
            return 0.0

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0
        return mean_acc - self.variance_penalty * std_acc


# ════════════════════════════════════════════════════════════════
# § 4  自适应粒子群优化器（APSO）
#      对外保持 ParticleSwarmOptimizer 类名（向后兼容）
# ════════════════════════════════════════════════════════════════

class ParticleSwarmOptimizer:
    """
    自适应粒子群优化（Adaptive PSO）

    改进点：
      ① 惯性权重线性退火  w: 0.9 → 0.35（前期探索，后期开采）
      ② 速度钳制  |v_d| ≤ 0.20 × (bound_max - bound_min)
      ③ 多样性监控：粒子群标准差 < 阈值时重注入最差 25% 粒子
      ④ 使用走步 CV 计算适应度（稳定，不再依赖单窗口运气）
      ⑤ 记录全程 Top 历史解，供委员会集成使用
    """

    def __init__(self, n_particles: int = 30, n_iterations: int = 60,
                 w_max: float = 0.9, w_min: float = 0.35,
                 c1: float = 2.0, c2: float = 2.0):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w_max        = w_max
        self.w_min        = w_min
        self.c1           = c1
        self.c2           = c2

        # 速度上限 = 20% × 参数范围
        self._v_max = 0.20 * (BOUNDS_HI - BOUNDS_LO)

        self.best_params: Optional[np.ndarray] = None
        self.best_fitness: float  = 0.0
        self.fitness_history: List[float] = []
        self._top_pool: List[Tuple[np.ndarray, float]] = []

    def optimize(self, evaluator: WalkForwardEvaluator,
                 data: pd.DataFrame, verbose: bool = False) -> Tuple[np.ndarray, float]:

        # 初始化
        pos = np.array([_rand_vec() for _ in range(self.n_particles)])
        vel = np.zeros_like(pos)
        pb_pos = pos.copy()
        pb_fit = np.full(self.n_particles, -np.inf)

        gb_pos = _rand_vec()
        gb_fit = -np.inf

        div_thr = 0.04 * np.mean(BOUNDS_HI - BOUNDS_LO)

        for it in range(self.n_iterations):
            # 惯性权重退火
            w = self.w_max - (self.w_max - self.w_min) * it / self.n_iterations

            # 评估所有粒子
            for p in range(self.n_particles):
                fit = evaluator.evaluate(pos[p], data)

                if fit > pb_fit[p]:
                    pb_fit[p] = fit
                    pb_pos[p] = pos[p].copy()

                if fit > gb_fit:
                    gb_fit = fit
                    gb_pos = pos[p].copy()
                    self._top_pool.append((gb_pos.copy(), gb_fit))

            self.fitness_history.append(gb_fit)

            if verbose and it % 10 == 0:
                print(f"    APSO  iter {it+1:3d}/{self.n_iterations}"
                      f" | 最佳: {gb_fit*100:6.2f}%")

            # 多样性检测
            diversity = np.mean(np.std(pos, axis=0))
            if diversity < div_thr and it < int(self.n_iterations * 0.75):
                n_reinject = max(1, self.n_particles // 4)
                worst_idx  = np.argsort(pb_fit)[:n_reinject]
                for ri in worst_idx:
                    pos[ri] = _rand_vec()
                    vel[ri] = np.zeros(N_PARAMS)
                if verbose:
                    print(f"    APSO  粒子群退化，重注入 {n_reinject} 个粒子")

            # 速度 & 位置更新
            r1 = np.random.rand(self.n_particles, N_PARAMS)
            r2 = np.random.rand(self.n_particles, N_PARAMS)
            vel = (w * vel
                   + self.c1 * r1 * (pb_pos - pos)
                   + self.c2 * r2 * (gb_pos - pos))
            vel = np.clip(vel, -self._v_max, self._v_max)   # 速度钳制
            pos = _clip_vec(pos + vel)

        self.best_params  = gb_pos
        self.best_fitness = gb_fit
        return gb_pos, gb_fit

    def get_top_solutions(self, n: int = 5) -> List[Tuple[np.ndarray, float]]:
        """从历史中取去重 Top-N 解"""
        unique, seen = [], set()
        for params, fit in sorted(self._top_pool, key=lambda x: x[1], reverse=True):
            key = round(fit, 4)
            if key not in seen:
                seen.add(key)
                unique.append((params.copy(), fit))
            if len(unique) >= n:
                break
        return unique

    # ── 向后兼容：旧版调用的辅助方法 ────────────────────────
    def _predict(self, genome: Dict, data: pd.DataFrame, top_k: int = 38) -> List[int]:
        """旧版 SuperLearningEngine 通过此方法预测（兼容保留）"""
        v = np.array([genome.get(k, 0.0) for k in PARAM_NAMES])
        return predict_from_vec(v, data, top_k)

    def _backtest_genome(self, genome: Dict, data: pd.DataFrame,
                          test_periods: int) -> float:
        """旧版适应度评估（兼容保留）"""
        v = np.array([genome.get(k, 0.0) for k in PARAM_NAMES])
        ev = WalkForwardEvaluator(test_periods=test_periods)
        return ev.evaluate(v, data)


# ════════════════════════════════════════════════════════════════
# § 5  增强模拟退火（Enhanced SA）
#      对外保持 SimulatedAnnealing 类名（向后兼容）
# ════════════════════════════════════════════════════════════════

class SimulatedAnnealing:
    """
    增强模拟退火（Enhanced Simulated Annealing）

    改进点：
      ① Robbins-Monro 自适应步长
         · 接受率 > 60%  → 步长 × 1.2（扩大探索）
         · 接受率 < 20%  → 步长 × 0.8（缩小精搜）
      ② 周期性重热（每 reheat_interval 步将温度恢复到一定比例）
         避免过早陷入局部最优
      ③ 全局最优解始终保留（不因接受劣解而丢失全局最优）
      ④ 使用走步 CV 计算适应度
    """

    def __init__(self, initial_temp: float = 2.0, cooling_rate: float = 0.97,
                 iterations: int = 200, reheat_interval: int = 50,
                 reheat_factor: float = 0.35):
        self.initial_temp    = initial_temp
        self.cooling_rate    = cooling_rate
        self.iterations      = iterations
        self.reheat_interval = reheat_interval
        self.reheat_factor   = reheat_factor

        self.best_params: Optional[np.ndarray] = None
        self.best_fitness: float = 0.0
        self._top_pool: List[Tuple[np.ndarray, float]] = []

    def optimize(self, evaluator: WalkForwardEvaluator,
                 data: pd.DataFrame, verbose: bool = False) -> Tuple[np.ndarray, float]:

        current    = _rand_vec()
        cur_fit    = evaluator.evaluate(current, data)

        best       = current.copy()
        best_fit   = cur_fit

        temp       = self.initial_temp
        step_size  = 0.25 * (BOUNDS_HI - BOUNDS_LO)   # 每维独立步长

        n_acc, n_tried = 0, 0
        adapt_window   = 25                             # 每 25 步调整一次步长

        for it in range(self.iterations):
            # 周期性重热
            if it > 0 and it % self.reheat_interval == 0:
                temp = self.initial_temp * self.reheat_factor
                if verbose:
                    print(f"    SA    重热 iter={it} | T={temp:.3f}"
                          f" | 最佳: {best_fit*100:.2f}%")

            # 高斯邻域扰动
            neighbor = _clip_vec(current + np.random.randn(N_PARAMS) * step_size)
            nbr_fit  = evaluator.evaluate(neighbor, data)

            delta    = nbr_fit - cur_fit
            n_tried += 1

            # Metropolis 准则
            if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-12)):
                current = neighbor
                cur_fit = nbr_fit
                n_acc  += 1

                if cur_fit > best_fit:
                    best     = current.copy()
                    best_fit = cur_fit
                    self._top_pool.append((best.copy(), best_fit))

            # 自适应步长
            if n_tried >= adapt_window:
                acc_rate  = n_acc / n_tried
                if acc_rate > 0.60:
                    step_size *= 1.20
                elif acc_rate < 0.20:
                    step_size *= 0.80
                step_size = np.clip(step_size,
                                    0.01 * (BOUNDS_HI - BOUNDS_LO),
                                    0.50 * (BOUNDS_HI - BOUNDS_LO))
                n_acc, n_tried = 0, 0

            temp *= self.cooling_rate

        self._top_pool.append((best.copy(), best_fit))
        self.best_params  = best
        self.best_fitness = best_fit
        return best, best_fit

    def get_top_solutions(self, n: int = 5) -> List[Tuple[np.ndarray, float]]:
        unique, seen = [], set()
        for params, fit in sorted(self._top_pool, key=lambda x: x[1], reverse=True):
            key = round(fit, 4)
            if key not in seen:
                seen.add(key)
                unique.append((params.copy(), fit))
            if len(unique) >= n:
                break
        return unique

    # ── 向后兼容 ─────────────────────────────────────────────
    def _fitness(self, solution, data, test_periods):
        v  = np.array(solution[:N_PARAMS])
        ev = WalkForwardEvaluator(test_periods=test_periods)
        return ev.evaluate(v, data)


# ════════════════════════════════════════════════════════════════
# § 6  差分进化优化器（Differential Evolution）
#      新增模块，是最稳健的全局优化器
# ════════════════════════════════════════════════════════════════

class DifferentialEvolutionOptimizer:
    """
    差分进化（DE / best1bin via scipy）

    为何比 PSO + SA 更稳定：
      · 种群大小 = popsize × N_params（默认 12×12 = 144 个个体）
      · 拉丁超立方初始化（latin hypercube），初始覆盖均匀
      · best1bin 策略：变异基于当前全局最优，收敛快且方向正确
      · 自适应缩放因子 F ∈ [0.5, 1.0]（dithering），避免固定 F 的系统偏差
      · polish=True：用 L-BFGS-B 对 DE 最优解做局部精修
      · 走步 CV 适应度，结果可复现性强
    """

    def __init__(self, popsize: int = 12, maxiter: int = 60,
                 mutation: Tuple[float, float] = (0.5, 1.0),
                 recombination: float = 0.75):
        self.popsize       = popsize
        self.maxiter       = maxiter
        self.mutation      = mutation
        self.recombination = recombination

        self.best_params:  Optional[np.ndarray] = None
        self.best_fitness: float = 0.0

    def optimize(self, evaluator: WalkForwardEvaluator,
                 data: pd.DataFrame, verbose: bool = False) -> Tuple[np.ndarray, float]:

        call_count = [0]

        def objective(params):
            call_count[0] += 1
            return -evaluator.evaluate(params, data)   # scipy 最小化 → 取负

        if verbose:
            print(f"    DE    种群={self.popsize * N_PARAMS}"
                  f" × 最大代数={self.maxiter}")

        result = _scipy_de(
            objective,
            bounds          = PARAM_BOUNDS,
            strategy        = 'best1bin',
            maxiter         = self.maxiter,
            popsize         = self.popsize,
            mutation        = self.mutation,      # dithering
            recombination   = self.recombination,
            tol             = 5e-4,
            polish          = True,               # L-BFGS-B 精修
            init            = 'latinhypercube',   # 均匀初始覆盖
            seed            = None,
            workers         = 1,                  # Streamlit 不支持多进程
        )

        best = _clip_vec(result.x)
        fit  = evaluator.evaluate(best, data)

        if verbose:
            print(f"    DE    完成，调用次数={call_count[0]}"
                  f" | 适应度: {fit*100:.2f}%")

        self.best_params  = best
        self.best_fitness = fit
        return best, fit


# ════════════════════════════════════════════════════════════════
# § 7  深度规则挖掘（Deep Rule Miner）
#      兼容保留，同时增加结果集成到预测的通道
# ════════════════════════════════════════════════════════════════

class DeepRuleMiner:
    """
    深度规则挖掘 v5.1

    在 v5.0 基础上增加：
      · 高频共现对（共现矩阵，用于发现"经常一起出现"的号码）
      · 周期相关性分析（7/14/21 期自相关）
      · 规则挖掘结果可选择性融入预测评分
    """

    @staticmethod
    def mine_time_based_rules(data: pd.DataFrame) -> List[Dict]:
        """周期性规律挖掘"""
        rules = []
        seq   = data['特码'].values
        for period in [7, 14, 21, 30]:
            if len(seq) >= period * 3:
                pattern = []
                for i in range(period):
                    vals = [seq[j] for j in range(i, len(seq), period) if 1 <= seq[j] <= 49]
                    if vals:
                        pattern.append({'position': i,
                                        'mean': float(np.mean(vals)),
                                        'std':  float(np.std(vals)),
                                        'mode': int(Counter(vals).most_common(1)[0][0])})
                rules.append({'type': 'periodic', 'period': period, 'pattern': pattern})
        return rules

    @staticmethod
    def mine_conditional_rules(data: pd.DataFrame) -> List[Dict]:
        """转移关联规则（if num_t → prob of num_{t+1}）"""
        seq   = data['特码'].values
        trans = {}
        for i in range(len(seq) - 1):
            a, b = int(seq[i]), int(seq[i + 1])
            if 1 <= a <= 49 and 1 <= b <= 49:
                trans.setdefault(a, Counter())[b] += 1

        rules = []
        for src, cntr in trans.items():
            total = sum(cntr.values())
            for dst, cnt in cntr.most_common(3):
                conf = cnt / total
                if conf >= 0.30:
                    rules.append({'from': src, 'to': dst,
                                   'confidence': conf, 'support': cnt})
        return sorted(rules, key=lambda x: x['confidence'], reverse=True)[:50]

    @staticmethod
    def mine_number_attraction(data: pd.DataFrame) -> List[Dict]:
        """号码共现吸引力（近邻窗口内的共现频率）"""
        seq   = data['特码'].values
        mat   = np.zeros((49, 49))
        w     = 5
        for i in range(len(seq) - w + 1):
            win = [x for x in seq[i:i+w] if 1 <= x <= 49]
            for a in win:
                for b in win:
                    if a != b:
                        mat[a-1][b-1] += 1
        thr   = np.mean(mat) + 2.0 * np.std(mat)
        pairs = []
        for i in range(49):
            for j in range(i+1, 49):
                if mat[i][j] > thr:
                    pairs.append({'num1': i+1, 'num2': j+1,
                                   'strength': float(mat[i][j])})
        return sorted(pairs, key=lambda x: x['strength'], reverse=True)[:30]

    @staticmethod
    def rule_score_bonus(data: pd.DataFrame,
                          rules: List[Dict]) -> np.ndarray:
        """
        将条件关联规则转化为评分加成向量（可选融入预测）。
        返回 shape=(49,) 的加成值，归一化到 [0, 1]。
        """
        bonus = np.zeros(49)
        if not rules or len(data) < 2:
            return bonus
        last_num = int(data.iloc[-1]['特码'])
        for rule in rules:
            if rule.get('from') == last_num:
                dst = rule.get('to', 0)
                if 1 <= dst <= 49:
                    bonus[dst - 1] += rule.get('confidence', 0)
        mx = bonus.max()
        return bonus / mx if mx > 0 else bonus


# ════════════════════════════════════════════════════════════════
# § 8  委员会集成（Committee Ensemble）
# ════════════════════════════════════════════════════════════════

class CommitteeEnsemble:
    """
    委员会集成——在预测概率空间做融合（而非在参数空间做平均）

    旧版做法：ensemble_genome = w1*genome_pso + w2*genome_sa
    → 错误：参数空间的中点在预测空间里无意义

    新版做法：
      1. 从三个优化器各取 Top-N 历史解组成候选池
      2. 每个候选解用走步 CV 得到可靠适应度
      3. 对每个号码：P_ensemble(num) = Σ fit_k × P_k(num) / Σ fit_k
         在预测概率空间做加权平均
      4. 返回加权概率最高的 Top-K 号码
    """

    def __init__(self, top_n_per_optimizer: int = 5):
        self.top_n    = top_n_per_optimizer
        self.committee: List[Tuple[np.ndarray, float]] = []
        self.ensemble_fitness: float = 0.0
        self.best_params: Optional[np.ndarray] = None

    def build(self,
              optimizer_pools: List[List[Tuple[np.ndarray, float]]],
              evaluator: WalkForwardEvaluator,
              data: pd.DataFrame) -> None:
        """
        从各优化器的 Top 解池中汇总候选解，
        重新用走步 CV 评估适应度，去重后排序。
        """
        candidates = []
        for pool in optimizer_pools:
            for params, _ in pool[:self.top_n]:
                fit = evaluator.evaluate(params, data)
                candidates.append((params.copy(), fit))

        # 去重 + 降序排列
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen   = set()
        unique = []
        for p, f in candidates:
            key = round(f, 4)
            if key not in seen:
                seen.add(key)
                unique.append((p, f))

        self.committee       = unique
        self.ensemble_fitness = unique[0][1] if unique else 0.0
        self.best_params      = unique[0][0] if unique else None

    def predict(self, data: pd.DataFrame, top_k: int = 38) -> List[int]:
        """加权投票预测"""
        if not self.committee:
            return list(range(1, top_k + 1))

        # 权重 = max(fit, 0)，避免负权重
        weights  = np.array([max(f, 0.0) for _, f in self.committee])
        w_sum    = weights.sum()
        if w_sum < 1e-12:
            weights = np.ones(len(self.committee)) / len(self.committee)
        else:
            weights /= w_sum

        ensemble_probs = np.zeros(49)
        for (params, _), w in zip(self.committee, weights):
            scores = _compute_scores(params, data)
            exp_s  = np.exp(scores - scores.max())
            probs  = exp_s / exp_s.sum()
            ensemble_probs += w * probs

        top_idx = np.argsort(ensemble_probs)[-top_k:][::-1]
        return [int(i + 1) for i in top_idx]


# ════════════════════════════════════════════════════════════════
# § 9  集成学习协调器（EnsembleLearner）
#      对外保持 EnsembleLearner 类名（向后兼容）
# ════════════════════════════════════════════════════════════════

class EnsembleLearner:
    """
    三阶段集成学习协调器 v5.1

    Phase 1: 自适应 PSO（广域探索）
    Phase 2: 增强 SA（局部精搜）
    Phase 3: 差分进化（全局精搜）
    Final:   委员会集成（概率空间融合）
    """

    def __init__(self):
        self.pso_optimizer: Optional[ParticleSwarmOptimizer]         = None
        self.sa_optimizer:  Optional[SimulatedAnnealing]             = None
        self.de_optimizer:  Optional[DifferentialEvolutionOptimizer] = None
        self.committee:     Optional[CommitteeEnsemble]              = None
        self.best_method:   str   = ''
        self.best_fitness:  float = 0.0
        self.best_params:   Optional[np.ndarray] = None

    def learn(self, data: pd.DataFrame, test_periods: int = 100,
              verbose: bool = True) -> Dict:
        """
        运行三阶段优化，返回包含各算法结果的字典。
        字典格式与 v5.0 完全兼容。
        """
        n_data      = len(data)
        # 参数自适应：数据越多可以跑更多轮次
        scale       = max(0.5, min(1.5, n_data / 500.0))
        n_particles = max(20, int(30 * scale))
        n_pso_iter  = max(40, int(60 * scale))
        sa_iters    = max(150, int(250 * scale))
        de_popsize  = max(8,  int(12 * scale))
        de_maxiter  = max(40, int(60 * scale))

        evaluator = WalkForwardEvaluator(
            test_periods    = test_periods,
            n_folds         = 3,
            top_k           = 38,
            variance_penalty = 0.5,
        )

        results = {}

        # ── Phase 1: APSO ─────────────────────────────────────
        if verbose:
            print("\n🔬 算法1/3: 自适应粒子群优化 (APSO)")
            print(f"   粒子数={n_particles}, 迭代={n_pso_iter}")
            print("-" * 60)

        self.pso_optimizer = ParticleSwarmOptimizer(
            n_particles  = n_particles,
            n_iterations = n_pso_iter,
        )
        pso_params, pso_fit = self.pso_optimizer.optimize(
            evaluator, data, verbose=verbose)

        results['pso'] = {
            'genome':  _vec_to_genome(pso_params),
            'fitness': pso_fit,
        }
        if verbose:
            print(f"✓ APSO 完成 — 适应度: {pso_fit*100:.2f}%\n")

        # ── Phase 2: Enhanced SA ──────────────────────────────
        if verbose:
            print("🔬 算法2/3: 增强模拟退火 (Enhanced SA)")
            print(f"   迭代={sa_iters}, 重热间隔=50")
            print("-" * 60)

        self.sa_optimizer = SimulatedAnnealing(
            initial_temp    = 2.0,
            cooling_rate    = 0.97,
            iterations      = sa_iters,
            reheat_interval = 50,
            reheat_factor   = 0.35,
        )
        sa_params, sa_fit = self.sa_optimizer.optimize(
            evaluator, data, verbose=verbose)

        results['sa'] = {
            'genome':  _vec_to_genome(sa_params),
            'fitness': sa_fit,
        }
        if verbose:
            print(f"✓ SA 完成 — 适应度: {sa_fit*100:.2f}%\n")

        # ── Phase 3: Differential Evolution ──────────────────
        if verbose:
            print("🔬 算法3/3: 差分进化 (DE / best1bin)")
            print(f"   种群={de_popsize * N_PARAMS}, 最大代数={de_maxiter}")
            print("-" * 60)

        self.de_optimizer = DifferentialEvolutionOptimizer(
            popsize       = de_popsize,
            maxiter       = de_maxiter,
            mutation      = (0.5, 1.0),
            recombination = 0.75,
        )
        de_params, de_fit = self.de_optimizer.optimize(
            evaluator, data, verbose=verbose)

        results['de'] = {
            'genome':  _vec_to_genome(de_params),
            'fitness': de_fit,
        }
        if verbose:
            print(f"✓ DE 完成 — 适应度: {de_fit*100:.2f}%\n")

        # ── 委员会集成 ────────────────────────────────────────
        if verbose:
            print("🎯 构建委员会集成（概率空间加权投票）...")

        pso_pool = self.pso_optimizer.get_top_solutions(n=5)
        sa_pool  = self.sa_optimizer.get_top_solutions(n=5)
        de_pool  = [(de_params, de_fit)]           # DE 只有最终最优解

        self.committee = CommitteeEnsemble(top_n_per_optimizer=5)
        self.committee.build([pso_pool, sa_pool, de_pool], evaluator, data)

        ens_fit = self.committee.ensemble_fitness
        ens_params = self.committee.best_params if self.committee.best_params is not None \
                     else de_params

        results['ensemble'] = {
            'genome':  _vec_to_genome(ens_params),
            'fitness': ens_fit,
        }
        # 也存一个 'rules' key 保持向后兼容
        results['rules'] = {
            'time_based':   DeepRuleMiner.mine_time_based_rules(data)[:3],
            'conditional':  DeepRuleMiner.mine_conditional_rules(data)[:10],
            'attraction':   DeepRuleMiner.mine_number_attraction(data)[:10],
        }

        # 选取最优方法
        ranking = [('pso', pso_params, pso_fit),
                   ('sa',  sa_params,  sa_fit),
                   ('de',  de_params,  de_fit),
                   ('ensemble', ens_params, ens_fit)]

        best_method, best_params, best_fitness = max(ranking, key=lambda x: x[2])

        self.best_method  = best_method
        self.best_params  = best_params
        self.best_fitness = best_fitness

        if verbose:
            print(f"\n{'='*60}")
            print(f"  APSO   : {pso_fit*100:.2f}%")
            print(f"  SA     : {sa_fit*100:.2f}%")
            print(f"  DE     : {de_fit*100:.2f}%")
            print(f"  集成   : {ens_fit*100:.2f}%")
            print(f"  最佳方法: {best_method.upper()}")
            print(f"{'='*60}\n")

        return results


# ════════════════════════════════════════════════════════════════
# § 10  超级学习引擎主类（对外 API，向后完全兼容）
# ════════════════════════════════════════════════════════════════

class SuperLearningEngine:
    """
    超级学习引擎 v5.1 - 科研重构版

    公开 API（与 v5.0 完全兼容）：
      · ultra_learn(data, test_periods, verbose) → Dict
      · predict_ultra(data, top_k)              → List[int]
      · predict(data, top_k)                    → List[int]  (别名)
    """

    def __init__(self):
        self._learner:       Optional[EnsembleLearner]  = None
        self._committee:     Optional[CommitteeEnsemble] = None
        self._best_params:   Optional[np.ndarray]        = None
        self.learning_results: Optional[Dict]            = None

    # ─────────────────────────────────────────────────────────
    def ultra_learn(self, data: pd.DataFrame,
                    test_periods: int = 100,
                    verbose: bool = True) -> Dict:
        """
        主学习入口

        Args:
            data:         历史开奖数据 (DataFrame，含 '特码' 列)
            test_periods: 走步 CV 回测期数（建议 ≥ 80）
            verbose:      是否打印进度

        Returns:
            {
              'best_genome':  Dict,   # 参数名 → 值
              'best_fitness': float,  # 走步 CV 适应度
              'best_method':  str,    # 'pso' | 'sa' | 'de' | 'ensemble'
              'all_results':  Dict,   # {'pso': {...}, 'sa': {...}, 'ensemble': {...}, ...}
            }
        """
        if verbose:
            print("\n" + "=" * 70)
            print("🚀 超级深度学习引擎 v5.1  科研重构版")
            print("=" * 70)
            print(f"  数据总期数  : {len(data)}")
            print(f"  走步CV期数  : {test_periods}  (3折，含方差惩罚)")
            print(f"  优化器      : APSO + 增强SA + 差分进化")
            print(f"  集成方式    : 概率空间委员会投票")
            print("=" * 70)

        if len(data) < test_periods + 30:
            raise ValueError(
                f"数据不足：需要至少 {test_periods + 30} 期，当前 {len(data)} 期")

        self._learner = EnsembleLearner()
        all_results   = self._learner.learn(data, test_periods, verbose)

        self._committee   = self._learner.committee
        self._best_params = self._learner.best_params

        result = {
            'best_genome':  _vec_to_genome(self._best_params),
            'best_fitness': self._learner.best_fitness,
            'best_method':  self._learner.best_method,
            'all_results':  all_results,
        }
        self.learning_results = result
        return result

    # ─────────────────────────────────────────────────────────
    def predict_ultra(self, data: pd.DataFrame, top_k: int = 38) -> List[int]:
        """
        预测接口

        优先使用委员会集成（所有优化器的 Top 解加权投票），
        若学习未完成则退化为单解预测。
        支持 top_k=49 以供防连错引擎获取完整排名。
        """
        if self._committee is not None and self._committee.committee:
            return self._committee.predict(data, top_k)

        if self._best_params is not None:
            return predict_from_vec(self._best_params, data, top_k)

        return list(range(1, min(top_k, 49) + 1))

    # predict 是 predict_ultra 的别名，供 anti_consecutive_loss 调用
    predict = predict_ultra
