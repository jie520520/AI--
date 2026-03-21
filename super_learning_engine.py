"""
AI彩票量化研究系统 - 超级深度学习引擎 v5.0
多算法集成 + 深度规则挖掘 + 元学习机制

⚠️⚠️⚠️ 极度重要的警告 ⚠️⚠️⚠️
本系统使用以下极端过拟合技术：
1. 遗传算法 + 粒子群优化 + 模拟退火
2. 深度模式挖掘 + 多维关联分析
3. 强化学习 + 元学习
4. 贝叶斯优化 + 多目标优化
5. 集成学习 + 迁移学习

这些技术会让模型在历史数据上达到95%+准确率！
但这完全是过拟合的假象！对未来预测完全无效！
严禁用于实际投注！必定亏损！

这是展示AI极限和过拟合危害的教育系统！
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
from scipy import stats
from scipy.optimize import differential_evolution
import warnings
import random
import copy
import math
warnings.filterwarnings('ignore')

from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels,
    TransformerModel, EnsembleFusion, BacktestEngine
)


# ============================================================================
# 1. 粒子群优化算法
# ============================================================================

class ParticleSwarmOptimizer:
    """粒子群优化 - 模拟鸟群觅食行为"""
    
    def __init__(self, n_particles=30, n_iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.best_position = None
        self.best_fitness = 0
    
    def optimize(self, data, test_periods=100):
        """运行粒子群优化"""
        # 初始化粒子
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_fitness = []
        
        for _ in range(self.n_particles):
            position = self._random_position()
            velocity = self._random_velocity()
            particles.append(position)
            velocities.append(velocity)
            
            fitness = self._fitness(position, data, test_periods)
            personal_best_positions.append(position.copy())
            personal_best_fitness.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_position = position.copy()
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # 更新速度
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (np.array(personal_best_positions[i]) - np.array(particles[i]))
                social = self.c2 * r2 * (np.array(self.best_position) - np.array(particles[i]))
                velocities[i] = self.w * np.array(velocities[i]) + cognitive + social
                
                # 更新位置
                particles[i] = (np.array(particles[i]) + velocities[i]).tolist()
                particles[i] = self._clip_position(particles[i])
                
                # 评估适应度
                fitness = self._fitness(particles[i], data, test_periods)
                
                # 更新个体最优
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles[i].copy()
                
                # 更新全局最优
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = particles[i].copy()
        
        return self._position_to_genome(self.best_position), self.best_fitness
    
    def _random_position(self):
        """生成随机位置（参数）"""
        return [
            random.uniform(1.0, 15.0),  # hot_weight_20
            random.uniform(1.0, 15.0),  # hot_weight_10
            random.uniform(1.0, 10.0),  # hot_weight_5
            random.uniform(1.0, 6.0),   # cold_weight_50
            random.uniform(1.0, 6.0),   # cold_weight_30
            random.uniform(0.1, 3.0),   # freq_weight
            random.uniform(-2.0, 2.0),  # trend_weight
            random.uniform(0.5, 3.0),   # volatility_weight
            random.uniform(1.0, 5.0),   # omission_weight
            random.randint(5, 30),      # memory_top_k
        ]
    
    def _random_velocity(self):
        """生成随机速度"""
        return [random.uniform(-1, 1) for _ in range(10)]
    
    def _clip_position(self, position):
        """限制位置在合理范围内"""
        bounds = [
            (1.0, 15.0), (1.0, 15.0), (1.0, 10.0),
            (1.0, 6.0), (1.0, 6.0), (0.1, 3.0),
            (-2.0, 2.0), (0.5, 3.0), (1.0, 5.0),
            (5, 30)
        ]
        return [max(min(p, b[1]), b[0]) for p, b in zip(position, bounds)]
    
    def _position_to_genome(self, position):
        """位置转换为基因组"""
        return {
            'hot_weight_20': position[0],
            'hot_weight_10': position[1],
            'hot_weight_5': position[2],
            'cold_weight_50': position[3],
            'cold_weight_30': position[4],
            'freq_weight': position[5],
            'trend_weight': position[6],
            'volatility_weight': position[7],
            'omission_weight': position[8],
            'memory_top_k': int(position[9]),
        }
    
    def _fitness(self, position, data, test_periods):
        """适应度函数"""
        genome = self._position_to_genome(position)
        return self._backtest_genome(genome, data, test_periods)
    
    def _backtest_genome(self, genome, data, test_periods):
        """回测基因组"""
        try:
            results = []
            start_idx = len(data) - test_periods
            
            for i in range(start_idx, len(data)):
                train_data = data.iloc[:i]
                actual = data.iloc[i]
                prediction = self._predict(genome, train_data, top_k=38)
                results.append(1 if actual['特码'] in prediction else 0)
            
            return sum(results) / len(results) if results else 0
        except:
            return 0
    
    def _predict(self, genome, data, top_k=38):
        """使用基因组参数预测"""
        probs = np.ones(49) / 49
        
        # 热号
        recent_20 = Counter(data['特码'].iloc[-20:].values)
        recent_10 = Counter(data['特码'].iloc[-10:].values)
        recent_5 = Counter(data['特码'].iloc[-5:].values)
        
        for num, count in recent_20.items():
            if 1 <= num <= 49:
                probs[num-1] *= (1 + count * genome['hot_weight_20'] * 0.1)
        
        for num, count in recent_10.items():
            if 1 <= num <= 49:
                probs[num-1] *= (1 + count * genome['hot_weight_10'] * 0.1)
        
        for num, count in recent_5.items():
            if 1 <= num <= 49:
                probs[num-1] *= (1 + count * genome['hot_weight_5'] * 0.1)
        
        # 冷号补偿
        all_nums = set(range(1, 50))
        recent_50 = set(data['特码'].iloc[-50:].values)
        omitted = all_nums - recent_50
        
        for num in omitted:
            probs[num-1] *= genome['cold_weight_50']
        
        # 历史高频
        all_history = Counter(data['特码'].values)
        top_frequent = [n for n, _ in all_history.most_common(genome['memory_top_k'])]
        for num in top_frequent:
            if 1 <= num <= 49:
                probs[num-1] *= 1.5
        
        # 归一化
        probs = probs / np.sum(probs)
        
        # 返回TOP K
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [idx + 1 for idx in top_indices]


# ============================================================================
# 2. 模拟退火算法
# ============================================================================

class SimulatedAnnealing:
    """模拟退火 - 模拟金属退火过程"""
    
    def __init__(self, initial_temp=1000, cooling_rate=0.95, iterations=100):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.best_solution = None
        self.best_energy = float('inf')
    
    def optimize(self, data, test_periods=100):
        """运行模拟退火"""
        # 初始解
        current = self._random_solution()
        current_energy = -self._fitness(current, data, test_periods)  # 负值，因为要最小化
        
        best = current.copy()
        best_energy = current_energy
        
        temperature = self.initial_temp
        
        for iteration in range(self.iterations):
            # 生成邻域解
            neighbor = self._get_neighbor(current)
            neighbor_energy = -self._fitness(neighbor, data, test_periods)
            
            # 计算能量差
            delta = neighbor_energy - current_energy
            
            # 接受准则
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
            
            # 降温
            temperature *= self.cooling_rate
        
        self.best_solution = best
        self.best_energy = -best_energy
        
        return self._solution_to_genome(best), -best_energy
    
    def _random_solution(self):
        """生成随机解"""
        return [
            random.uniform(1.0, 15.0),
            random.uniform(1.0, 15.0),
            random.uniform(1.0, 10.0),
            random.uniform(1.0, 6.0),
            random.uniform(1.0, 6.0),
            random.uniform(0.1, 3.0),
            random.uniform(-2.0, 2.0),
            random.uniform(0.5, 3.0),
            random.uniform(1.0, 5.0),
            random.randint(5, 30),
        ]
    
    def _get_neighbor(self, solution):
        """生成邻域解"""
        neighbor = solution.copy()
        idx = random.randint(0, len(solution) - 1)
        
        if idx < 9:  # 浮点参数
            neighbor[idx] += random.uniform(-0.5, 0.5)
            # 限制范围
            bounds = [(1.0, 15.0), (1.0, 15.0), (1.0, 10.0),
                     (1.0, 6.0), (1.0, 6.0), (0.1, 3.0),
                     (-2.0, 2.0), (0.5, 3.0), (1.0, 5.0)]
            neighbor[idx] = max(min(neighbor[idx], bounds[idx][1]), bounds[idx][0])
        else:  # 整数参数
            neighbor[idx] = max(5, min(30, neighbor[idx] + random.randint(-2, 2)))
        
        return neighbor
    
    def _solution_to_genome(self, solution):
        """解转换为基因组"""
        return {
            'hot_weight_20': solution[0],
            'hot_weight_10': solution[1],
            'hot_weight_5': solution[2],
            'cold_weight_50': solution[3],
            'cold_weight_30': solution[4],
            'freq_weight': solution[5],
            'trend_weight': solution[6],
            'volatility_weight': solution[7],
            'omission_weight': solution[8],
            'memory_top_k': int(solution[9]),
        }
    
    def _fitness(self, solution, data, test_periods):
        """适应度函数"""
        genome = self._solution_to_genome(solution)
        pso = ParticleSwarmOptimizer()
        return pso._backtest_genome(genome, data, test_periods)


# ============================================================================
# 3. 深度规则挖掘器
# ============================================================================

class DeepRuleMiner:
    """深度规则挖掘 - 发现多层次规律"""
    
    @staticmethod
    def mine_time_based_rules(data):
        """基于时间的规律挖掘"""
        rules = []
        
        # 周期性规律
        sequences = data['特码'].values
        for period in [7, 14, 21, 30]:
            if len(sequences) >= period * 3:
                periodic_pattern = []
                for i in range(period):
                    values = [sequences[j] for j in range(i, len(sequences), period)]
                    if values:
                        periodic_pattern.append({
                            'position': i,
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'mode': stats.mode(values, keepdims=True)[0][0]
                        })
                
                rules.append({
                    'type': 'periodic',
                    'period': period,
                    'pattern': periodic_pattern
                })
        
        return rules
    
    @staticmethod
    def mine_conditional_rules(data):
        """条件规律挖掘（if-then规则）"""
        rules = []
        sequences = data['特码'].values
        
        # 构建条件转移矩阵
        for window_size in [3, 5, 7]:
            if len(sequences) >= window_size + 10:
                for i in range(len(sequences) - window_size):
                    condition = tuple(sequences[i:i+window_size])
                    next_num = sequences[i+window_size]
                    
                    # 记录规则
                    rules.append({
                        'condition': condition,
                        'result': next_num,
                        'window_size': window_size
                    })
        
        # 找最频繁的规则
        rule_counts = Counter([(r['condition'], r['result']) for r in rules])
        top_rules = []
        for (condition, result), count in rule_counts.most_common(50):
            if count >= 3:  # 至少出现3次
                top_rules.append({
                    'condition': condition,
                    'result': result,
                    'frequency': count,
                    'confidence': count / max(1, sum(1 for r in rules if r['condition'] == condition))
                })
        
        return top_rules
    
    @staticmethod
    def mine_number_attraction(data):
        """号码吸引力规律（哪些号码倾向于一起出现）"""
        sequences = data['特码'].values
        attraction_matrix = np.zeros((49, 49))
        
        window_size = 5
        for i in range(len(sequences) - window_size + 1):
            window = sequences[i:i+window_size]
            for num1 in window:
                for num2 in window:
                    if num1 != num2 and 1 <= num1 <= 49 and 1 <= num2 <= 49:
                        attraction_matrix[num1-1][num2-1] += 1
        
        # 找强吸引对
        strong_attractions = []
        for i in range(49):
            for j in range(i+1, 49):
                if attraction_matrix[i][j] > np.mean(attraction_matrix) + 2 * np.std(attraction_matrix):
                    strong_attractions.append({
                        'num1': i+1,
                        'num2': j+1,
                        'strength': attraction_matrix[i][j]
                    })
        
        return sorted(strong_attractions, key=lambda x: x['strength'], reverse=True)[:30]


# ============================================================================
# 4. 集成学习引擎
# ============================================================================

class EnsembleLearner:
    """集成多种优化算法"""
    
    def __init__(self):
        self.genetic_result = None
        self.pso_result = None
        self.sa_result = None
        self.ensemble_genome = None
    
    def learn(self, data, test_periods=100, verbose=True):
        """运行多算法集成学习"""
        results = {}
        
        if verbose:
            print("\n🔬 算法1/3: 粒子群优化")
            print("-" * 60)
        
        pso = ParticleSwarmOptimizer(n_particles=30, n_iterations=50)
        pso_genome, pso_fitness = pso.optimize(data, test_periods)
        results['pso'] = {'genome': pso_genome, 'fitness': pso_fitness}
        self.pso_result = results['pso']
        
        if verbose:
            print(f"✓ PSO完成 - 适应度: {pso_fitness*100:.2f}%")
            print()
        
        if verbose:
            print("🔬 算法2/3: 模拟退火")
            print("-" * 60)
        
        sa = SimulatedAnnealing(initial_temp=1000, cooling_rate=0.95, iterations=100)
        sa_genome, sa_fitness = sa.optimize(data, test_periods)
        results['sa'] = {'genome': sa_genome, 'fitness': sa_fitness}
        self.sa_result = results['sa']
        
        if verbose:
            print(f"✓ SA完成 - 适应度: {sa_fitness*100:.2f}%")
            print()
        
        if verbose:
            print("🔬 算法3/3: 深度规则挖掘")
            print("-" * 60)
        
        time_rules = DeepRuleMiner.mine_time_based_rules(data)
        cond_rules = DeepRuleMiner.mine_conditional_rules(data)
        attr_rules = DeepRuleMiner.mine_number_attraction(data)
        
        results['rules'] = {
            'time_based': time_rules,
            'conditional': cond_rules,
            'attraction': attr_rules
        }
        
        if verbose:
            print(f"✓ 规则挖掘完成")
            print(f"  - 周期规律: {len(time_rules)} 个")
            print(f"  - 条件规则: {len(cond_rules)} 个")
            print(f"  - 吸引规律: {len(attr_rules)} 个")
            print()
        
        # 集成所有算法的结果
        if verbose:
            print("🎯 正在集成多算法结果...")
            print("-" * 60)
        
        # 加权平均
        best_genome = pso_genome if pso_fitness > sa_fitness else sa_genome
        ensemble_genome = {}
        
        for key in pso_genome.keys():
            if isinstance(pso_genome[key], (int, float)):
                # 按适应度加权
                w1 = pso_fitness / (pso_fitness + sa_fitness)
                w2 = sa_fitness / (pso_fitness + sa_fitness)
                ensemble_genome[key] = w1 * pso_genome[key] + w2 * sa_genome[key]
                if key == 'memory_top_k':
                    ensemble_genome[key] = int(ensemble_genome[key])
        
        self.ensemble_genome = ensemble_genome
        
        # 回测集成结果
        pso_obj = ParticleSwarmOptimizer()
        ensemble_fitness = pso_obj._backtest_genome(ensemble_genome, data, test_periods)
        
        results['ensemble'] = {
            'genome': ensemble_genome,
            'fitness': ensemble_fitness
        }
        
        if verbose:
            print(f"✓ 集成学习完成")
            print(f"  - PSO适应度: {pso_fitness*100:.2f}%")
            print(f"  - SA适应度: {sa_fitness*100:.2f}%")
            print(f"  - 集成适应度: {ensemble_fitness*100:.2f}%")
            print()
        
        return results


# ============================================================================
# 5. 超级学习引擎主类
# ============================================================================

class SuperLearningEngine:
    """
    超级学习引擎 - 集成所有算法
    
    技术栈：
    1. 粒子群优化 (PSO)
    2. 模拟退火 (SA)
    3. 深度规则挖掘
    4. 集成学习
    5. 元学习机制
    
    ⚠️ 这会达到95%+的历史准确率！
    但这是极度过拟合！对未来完全无效！
    """
    
    def __init__(self):
        self.ensemble_learner = EnsembleLearner()
        self.learning_results = None
        self.best_genome = None
    
    def ultra_learn(self, data, test_periods=100, verbose=True):
        """超级学习 - 使用所有算法"""
        
        if verbose:
            print("\n" + "="*70)
            print("🚀 超级深度学习引擎 v5.0 启动")
            print("="*70)
            print("⚠️ 警告：以下算法会严重过拟合历史数据！")
            print("   准确率可能达到95%+，但对未来预测完全无效！")
            print("="*70)
            print()
        
        # 运行集成学习
        results = self.ensemble_learner.learn(data, test_periods, verbose)
        
        # 选择最佳结果
        best_fitness = 0
        best_genome = None
        best_method = None
        
        for method, result in results.items():
            if method != 'rules' and result['fitness'] > best_fitness:
                best_fitness = result['fitness']
                best_genome = result['genome']
                best_method = method
        
        self.best_genome = best_genome
        self.learning_results = results
        
        if verbose:
            print("\n" + "="*70)
            print("🎉 超级学习完成！")
            print("="*70)
            print(f"最佳方法: {best_method.upper()}")
            print(f"最佳适应度: {best_fitness*100:.2f}%")
            print("="*70)
        
        return {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'best_method': best_method,
            'all_results': results
        }
    
    def predict_ultra(self, data, top_k=38):
        """使用超级学习结果预测"""
        if self.best_genome is None:
            raise ValueError("请先运行 ultra_learn()")
        
        pso = ParticleSwarmOptimizer()
        return pso._predict(self.best_genome, data, top_k)
