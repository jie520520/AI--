"""
AI彩票量化研究系统 - 自主学习引擎 v4.0
使用遗传算法、模式挖掘、规则学习自动优化

⚠️ 极其重要的警告：
本模块使用进化算法在历史数据中寻找"规律"，这些规律是过拟合的假象！
在真正随机的数据中，任何发现的"规律"都不具有预测能力！
严禁用于实际投注！

学习策略：
1. 遗传算法优化特征权重（100代进化）
2. 模式挖掘算法（频繁项集、关联规则）
3. 自适应规则学习（强化学习式调参）
4. 多代进化找最佳参数组合
5. 回测验证并持续优化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
from scipy import stats
import warnings
import random
import copy
warnings.filterwarnings('ignore')

from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels,
    TransformerModel, EnsembleFusion, BacktestEngine
)


# ============================================================================
# 遗传算法优化引擎
# ============================================================================

class GeneticOptimizer:
    """遗传算法优化器 - 自动寻找最优参数"""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_genome = None
        self.best_fitness = 0
        self.evolution_history = []
    
    def create_genome(self):
        """创建一个基因组（参数集）"""
        return {
            # 热号权重 (1.0-10.0)
            'hot_weight_20': random.uniform(1.0, 10.0),
            'hot_weight_10': random.uniform(1.0, 10.0),
            'hot_weight_5': random.uniform(1.0, 10.0),
            
            # 冷号补偿权重 (1.0-5.0)
            'cold_weight_50': random.uniform(1.0, 5.0),
            'cold_weight_30': random.uniform(1.0, 5.0),
            'cold_weight_20': random.uniform(1.0, 5.0),
            
            # 频率统计权重 (0.1-2.0)
            'freq_weight_100': random.uniform(0.1, 2.0),
            'freq_weight_50': random.uniform(0.1, 2.0),
            'freq_weight_20': random.uniform(0.1, 2.0),
            
            # 趋势权重 (-1.0-1.0)
            'trend_weight': random.uniform(-1.0, 1.0),
            
            # 波动权重 (0.5-2.0)
            'volatility_weight': random.uniform(0.5, 2.0),
            
            # RSI权重 (0.5-2.0)
            'rsi_weight': random.uniform(0.5, 2.0),
            
            # 遗漏权重 (1.0-4.0)
            'omission_weight': random.uniform(1.0, 4.0),
            
            # 历史记忆数量 (5-25)
            'memory_top_k': random.randint(5, 25),
            
            # 模式长度 (2-10)
            'pattern_length': random.randint(2, 10),
        }
    
    def fitness(self, genome, data, test_periods=50):
        """
        适应度函数 - 在历史数据上的回测准确率
        ⚠️ 这会导致严重过拟合！
        """
        try:
            # 使用该基因组进行回测
            results = []
            start_idx = len(data) - test_periods
            
            for i in range(start_idx, len(data)):
                train_data = data.iloc[:i]
                actual = data.iloc[i]
                
                # 使用基因组参数进行预测
                prediction = self._predict_with_genome(genome, train_data)
                
                # 检查是否命中
                if actual['特码'] in prediction:
                    results.append(1)
                else:
                    results.append(0)
            
            # 返回准确率
            accuracy = sum(results) / len(results) if results else 0
            return accuracy
            
        except Exception as e:
            return 0.0
    
    def _predict_with_genome(self, genome, data, top_k=38):
        """使用基因组参数进行预测"""
        # 初始化概率数组
        probs = np.ones(49) / 49
        
        # 1. 热号加权
        recent_20 = data['特码'].iloc[-20:].values
        recent_10 = data['特码'].iloc[-10:].values
        recent_5 = data['特码'].iloc[-5:].values
        
        for num in recent_20:
            if 1 <= num <= 49:
                probs[num-1] *= genome['hot_weight_20']
        
        for num in recent_10:
            if 1 <= num <= 49:
                probs[num-1] *= genome['hot_weight_10']
        
        for num in recent_5:
            if 1 <= num <= 49:
                probs[num-1] *= genome['hot_weight_5']
        
        # 2. 冷号补偿（遗漏分析）
        all_numbers = set(range(1, 50))
        recent_50_set = set(data['特码'].iloc[-50:].values)
        recent_30_set = set(data['特码'].iloc[-30:].values)
        recent_20_set = set(data['特码'].iloc[-20:].values)
        
        omitted_50 = all_numbers - recent_50_set
        omitted_30 = all_numbers - recent_30_set
        omitted_20 = all_numbers - recent_20_set
        
        for num in omitted_50:
            probs[num-1] *= genome['cold_weight_50']
        
        for num in omitted_30:
            probs[num-1] *= genome['cold_weight_30']
        
        for num in omitted_20:
            probs[num-1] *= genome['cold_weight_20']
        
        # 3. 频率统计
        recent_100 = data['特码'].iloc[-100:] if len(data) >= 100 else data['特码']
        recent_50 = data['特码'].iloc[-50:]
        recent_20_freq = data['特码'].iloc[-20:]
        
        freq_100 = Counter(recent_100.values)
        freq_50 = Counter(recent_50.values)
        freq_20 = Counter(recent_20_freq.values)
        
        for num, count in freq_100.items():
            if 1 <= num <= 49:
                probs[num-1] *= (1 + count * 0.01 * genome['freq_weight_100'])
        
        for num, count in freq_50.items():
            if 1 <= num <= 49:
                probs[num-1] *= (1 + count * 0.02 * genome['freq_weight_50'])
        
        for num, count in freq_20.items():
            if 1 <= num <= 49:
                probs[num-1] *= (1 + count * 0.05 * genome['freq_weight_20'])
        
        # 4. 历史记忆（全局高频）
        all_history = Counter(data['特码'].values)
        top_memory = [num for num, _ in all_history.most_common(int(genome['memory_top_k']))]
        for num in top_memory:
            if 1 <= num <= 49:
                probs[num-1] *= 1.5
        
        # 归一化
        probs = probs / np.sum(probs)
        
        # 返回TOP K
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [idx + 1 for idx in top_indices]
    
    def crossover(self, parent1, parent2):
        """交叉操作 - 生成子代"""
        child = {}
        for key in parent1.keys():
            # 随机从父母中选择
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def mutate(self, genome):
        """变异操作"""
        mutated = copy.deepcopy(genome)
        for key in mutated.keys():
            if random.random() < self.mutation_rate:
                if 'weight' in key:
                    if 'hot' in key:
                        mutated[key] = random.uniform(1.0, 10.0)
                    elif 'cold' in key:
                        mutated[key] = random.uniform(1.0, 5.0)
                    elif 'freq' in key:
                        mutated[key] = random.uniform(0.1, 2.0)
                    elif 'trend' in key:
                        mutated[key] = random.uniform(-1.0, 1.0)
                    elif 'volatility' in key or 'rsi' in key:
                        mutated[key] = random.uniform(0.5, 2.0)
                    elif 'omission' in key:
                        mutated[key] = random.uniform(1.0, 4.0)
                elif 'top_k' in key:
                    mutated[key] = random.randint(5, 25)
                elif 'length' in key:
                    mutated[key] = random.randint(2, 10)
        return mutated
    
    def evolve(self, data, test_periods=50, verbose=True):
        """
        运行遗传算法进化
        ⚠️ 这会在历史数据上过拟合！
        """
        if verbose:
            print("\n" + "="*60)
            print("🧬 启动遗传算法自主学习引擎")
            print("="*60)
            print(f"种群大小: {self.population_size}")
            print(f"进化代数: {self.generations}")
            print(f"变异率: {self.mutation_rate}")
            print(f"回测期数: {test_periods}")
            print("="*60)
            print()
        
        # 初始化种群
        population = [self.create_genome() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = []
            for genome in population:
                score = self.fitness(genome, data, test_periods)
                fitness_scores.append((genome, score))
            
            # 排序
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 记录最佳个体
            best_genome, best_score = fitness_scores[0]
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best_score,
                'avg_fitness': np.mean([s for _, s in fitness_scores]),
                'best_genome': copy.deepcopy(best_genome)
            })
            
            if best_score > self.best_fitness:
                self.best_fitness = best_score
                self.best_genome = copy.deepcopy(best_genome)
            
            if verbose and generation % 10 == 0:
                print(f"第{generation:3d}代 | 最佳适应度: {best_score*100:6.2f}% | "
                      f"平均适应度: {np.mean([s for _, s in fitness_scores])*100:6.2f}%")
            
            # 选择（保留前50%）
            survivors = [genome for genome, _ in fitness_scores[:self.population_size//2]]
            
            # 生成新一代
            new_population = survivors.copy()
            
            while len(new_population) < self.population_size:
                # 随机选择两个父母
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 变异
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        if verbose:
            print()
            print("="*60)
            print("🎉 进化完成！")
            print(f"最终最佳适应度: {self.best_fitness*100:.2f}%")
            print("="*60)
            print()
            print("最优参数组合:")
            for key, value in self.best_genome.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:6.3f}")
                else:
                    print(f"  {key:20s}: {value}")
            print("="*60)
        
        return self.best_genome, self.best_fitness


# ============================================================================
# 模式挖掘引擎
# ============================================================================

class PatternMiner:
    """模式挖掘引擎 - 发现历史数据中的频繁模式"""
    
    @staticmethod
    def find_frequent_patterns(data, min_support=0.1, max_length=5):
        """
        发现频繁模式
        ⚠️ 在随机数据中的模式是偶然性，不是规律！
        """
        sequences = data['特码'].values
        patterns = defaultdict(int)
        
        # 查找不同长度的模式
        for length in range(2, max_length + 1):
            for i in range(len(sequences) - length + 1):
                pattern = tuple(sequences[i:i+length])
                patterns[pattern] += 1
        
        # 过滤低支持度的模式
        total_sequences = len(sequences)
        frequent_patterns = {
            pattern: count
            for pattern, count in patterns.items()
            if count / total_sequences >= min_support
        }
        
        return sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def find_association_rules(data, min_confidence=0.5):
        """
        发现关联规则 (如果A出现，B也会出现)
        ⚠️ 在随机数据中的关联是虚假的！
        """
        sequences = data['特码'].values
        rules = []
        
        # 构建转移矩阵
        transition_matrix = defaultdict(Counter)
        
        for i in range(len(sequences) - 1):
            current = sequences[i]
            next_num = sequences[i + 1]
            transition_matrix[current][next_num] += 1
        
        # 提取高置信度规则
        for from_num, to_counter in transition_matrix.items():
            total = sum(to_counter.values())
            for to_num, count in to_counter.items():
                confidence = count / total
                if confidence >= min_confidence:
                    rules.append({
                        'from': from_num,
                        'to': to_num,
                        'confidence': confidence,
                        'support': count
                    })
        
        return sorted(rules, key=lambda x: x['confidence'], reverse=True)
    
    @staticmethod
    def find_number_groups(data, n_clusters=5):
        """
        发现号码聚类（哪些号码经常一起出现）
        ⚠️ 在随机数据中的聚类是偶然的！
        """
        # 构建共现矩阵
        co_occurrence = np.zeros((49, 49))
        
        sequences = data['特码'].values
        window_size = 10
        
        for i in range(len(sequences) - window_size + 1):
            window = sequences[i:i+window_size]
            for num1 in window:
                for num2 in window:
                    if num1 != num2 and 1 <= num1 <= 49 and 1 <= num2 <= 49:
                        co_occurrence[num1-1][num2-1] += 1
        
        # 找出共现频率最高的号码对
        pairs = []
        for i in range(49):
            for j in range(i+1, 49):
                if co_occurrence[i][j] > 0:
                    pairs.append({
                        'num1': i+1,
                        'num2': j+1,
                        'frequency': co_occurrence[i][j]
                    })
        
        return sorted(pairs, key=lambda x: x['frequency'], reverse=True)


# ============================================================================
# 自适应学习引擎
# ============================================================================

class AdaptiveLearner:
    """自适应学习引擎 - 根据回测结果持续优化"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.parameters = {
            'hot_multiplier': 2.0,
            'cold_multiplier': 1.5,
            'memory_strength': 1.3,
        }
        self.performance_history = []
    
    def update_parameters(self, performance):
        """
        根据表现更新参数（类似强化学习）
        ⚠️ 这只会让模型更加过拟合历史数据！
        """
        self.performance_history.append(performance)
        
        # 如果表现提升，增强当前参数
        if len(self.performance_history) >= 2:
            if performance > self.performance_history[-2]:
                # 表现提升，增强参数
                for key in self.parameters:
                    self.parameters[key] *= (1 + self.learning_rate)
            else:
                # 表现下降，减弱参数
                for key in self.parameters:
                    self.parameters[key] *= (1 - self.learning_rate)
        
        return self.parameters
    
    def predict_adaptive(self, data, genome, top_k=38):
        """使用自适应参数进行预测"""
        # 基础预测
        optimizer = GeneticOptimizer()
        base_prediction = optimizer._predict_with_genome(genome, data, top_k)
        
        # 应用自适应调整
        # ... (可以根据实时表现动态调整)
        
        return base_prediction


# ============================================================================
# 主学习引擎
# ============================================================================

class SelfLearningEngine:
    """
    主学习引擎 - 整合所有学习算法
    
    功能：
    1. 遗传算法优化参数
    2. 模式挖掘发现规律
    3. 自适应学习持续优化
    4. 回测验证效果
    
    ⚠️ 警告：这是一个强大的过拟合系统！
    """
    
    def __init__(self):
        self.genetic_optimizer = None
        self.pattern_miner = PatternMiner()
        self.adaptive_learner = AdaptiveLearner()
        self.best_genome = None
        self.discovered_patterns = None
        self.learning_results = {}
    
    def auto_learn(self, data, test_periods=100, generations=100, 
                   population_size=50, verbose=True):
        """
        自动学习 - 从数据中发现"最优规律"
        
        ⚠️ 警告：这会严重过拟合历史数据！
        发现的"规律"对未来预测无效！
        """
        if verbose:
            print("\n" + "="*70)
            print("🤖 启动自主学习引擎 - AI自动发现规律")
            print("="*70)
            print("⚠️  警告：以下过程会在历史数据中寻找'规律'")
            print("    这些'规律'是过拟合的假象，对未来预测无效！")
            print("="*70)
            print()
        
        # 阶段1：遗传算法优化
        if verbose:
            print("📊 阶段1/3: 遗传算法参数优化")
            print("-" * 70)
        
        self.genetic_optimizer = GeneticOptimizer(
            population_size=population_size,
            generations=generations,
            mutation_rate=0.1
        )
        
        self.best_genome, best_fitness = self.genetic_optimizer.evolve(
            data, test_periods, verbose=verbose
        )
        
        self.learning_results['genetic_fitness'] = best_fitness
        
        # 阶段2：模式挖掘
        if verbose:
            print("\n📊 阶段2/3: 模式挖掘")
            print("-" * 70)
        
        frequent_patterns = self.pattern_miner.find_frequent_patterns(
            data, min_support=0.05, max_length=5
        )
        
        association_rules = self.pattern_miner.find_association_rules(
            data, min_confidence=0.3
        )
        
        number_groups = self.pattern_miner.find_number_groups(data, n_clusters=5)
        
        self.discovered_patterns = {
            'frequent_patterns': frequent_patterns[:10],
            'association_rules': association_rules[:10],
            'number_groups': number_groups[:20]
        }
        
        if verbose:
            print(f"发现频繁模式: {len(frequent_patterns)} 个")
            print(f"发现关联规则: {len(association_rules)} 个")
            print(f"发现号码组合: {len(number_groups)} 个")
            print()
            
            print("TOP 5 频繁模式:")
            for pattern, count in frequent_patterns[:5]:
                print(f"  {pattern} -> 出现 {count} 次")
            
            print("\nTOP 5 关联规则:")
            for rule in association_rules[:5]:
                print(f"  {rule['from']:02d} -> {rule['to']:02d} "
                      f"(置信度: {rule['confidence']*100:.1f}%)")
        
        # 阶段3：综合评估
        if verbose:
            print("\n📊 阶段3/3: 综合评估")
            print("-" * 70)
        
        # 使用最优参数进行最终回测
        final_accuracy = self.genetic_optimizer.fitness(
            self.best_genome, data, test_periods
        )
        
        self.learning_results['final_accuracy'] = final_accuracy
        
        if verbose:
            print(f"✅ 学习完成！")
            print(f"最终回测准确率: {final_accuracy*100:.2f}%")
            print("="*70)
            print()
        
        return {
            'best_genome': self.best_genome,
            'best_fitness': best_fitness,
            'final_accuracy': final_accuracy,
            'patterns': self.discovered_patterns,
            'evolution_history': self.genetic_optimizer.evolution_history
        }
    
    def predict_with_learned_rules(self, data, top_k=38):
        """使用学习到的规律进行预测"""
        if self.best_genome is None:
            raise ValueError("请先运行 auto_learn() 进行学习！")
        
        return self.genetic_optimizer._predict_with_genome(
            self.best_genome, data, top_k
        )
    
    def get_learning_report(self):
        """获取学习报告"""
        if not self.learning_results:
            return "尚未进行学习"
        
        report = []
        report.append("="*70)
        report.append("📊 自主学习报告")
        report.append("="*70)
        report.append(f"遗传算法适应度: {self.learning_results['genetic_fitness']*100:.2f}%")
        report.append(f"最终回测准确率: {self.learning_results['final_accuracy']*100:.2f}%")
        report.append("")
        
        if self.discovered_patterns:
            report.append("发现的模式:")
            report.append(f"  频繁模式: {len(self.discovered_patterns['frequent_patterns'])} 个")
            report.append(f"  关联规则: {len(self.discovered_patterns['association_rules'])} 个")
            report.append(f"  号码组合: {len(self.discovered_patterns['number_groups'])} 个")
        
        report.append("="*70)
        report.append("⚠️  警告: 这些'规律'是过拟合的假象！")
        report.append("    在真正随机的数据中，任何'规律'都不具有预测能力！")
        report.append("="*70)
        
        return "\n".join(report)
