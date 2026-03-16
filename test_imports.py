"""
测试脚本 - 验证所有模块导入
运行此脚本检查所有依赖是否正确
"""

print("开始测试模块导入...")
print("=" * 60)

# 测试1: 基础库
print("\n测试1: 基础库")
try:
    import streamlit as st
    print("✓ streamlit")
except Exception as e:
    print(f"✗ streamlit: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import scipy
    print("✓ scipy")
except Exception as e:
    print(f"✗ scipy: {e}")

try:
    import plotly
    print("✓ plotly")
except Exception as e:
    print(f"✗ plotly: {e}")

try:
    from sklearn import tree
    print("✓ scikit-learn")
except Exception as e:
    print(f"✗ scikit-learn: {e}")

try:
    import openpyxl
    print("✓ openpyxl")
except Exception as e:
    print(f"✗ openpyxl: {e}")

# 测试2: 核心模块
print("\n测试2: 核心模块")
try:
    from lottery_core import DataProcessor
    print("✓ lottery_core.DataProcessor")
except Exception as e:
    print(f"✗ lottery_core.DataProcessor: {e}")

try:
    from lottery_core import FeatureEngineering
    print("✓ lottery_core.FeatureEngineering")
except Exception as e:
    print(f"✗ lottery_core.FeatureEngineering: {e}")

try:
    from lottery_core import MLModels
    print("✓ lottery_core.MLModels")
except Exception as e:
    print(f"✗ lottery_core.MLModels: {e}")

try:
    from lottery_core import TransformerModel
    print("✓ lottery_core.TransformerModel")
except Exception as e:
    print(f"✗ lottery_core.TransformerModel: {e}")

try:
    from lottery_core import EnsembleFusion
    print("✓ lottery_core.EnsembleFusion")
except Exception as e:
    print(f"✗ lottery_core.EnsembleFusion: {e}")

# 测试3: 增强模块
print("\n测试3: 增强模块")
try:
    from lottery_core_enhanced import AuxiliaryPredictor
    print("✓ lottery_core_enhanced.AuxiliaryPredictor")
except Exception as e:
    print(f"✗ lottery_core_enhanced.AuxiliaryPredictor: {e}")

try:
    from lottery_core_enhanced import AuxiliaryBacktest
    print("✓ lottery_core_enhanced.AuxiliaryBacktest")
except Exception as e:
    print(f"✗ lottery_core_enhanced.AuxiliaryBacktest: {e}")

try:
    from lottery_core_enhanced import AggressiveEnsembleFusion
    print("✓ lottery_core_enhanced.AggressiveEnsembleFusion")
except Exception as e:
    print(f"✗ lottery_core_enhanced.AggressiveEnsembleFusion: {e}")

# 测试4: 遗传算法引擎
print("\n测试4: 遗传算法引擎")
try:
    from self_learning_engine import GeneticOptimizer
    print("✓ self_learning_engine.GeneticOptimizer")
except Exception as e:
    print(f"✗ self_learning_engine.GeneticOptimizer: {e}")

try:
    from self_learning_engine import PatternMiner
    print("✓ self_learning_engine.PatternMiner")
except Exception as e:
    print(f"✗ self_learning_engine.PatternMiner: {e}")

try:
    from self_learning_engine import SelfLearningEngine
    print("✓ self_learning_engine.SelfLearningEngine")
except Exception as e:
    print(f"✗ self_learning_engine.SelfLearningEngine: {e}")

# 测试5: 超级学习引擎
print("\n测试5: 超级学习引擎")
try:
    from super_learning_engine import ParticleSwarmOptimizer
    print("✓ super_learning_engine.ParticleSwarmOptimizer")
except Exception as e:
    print(f"✗ super_learning_engine.ParticleSwarmOptimizer: {e}")

try:
    from super_learning_engine import SimulatedAnnealing
    print("✓ super_learning_engine.SimulatedAnnealing")
except Exception as e:
    print(f"✗ super_learning_engine.SimulatedAnnealing: {e}")

try:
    from super_learning_engine import DeepRuleMiner
    print("✓ super_learning_engine.DeepRuleMiner")
except Exception as e:
    print(f"✗ super_learning_engine.DeepRuleMiner: {e}")

try:
    from super_learning_engine import SuperLearningEngine
    print("✓ super_learning_engine.SuperLearningEngine")
except Exception as e:
    print(f"✗ super_learning_engine.SuperLearningEngine: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("\n如果所有项都显示 ✓，说明系统准备就绪！")
print("如果有 ✗，请检查对应的文件和依赖包。")
