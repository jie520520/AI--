#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键验证脚本 - 检查所有文件和依赖是否完整
"""

import sys
import os

print("="*70)
print("🔍 AI彩票量化研究系统 v8.1 - 文件完整性检查")
print("="*70)
print()

# 检查核心文件
print("📦 检查核心文件...")
print("-"*70)

required_files = [
    'lottery_app_v4_complete.py',
    'lottery_core.py',
    'lottery_core_enhanced.py',
    'self_learning_engine.py',
    'super_learning_engine.py',
    'extreme_optimizer.py',
    'ultra_optimizer.py',
    'mean_reversion_engine.py',  # v8.0新增
    'anti_consecutive_loss.py'    # v8.1新增
]

all_files_exist = True
for filename in required_files:
    exists = os.path.exists(filename)
    status = "✅" if exists else "❌"
    size = f"({os.path.getsize(filename)//1024}KB)" if exists else "(缺失)"
    print(f"{status} {filename:<35} {size}")
    if not exists:
        all_files_exist = False

print()

if not all_files_exist:
    print("❌ 文件不完整！请确保所有9个核心文件都在同一文件夹中。")
    sys.exit(1)

print("✅ 所有核心文件完整！")
print()

# 检查Python版本
print("🐍 检查Python版本...")
print("-"*70)
py_version = sys.version_info
print(f"当前版本: Python {py_version.major}.{py_version.minor}.{py_version.micro}")

if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
    print("❌ Python版本过低！需要Python 3.8或更高版本")
    sys.exit(1)
else:
    print("✅ Python版本符合要求")

print()

# 检查依赖包
print("📚 检查依赖包...")
print("-"*70)

required_packages = {
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'plotly': 'plotly',
    'sklearn': 'scikit-learn',
    'openpyxl': 'openpyxl'
}

all_packages_installed = True
for package_name, install_name in required_packages.items():
    try:
        __import__(package_name)
        print(f"✅ {install_name:<20} 已安装")
    except ImportError:
        print(f"❌ {install_name:<20} 未安装")
        all_packages_installed = False

print()

if not all_packages_installed:
    print("❌ 部分依赖包未安装！")
    print()
    print("请运行以下命令安装：")
    print("pip install streamlit pandas numpy scipy plotly scikit-learn openpyxl")
    sys.exit(1)

print("✅ 所有依赖包已安装！")
print()

# 测试模块导入
print("🔧 测试模块导入...")
print("-"*70)

modules_to_test = [
    ('lottery_core', 'DataProcessor'),
    ('lottery_core_enhanced', 'AuxiliaryPredictor'),
    ('self_learning_engine', 'SelfLearningEngine'),
    ('super_learning_engine', 'SuperLearningEngine'),
    ('extreme_optimizer', 'ExtremeLearningEngine'),
    ('ultra_optimizer', 'AutoLearningSystem'),
]

all_imports_ok = True
for module_name, class_name in modules_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        getattr(module, class_name)
        print(f"✅ {module_name}.{class_name}")
    except Exception as e:
        print(f"❌ {module_name}.{class_name}: {str(e)}")
        all_imports_ok = False

print()

if not all_imports_ok:
    print("❌ 模块导入失败！请检查文件是否损坏")
    sys.exit(1)

print("✅ 所有模块导入成功！")
print()

# 最终总结
print("="*70)
print("🎉 系统检查完成！")
print("="*70)
print()
print("✅ 文件完整性: 通过")
print("✅ Python版本: 通过")
print("✅ 依赖包: 通过")
print("✅ 模块导入: 通过")
print()
print("🚀 系统准备就绪！")
print()
print("下一步: 运行以下命令启动系统")
print("streamlit run lottery_app_v4_complete.py")
print()
print("="*70)
