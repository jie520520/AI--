"""
模型保存/加载管理器 - Model Save/Load Manager

功能：
1. 保存学习到的模型参数和规律
2. 加载已保存的模型
3. 确保结果可重复性
4. 支持所有5种学习模式

解决的问题：
- 每次运行结果不同 → 固定随机种子
- 无法保存模型 → 提供保存/加载功能
- 缺乏参考价值 → 确保可重复性
"""

import pickle
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import random
import numpy as np


class ModelManager:
    """模型管理器"""
    
    def __init__(self, save_dir: str = "saved_models"):
        """
        初始化模型管理器
        
        Args:
            save_dir: 模型保存目录
        """
        self.save_dir = save_dir
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    @staticmethod
    def set_random_seed(seed: int = 42):
        """
        设置随机种子，确保结果可重复
        
        Args:
            seed: 随机种子值
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def save_model(
        self, 
        model_data: Dict[str, Any],
        model_name: str,
        mode: str,
        description: str = ""
    ) -> str:
        """
        保存模型
        
        Args:
            model_data: 模型数据（学习结果）
            model_name: 模型名称
            mode: 学习模式（genetic/super/extreme/auto/mean_reversion）
            description: 模型描述
            
        Returns:
            保存的文件路径
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{mode}_{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        # 准备保存数据
        save_data = {
            'model_data': model_data,
            'metadata': {
                'model_name': model_name,
                'mode': mode,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # 同时保存一个可读的JSON摘要
        summary_path = filepath.replace('.pkl', '_summary.json')
        summary = {
            'model_name': model_name,
            'mode': mode,
            'description': description,
            'created_at': save_data['metadata']['created_at'],
            'accuracy': model_data.get('accuracy', 'N/A'),
            'filepath': filepath
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            模型数据
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        return save_data
    
    def list_models(self, mode: Optional[str] = None) -> list:
        """
        列出已保存的模型
        
        Args:
            mode: 筛选模式（可选）
            
        Returns:
            模型列表
        """
        models = []
        
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.save_dir, filename)
                
                try:
                    # 尝试加载模型
                    save_data = self.load_model(filepath)
                    metadata = save_data.get('metadata', {})
                    
                    # 如果指定了模式，只返回该模式的模型
                    if mode and metadata.get('mode') != mode:
                        continue
                    
                    models.append({
                        'filepath': filepath,
                        'filename': filename,
                        'model_name': metadata.get('model_name', 'Unknown'),
                        'mode': metadata.get('mode', 'Unknown'),
                        'description': metadata.get('description', ''),
                        'created_at': metadata.get('created_at', ''),
                        'accuracy': save_data['model_data'].get('accuracy', 'N/A')
                    })
                except:
                    # 如果加载失败，跳过
                    continue
        
        # 按创建时间排序（最新的在前）
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models
    
    def delete_model(self, filepath: str) -> bool:
        """
        删除模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            是否成功删除
        """
        try:
            # 删除pkl文件
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # 删除对应的summary文件
            summary_path = filepath.replace('.pkl', '_summary.json')
            if os.path.exists(summary_path):
                os.remove(summary_path)
            
            return True
        except:
            return False
    
    def export_model_info(self, filepath: str) -> Dict[str, Any]:
        """
        导出模型信息（用于显示）
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            模型信息字典
        """
        save_data = self.load_model(filepath)
        metadata = save_data.get('metadata', {})
        model_data = save_data.get('model_data', {})
        
        info = {
            'metadata': metadata,
            'accuracy': model_data.get('accuracy', 'N/A'),
            'best_fitness': model_data.get('best_fitness', 'N/A'),
            'iterations': model_data.get('iterations', 'N/A'),
            'test_periods': model_data.get('test_periods', 'N/A')
        }
        
        # 根据不同模式添加特定信息
        mode = metadata.get('mode', '')
        
        if mode == 'genetic':
            info['best_genome'] = model_data.get('best_genome', {})
        elif mode == 'super':
            info['best_method'] = model_data.get('best_method', 'N/A')
            info['best_params'] = model_data.get('best_params', {})
        elif mode == 'extreme':
            info['weights'] = model_data.get('weights', {})
        elif mode == 'auto':
            info['weights'] = model_data.get('weights', {})
            info['success'] = model_data.get('success', False)
        elif mode == 'mean_reversion':
            info['best_analysis_periods'] = model_data.get('best_analysis_periods', 'N/A')
        
        return info


def ensure_reproducibility(seed: int = 42):
    """
    确保结果可重复性
    
    在学习开始前调用此函数，确保每次运行结果相同
    
    Args:
        seed: 随机种子（默认42）
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 如果使用了其他可能有随机性的库，也要设置
    # 例如：torch.manual_seed(seed) if torch is available


def create_model_snapshot(
    learning_results: Dict[str, Any],
    data_info: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    创建模型快照（完整保存）
    
    Args:
        learning_results: 学习结果
        data_info: 数据信息（期数、日期范围等）
        parameters: 学习参数
        
    Returns:
        完整的模型快照
    """
    snapshot = {
        'learning_results': learning_results,
        'data_info': data_info,
        'parameters': parameters,
        'created_at': datetime.now().isoformat(),
        'reproducibility_seed': 42  # 记录使用的种子
    }
    
    return snapshot
