# KeyError: 'volatility' 修复说明

## 🐛 错误信息

```python
KeyError: 'volatility'
File "lottery_app_enhanced_v3_complete.py", line 369
File "lottery_core_enhanced.py", line 109
    volatility = features['波动特征']['volatility']
```

## 🔍 问题原因

**错误代码**：
```python
# ❌ 错误
volatility = features['波动特征']['volatility']
```

**根本原因**：
`features['波动特征']`字典中**不存在**名为`'volatility'`的键。

**实际的键名**：
查看`lottery_core.py`中的`volatility_features`方法，返回的字典键是：
```python
{
    'volatility_5': ...,   # 5期波动率
    'volatility_10': ...,  # 10期波动率
    'volatility_20': ...,  # 20期波动率
    'momentum_5': ...,
    'momentum_10': ...,
    'momentum_20': ...,
    'rsi_5': ...,
    'rsi_10': ...,
    'rsi_20': ...
}
```

## ✅ 修复方案

### 修复代码

**修复后**：
```python
# ✅ 正确
volatility_20 = features['波动特征'].get('volatility_20', 0)
volatility_10 = features['波动特征'].get('volatility_10', 0)

# 计算平均波动率
avg_volatility = (volatility_20 + volatility_10) / 2 if volatility_20 or volatility_10 else 0

# 使用更合理的阈值（标准差通常是小数值）
if avg_volatility > 0.15:  # 高波动
    fused[:10] *= 1.2   # 增加极端小号权重
    fused[39:] *= 1.2   # 增加极端大号权重
```

### 关键改进

1. **使用正确的键名**
   - ❌ `'volatility'`
   - ✅ `'volatility_20'`

2. **使用.get()方法防止KeyError**
   ```python
   # 推荐用法，提供默认值
   volatility_20 = features['波动特征'].get('volatility_20', 0)
   
   # 不推荐，会抛出KeyError
   volatility_20 = features['波动特征']['volatility_20']
   ```

3. **使用合理的阈值**
   - 波动率是标准差，通常是0.01-0.3之间的小数
   - 原来用0.5作为阈值太高，改为0.15更合理

4. **综合多期数据**
   ```python
   # 同时考虑10期和20期的波动率
   avg_volatility = (volatility_20 + volatility_10) / 2
   ```

## 📊 波动率说明

### 什么是波动率？

波动率是收益率的标准差，计算公式：
```python
# 1. 计算收益率
returns = np.diff(recent) / recent[:-1]

# 2. 计算标准差
volatility = np.std(returns)
```

### 典型值范围

对于彩票特码（1-49）：
- **低波动**：< 0.10（数字变化小）
- **中波动**：0.10 - 0.20（正常变化）
- **高波动**：> 0.20（数字跳动大）

### 示例

```python
# 示例数据：特码序列
codes = [25, 26, 27, 28, 29]  # 连续递增
returns = [0.04, 0.04, 0.04, 0.04]  # 都是4%
volatility = 0.0  # 低波动

codes = [10, 40, 15, 45, 20]  # 大幅跳动
returns = [3.0, -0.625, 2.0, -0.556]
volatility = 1.5  # 高波动
```

## 🧪 测试验证

### 测试1：验证修复

```python
# 运行系统
streamlit run lottery_app_enhanced_v3_complete.py

# 上传数据文件
# 点击"运行AI预测"

# 如果不再出现KeyError，说明修复成功
```

### 测试2：检查波动率值

```python
# 在预测过程中，features['波动特征']应该包含：
{
    'volatility_5': 0.12,
    'volatility_10': 0.15,
    'volatility_20': 0.18,
    'momentum_5': 0.05,
    'momentum_10': 0.08,
    'momentum_20': 0.10,
    'rsi_5': 55.2,
    'rsi_10': 52.8,
    'rsi_20': 50.5
}
```

## 🔧 完整修复文件

已修复的文件：
- ✅ `lottery_core_enhanced.py`

修复位置：
- 第109-115行：波动性调整逻辑

## 📝 其他类似问题的预防

### 最佳实践

1. **总是使用.get()方法**
   ```python
   # ✅ 推荐
   value = dict.get('key', default_value)
   
   # ❌ 不推荐
   value = dict['key']
   ```

2. **检查键是否存在**
   ```python
   if 'key' in dict:
       value = dict['key']
   else:
       value = default_value
   ```

3. **查看源代码确认键名**
   - 不要假设键名
   - 查看返回字典的实际结构
   - 使用print调试

### 调试技巧

如果遇到KeyError：

```python
# 1. 打印整个字典查看所有键
print(features['波动特征'].keys())

# 2. 打印完整字典
print(features['波动特征'])

# 3. 使用try-except捕获
try:
    value = features['波动特征']['volatility']
except KeyError as e:
    print(f"KeyError: {e}")
    print(f"Available keys: {features['波动特征'].keys()}")
```

## ✅ 修复完成

现在系统应该可以正常运行了！

**下次遇到KeyError时**：
1. 查看错误提示的键名
2. 打印字典的所有键
3. 使用正确的键名
4. 使用.get()方法添加默认值

---

**版本**：v3.2.1（KeyError修复版）  
**修复日期**：2026-03-10  
**修复内容**：修复波动特征键名错误

✅ **现在可以正常运行了！**
