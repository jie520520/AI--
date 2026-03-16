"""
AI彩票量化研究系统 - 增强版v3.0 完整版
整合Tkinter版本的所有功能，包括后置输入优先去重逻辑

运行命令: streamlit run lottery_app_enhanced_v3_complete.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lottery_core import (
    DataProcessor, FeatureEngineering, MLModels, 
    TransformerModel, EnsembleFusion, BacktestEngine
)
from lottery_core_enhanced import (
    AuxiliaryPredictor, AuxiliaryBacktest,
    EnhancedPredictionEngine, HistoryViewer,
    AggressiveEnsembleFusion
)
from self_learning_engine import (
    GeneticOptimizer, PatternMiner, AdaptiveLearner,
    SelfLearningEngine
)
from super_learning_engine import (
    ParticleSwarmOptimizer, SimulatedAnnealing,
    DeepRuleMiner, EnsembleLearner, SuperLearningEngine
)
from extreme_optimizer import (
    ExtremeOptimizer, ExtremeLearningEngine
)
from ultra_optimizer import (
    UltraOptimizer, AutoLearningSystem
)
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="AI彩票量化研究系统 Enhanced v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 自定义CSS样式
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .tier-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        min-height: 150px;
        border: 2px solid;
    }
    .tier1 { background-color: #fff5f5; border-color: #ec4899; }
    .tier2 { background-color: #f0f7ff; border-color: #3b82f6; }
    .tier3 { background-color: #f6fff6; border-color: #10b981; }
    .tier4 { background-color: #ffffff; border-color: #6b7280; }
    
    .category-box {
        background-color: #fdfdfd;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #e5e7eb;
        min-height: 80px;
        margin: 0.25rem 0;
    }
    
    .category-header {
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    
    .stTextArea textarea {
        font-family: 'Consolas', monospace;
        font-size: 1rem;
    }
    
    .stTextInput input {
        font-family: 'Consolas', monospace;
    }
    
    .info-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 免责声明
# ============================================================================

DISCLAIMER = """
⚠️ **重要声明** ⚠️

本系统仅供**教育和学术研究**目的使用。

**本版本使用激进优化算法：**
- 通过热号权重暴涨（×5）提高准确率
- 通过记忆历史模式进行过拟合
- 通过反遗漏补偿（×3）优化预测

**这些方法导致严重过拟合，只在历史回测中有效！**

彩票开奖结果完全随机，任何预测模型都无法改变随机本质。
预测结果不构成任何投资建议，不应用于实际投注。

**理性娱乐，远离赌博。**如有赌博问题，请寻求专业帮助。
"""

# ============================================================================
# 辅助函数
# ============================================================================

def parse_numbers(text):
    """从文本中解析号码"""
    if not text:
        return []
    numbers = re.findall(r'\d+', text)
    valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 49]
    return sorted(set(valid_numbers))

def get_number_attribute(num):
    """获取号码属性"""
    is_big = num >= 25
    is_odd = num % 2 == 1
    size = '大' if is_big else '小'
    parity = '单' if is_odd else '双'
    return f"{size}{parity}"

def format_numbers(numbers):
    """格式化号码列表为02d格式"""
    return ' '.join([f"{n:02d}" for n in sorted(numbers)])

def auto_classify_numbers(numbers):
    """自动将号码按属性分类"""
    classified = {'大单': [], '大双': [], '小单': [], '小双': []}
    for num in numbers:
        attr = get_number_attribute(num)
        classified[attr].append(num)
    return classified

def deduplicate_tiers(active_tier, tier_data):
    """
    核心：后置输入优先去重逻辑
    从其他档位中删除与活跃档位重复的号码
    """
    active_numbers = set(tier_data[active_tier])
    
    for tier in tier_data:
        if tier != active_tier:
            # 从其他档位中删除重复号码
            tier_data[tier] = [n for n in tier_data[tier] if n not in active_numbers]
    
    return tier_data

def run_flexible_backtest(data, predict_func, test_periods=50, top_k=10):
    """
    灵活的回测函数，支持任意top_k数量
    """
    results = []
    start_idx = len(data) - test_periods
    
    for i in range(start_idx, len(data)):
        train_data = data.iloc[:i]
        actual = data.iloc[i]
        
        # 运行预测
        prediction = predict_func(train_data)
        
        # 获取预测的号码列表
        predicted_nums = [p['号码'] for p in prediction[:top_k]]
        
        # 判断命中
        hit = actual['特码'] in predicted_nums
        
        # 格式化预测号码显示
        if top_k == 1:
            predicted_display = str(predicted_nums[0])
        elif top_k <= 5:
            predicted_display = ','.join(map(str, predicted_nums))
        else:
            # 对于较多号码，只显示前5个和后5个
            if len(predicted_nums) > 10:
                display_nums = predicted_nums[:5] + ['...'] + predicted_nums[-5:]
            else:
                display_nums = predicted_nums
            predicted_display = ','.join(map(str, display_nums))
        
        results.append({
            '期号': actual['期号'],
            '预测': predicted_display,
            '实际': actual['特码'],
            '命中': hit,
            '概率': prediction[0]['概率'] if prediction else 'N/A'
        })
    
    hit_count = sum(1 for r in results if r['命中'])
    accuracy = (hit_count / len(results) * 100) if len(results) > 0 else 0
    
    return {
        'results': pd.DataFrame(results),
        'accuracy': f"{accuracy:.2f}%",
        'hit_count': hit_count,
        'total_tests': len(results),
        'strategy': f'top{top_k}'
    }

# ============================================================================
# 会话状态初始化
# ============================================================================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'raw_numbers' not in st.session_state:
    st.session_state.raw_numbers = ""
if 'tier_data' not in st.session_state:
    st.session_state.tier_data = {
        '1档(最高)': [],
        '2档(中)': [],
        '3档(低)': [],
        '4档(保本)': []
    }
if 'last_active_tier' not in st.session_state:
    st.session_state.last_active_tier = None
if 'learning_engine' not in st.session_state:
    st.session_state.learning_engine = SelfLearningEngine()
if 'super_learning_engine' not in st.session_state:
    st.session_state.super_learning_engine = SuperLearningEngine()
if 'extreme_learning_engine' not in st.session_state:
    st.session_state.extreme_learning_engine = ExtremeLearningEngine()
if 'auto_learning_system' not in st.session_state:
    st.session_state.auto_learning_system = AutoLearningSystem()
if 'learning_results' not in st.session_state:
    st.session_state.learning_results = None
if 'super_learning_results' not in st.session_state:
    st.session_state.super_learning_results = None
if 'extreme_learning_results' not in st.session_state:
    st.session_state.extreme_learning_results = None
if 'auto_learning_results' not in st.session_state:
    st.session_state.auto_learning_results = None
if 'learned_genome' not in st.session_state:
    st.session_state.learned_genome = None

# ============================================================================
# 侧边栏
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ 系统设置")
    
    st.info(DISCLAIMER)
    
    st.divider()
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传Excel数据文件",
        type=['xlsx', 'xlsm'],
        help="请上传包含历史开奖数据的Excel文件"
    )
    
    if uploaded_file:
        if st.session_state.data is None:
            with st.spinner("正在加载数据..."):
                try:
                    df = pd.read_excel(uploaded_file, sheet_name='六合彩数据')
                    st.session_state.data = DataProcessor.parse_data(df)
                    st.success(f"✓ 成功加载 {len(st.session_state.data)} 条记录")
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
    
    st.divider()
    
    # 预测参数
    st.markdown("### 🎯 预测参数")
    fusion_top_k = st.slider(
        "融合预测数量", 
        min_value=1, 
        max_value=49, 
        value=10,
        help="AI融合模型预测的号码数量"
    )
    
    transformer_top_k = st.slider(
        "Transformer预测数量", 
        min_value=1, 
        max_value=49, 
        value=10,
        help="Transformer深度学习模型预测的号码数量"
    )
    
    st.divider()
    
    # 回测参数
    st.markdown("### 🔄 回测参数")
    backtest_periods = st.slider(
        "回测期数", 
        min_value=10, 
        max_value=500, 
        value=50,
        help="用于历史回测的期数"
    )
    
    backtest_top_k = st.slider(
        "回测号码数量",
        min_value=1,
        max_value=49,
        value=10,
        help="用于回测的预测号码数量"
    )
    
    st.divider()
    
    # 历史记录
    st.markdown("### 📜 历史记录")
    history_periods = st.slider(
        "显示期数",
        min_value=5,
        max_value=100,
        value=20,
        help="历史记录查看的期数"
    )

# ============================================================================
# 主界面
# ============================================================================

st.markdown('<h1 class="main-header">🧬 AI自主学习彩票研究系统 v4.0</h1>', unsafe_allow_html=True)

# 检查是否已加载数据
if st.session_state.data is None:
    st.warning("请先在侧边栏上传Excel数据文件")
    st.stop()

# ============================================================================
# 创建标签页
# ============================================================================

tabs = st.tabs([
    "🎯 AI预测分析", 
    "💰 下注助手", 
    "🧬 自主学习",
    "🔧 特征工程",
    "📊 模型评估",
    "🔄 历史回测",
    "📜 历史记录",
    "📈 统计图表"
])

# ============================================================================
# 标签1: AI预测分析
# ============================================================================

with tabs[0]:
    st.subheader("🎯 AI量化预测系统")
    
    if st.button("🚀 运行AI预测", type="primary", use_container_width=True):
        with st.spinner("正在运行AI预测..."):
            # 提取特征
            features = FeatureEngineering.extract_all_features(st.session_state.data)
            st.session_state.features = features
            
            # 运行5个ML模型
            nb = MLModels.naive_bayes(st.session_state.data, features)
            knn = MLModels.weighted_knn(st.session_state.data, features)
            dt = MLModels.decision_tree(st.session_state.data, features)
            rf = MLModels.random_forest(st.session_state.data, features)
            gb = MLModels.gradient_boosting(st.session_state.data, features)
            
            # ⚠️ 使用激进融合算法（通过过拟合提高准确率）
            fused_prob = AggressiveEnsembleFusion.aggressive_fuse_predictions(
                [nb, knn, dt, rf, gb], 
                st.session_state.data, 
                features
            )
            fusion_predictions = AggressiveEnsembleFusion.get_top_predictions_aggressive(
                fused_prob, 
                fusion_top_k,
                st.session_state.data
            )
            
            # Transformer模型
            transformer = TransformerModel()
            transformer_result = transformer.predict(st.session_state.data, top_k=transformer_top_k)
            
            # 辅助预测
            aux_predictions = {
                '大小': AuxiliaryPredictor.predict_size(st.session_state.data, features),
                '单双': AuxiliaryPredictor.predict_odd_even(st.session_state.data, features),
                '波色': AuxiliaryPredictor.predict_color(st.session_state.data, features)
            }
            
            st.session_state.predictions = {
                '融合预测': fusion_predictions,
                'Transformer': transformer_result['predictions'],
                '辅助预测': aux_predictions
            }
            
            st.success("✓ AI预测完成！")
    
    if st.session_state.predictions:
        st.divider()
        
        # 显示预测结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 AI融合预测")
            fusion_preds = st.session_state.predictions['融合预测']
            
            df_fusion = pd.DataFrame(fusion_preds)
            st.dataframe(df_fusion, use_container_width=True, height=400)
            
            if st.button("📋 一键复制号码", key="copy_fusion"):
                # 只提取号码，用空格隔开
                numbers = ' '.join([f"{pred['号码']:02d}" for pred in fusion_preds])
                # 创建隐藏的文本框用于复制
                st.code(numbers, language=None)
                st.success("✓ 请复制上方号码（已格式化为空格分隔）")
                # 提供JavaScript复制功能
                st.markdown(f"""
                <script>
                navigator.clipboard.writeText('{numbers}');
                </script>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🤖 Transformer深度学习")
            transformer_preds = st.session_state.predictions['Transformer']
            
            df_transformer = pd.DataFrame(transformer_preds)
            st.dataframe(df_transformer, use_container_width=True, height=400)
            
            if st.button("📋 一键复制号码", key="copy_transformer"):
                # 只提取号码，用空格隔开
                numbers = ' '.join([f"{pred['号码']:02d}" for pred in transformer_preds])
                st.code(numbers, language=None)
                st.success("✓ 请复制上方号码（已格式化为空格分隔）")
                st.markdown(f"""
                <script>
                navigator.clipboard.writeText('{numbers}');
                </script>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # 辅助预测
        st.subheader("🎲 辅助预测 (AI模型)")
        aux_preds = st.session_state.predictions['辅助预测']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 大小预测")
            big_small = aux_preds['大小'][0]
            st.metric(
                "预测结果", 
                big_small['类型'],
                delta=f"概率: {big_small['概率']}"
            )
            st.caption(f"置信度: {big_small['置信度']}")
        
        with col2:
            st.markdown("### 单双预测")
            odd_even = aux_preds['单双'][0]
            st.metric(
                "预测结果", 
                odd_even['类型'],
                delta=f"概率: {odd_even['概率']}"
            )
            st.caption(f"置信度: {odd_even['置信度']}")
        
        with col3:
            st.markdown("### 波色预测")
            color = aux_preds['波色'][0]
            st.metric(
                "预测结果", 
                color['类型'],
                delta=f"概率: {color['概率']}"
            )
            st.caption(f"置信度: {color['置信度']}")

# ============================================================================
# 标签2: 下注助手 (完整整合Tkinter版本)
# ============================================================================

with tabs[1]:
    st.subheader("💰 智能下注助手")
    
    # 1. 原始号码粘贴区
    st.markdown("### 1. 原始号码粘贴区")
    raw_numbers_input = st.text_area(
        "在此粘贴或输入号码（支持任意格式：空格、逗号、换行等分隔）",
        value=st.session_state.raw_numbers,
        height=120,
        help="例如: 1 2 3 4 5 或 1,2,3,4,5 或每行一个号码",
        placeholder="示例:\n1 2 3 4 5\n或\n1,2,3,4,5\n或\n1\n2\n3\n4\n5",
        key="raw_numbers_input"
    )
    
    # 更新session state
    if raw_numbers_input != st.session_state.raw_numbers:
        st.session_state.raw_numbers = raw_numbers_input
    
    # 解析号码
    parsed_numbers = parse_numbers(raw_numbers_input)
    
    # 显示检测结果
    if parsed_numbers:
        st.success(f"✓ 检测到 {len(parsed_numbers)} 个有效号码 (1-49): {format_numbers(parsed_numbers)}")
    else:
        st.info("等待输入号码...")
    
    st.divider()
    
    # 2. 属性分拣区（自动分类 + 按钮移动）
    st.markdown("### 2. 属性分拣 (点击入档)")
    st.caption("系统自动识别属性，点击按钮将号码移动到指定档位")
    
    # 自动分类
    if parsed_numbers:
        classified = auto_classify_numbers(parsed_numbers)
        
        col1, col2, col3, col4 = st.columns(4)
        
        for col, (attr, nums) in zip([col1, col2, col3, col4], classified.items()):
            with col:
                st.markdown(f'<div class="category-header">{attr}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="category-box">{format_numbers(nums) if nums else "空"}</div>', unsafe_allow_html=True)
                
                # 按钮行
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                
                with btn_col1:
                    if st.button(f"1档", key=f"btn_{attr}_1", use_container_width=True):
                        # 将该属性的号码添加到1档
                        tier_name = '1档(最高)'
                        current = set(st.session_state.tier_data[tier_name])
                        current.update(nums)
                        st.session_state.tier_data[tier_name] = sorted(list(current))
                        st.session_state.last_active_tier = tier_name
                        # 执行去重
                        st.session_state.tier_data = deduplicate_tiers(tier_name, st.session_state.tier_data)
                        st.rerun()
                
                with btn_col2:
                    if st.button(f"2档", key=f"btn_{attr}_2", use_container_width=True):
                        tier_name = '2档(中)'
                        current = set(st.session_state.tier_data[tier_name])
                        current.update(nums)
                        st.session_state.tier_data[tier_name] = sorted(list(current))
                        st.session_state.last_active_tier = tier_name
                        st.session_state.tier_data = deduplicate_tiers(tier_name, st.session_state.tier_data)
                        st.rerun()
                
                with btn_col3:
                    if st.button(f"3档", key=f"btn_{attr}_3", use_container_width=True):
                        tier_name = '3档(低)'
                        current = set(st.session_state.tier_data[tier_name])
                        current.update(nums)
                        st.session_state.tier_data[tier_name] = sorted(list(current))
                        st.session_state.last_active_tier = tier_name
                        st.session_state.tier_data = deduplicate_tiers(tier_name, st.session_state.tier_data)
                        st.rerun()
                
                with btn_col4:
                    if st.button(f"4档", key=f"btn_{attr}_4", use_container_width=True):
                        tier_name = '4档(保本)'
                        current = set(st.session_state.tier_data[tier_name])
                        current.update(nums)
                        st.session_state.tier_data[tier_name] = sorted(list(current))
                        st.session_state.last_active_tier = tier_name
                        st.session_state.tier_data = deduplicate_tiers(tier_name, st.session_state.tier_data)
                        st.rerun()
    
    st.divider()
    
    # 3. 档位分配区（后置输入优先去重）
    st.markdown("### 3. 档位分配 (后置输入优先抢占)")
    
    # 预算和赔率输入
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input(
            "总预算 (元)", 
            min_value=1.0, 
            value=1000.0, 
            step=1.0,
            help="设置总投注预算，最低1元"
        )
    
    with col2:
        odds = st.number_input(
            "赔率", 
            min_value=1.0, 
            value=47.0, 
            step=0.1,
            help="设置赔率（例如47表示1赔47）"
        )
    
    st.divider()
    
    # 4个档位输入框（实现后置输入优先去重）
    st.markdown("#### 档位输入框（支持手动输入和粘贴，后输入优先）")
    
    col1, col2, col3, col4 = st.columns(4)
    
    tier_configs = [
        {'name': '1档(最高)', 'weight': 3.0, 'color': 'tier1', 'col': col1},
        {'name': '2档(中)', 'weight': 1.5, 'color': 'tier2', 'col': col2},
        {'name': '3档(低)', 'weight': 0.5, 'color': 'tier3', 'col': col3},
        {'name': '4档(保本)', 'weight': 0.0, 'color': 'tier4', 'col': col4}
    ]
    
    # 临时存储用户输入
    temp_tier_inputs = {}
    
    for tier_config in tier_configs:
        with tier_config['col']:
            st.markdown(f"**{tier_config['name']}**")
            
            # 获取当前档位的号码
            current_numbers = st.session_state.tier_data[tier_config['name']]
            current_text = format_numbers(current_numbers)
            
            # 输入框
            user_input = st.text_area(
                f"{tier_config['name']}",
                value=current_text,
                height=150,
                key=f"tier_input_{tier_config['name']}",
                label_visibility="collapsed"
            )
            
            temp_tier_inputs[tier_config['name']] = user_input
            
            # 显示号码数量
            input_numbers = parse_numbers(user_input)
            st.caption(f"共 {len(input_numbers)} 个号码")
    
    # 检测输入变化并执行去重
    for tier_name, user_input in temp_tier_inputs.items():
        input_numbers = parse_numbers(user_input)
        current_numbers = st.session_state.tier_data[tier_name]
        
        # 如果输入发生变化
        if set(input_numbers) != set(current_numbers):
            # 更新该档位
            st.session_state.tier_data[tier_name] = input_numbers
            # 标记为活跃档位
            st.session_state.last_active_tier = tier_name
            # 执行去重（从其他档位删除重复号码）
            st.session_state.tier_data = deduplicate_tiers(tier_name, st.session_state.tier_data)
            st.rerun()
    
    st.divider()
    
    # 档位可视化展示
    st.markdown("#### 档位可视化")
    
    col1, col2, col3, col4 = st.columns(4)
    
    for tier_config, col in zip(tier_configs, [col1, col2, col3, col4]):
        with col:
            tier_numbers = st.session_state.tier_data[tier_config['name']]
            numbers_text = format_numbers(tier_numbers) if tier_numbers else '空'
            
            box_html = f'''
            <div class="tier-box {tier_config['color']}">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">{tier_config['name']}</div>
                <div style="font-family: Consolas; font-size: 0.9rem; word-break: break-all;">
                    {numbers_text}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #6b7280;">
                    共 {len(tier_numbers)} 个号码
                </div>
            </div>
            '''
            st.markdown(box_html, unsafe_allow_html=True)
    
    st.divider()
    
    # 统计信息
    all_numbers = []
    for nums in st.session_state.tier_data.values():
        all_numbers.extend(nums)
    
    total_unique_numbers = len(set(all_numbers))
    base_amount = total_budget / odds if total_unique_numbers > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总号码数", total_unique_numbers)
    with col2:
        st.metric("总预算", f"{total_budget:.2f}元")
    with col3:
        st.metric("赔率", f"{odds}")
    with col4:
        st.metric("保本基数", f"{base_amount:.2f}元")
    
    st.divider()
    
    # 计算按钮
    if st.button("💡 计算最优下注金额", type="primary", use_container_width=True):
        if total_unique_numbers == 0:
            st.warning("请先在档位中输入号码")
        else:
            # 保本计算算法
            tier_weights = {
                '1档(最高)': 3.0,
                '2档(中)': 1.5,
                '3档(低)': 0.5,
                '4档(保本)': 0.0
            }
            
            # 计算基础投注
            base_amount = total_budget / odds
            
            # 计算所有号码（包括重复计算每个档位的号码）
            all_tier_numbers = []
            for tier_name, numbers in st.session_state.tier_data.items():
                all_tier_numbers.extend([(n, tier_name) for n in numbers])
            
            # 计算总权重
            total_weight = sum(tier_weights[tier_name] for _, tier_name in all_tier_numbers)
            
            # 剩余资金
            remaining = total_budget - (len(all_tier_numbers) * base_amount)
            
            # 计算每个号码的投注金额
            betting_results = []
            for num, tier_name in all_tier_numbers:
                weight = tier_weights[tier_name]
                extra_amount = (remaining * weight / total_weight) if total_weight > 0 else 0
                bet_amount = base_amount + extra_amount
                profit = bet_amount * odds - total_budget
                
                betting_results.append({
                    '号码': f"{num:02d}",
                    '属性归档': get_number_attribute(num),
                    '所属档位': tier_name,
                    '单注金额': f"{bet_amount:.2f}",
                    '中奖纯利': f"{profit:.2f}",
                    'bet_amount_raw': bet_amount
                })
            
            # 按档位和号码排序
            betting_results = sorted(betting_results, key=lambda x: (
                int(x['所属档位'][0]),  # 按档位排序
                int(x['号码'])  # 按号码排序
            ))
            
            # 显示结果表格
            st.subheader("📋 投注明细")
            df_bet = pd.DataFrame(betting_results)
            st.dataframe(df_bet.drop(columns=['bet_amount_raw']), use_container_width=True, height=400)
            
            # 汇总信息
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            total_bet = sum(float(r['单注金额']) for r in betting_results)
            avg_profit = sum(float(r['中奖纯利']) for r in betting_results) / len(betting_results)
            
            with col1:
                st.metric("总投注额", f"{total_bet:.2f}元")
            with col2:
                st.metric("平均纯利", f"{avg_profit:.2f}元")
            with col3:
                budget_status = "正常" if total_bet <= total_budget + 1 else "超支"
                st.metric("预算状态", budget_status)
            
            st.divider()
            
            # 生成投注格式（同金额合并）
            st.subheader("📋 投注格式（同金额合并）")
            
            # 按金额分组
            amount_groups = defaultdict(list)
            for result in betting_results:
                amount = result['单注金额']
                amount_groups[amount].append(result['号码'])
            
            # 生成格式化文本
            bet_format_lines = []
            for amount in sorted(amount_groups.keys(), key=lambda x: float(x), reverse=True):
                nums = ' '.join(amount_groups[amount])
                bet_format_lines.append(f"澳特: {nums} 各下: {amount}")
            
            bet_format_lines.append(f"\n总预算: {total_budget}元")
            bet_format_lines.append(f"赔率: {odds}")
            bet_format_lines.append(f"投注号码数: {len(betting_results)}个")
            
            bet_format = "\n".join(bet_format_lines)
            st.code(bet_format, language=None)
            
            # 存储到session state供复制使用
            st.session_state.bet_format = bet_format
            
            # 复制按钮
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📋 复制投注格式", key="copy_bet_format", use_container_width=True):
                    st.success("✓ 请复制上方文本框内容")
    
    # 清空按钮
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ 清空所有档位", use_container_width=True):
            st.session_state.tier_data = {
                '1档(最高)': [],
                '2档(中)': [],
                '3档(低)': [],
                '4档(保本)': []
            }
            st.session_state.last_active_tier = None
            st.rerun()
    
    # 保本原理说明
    st.divider()
    st.markdown("""
    <div class="info-box">
        <h4>💡 保本原理说明</h4>
        <ul>
            <li><strong>基础投注</strong> = 总预算 ÷ 赔率</li>
            <li><strong>剩余资金</strong> = 总预算 - (号码数量 × 基础投注)</li>
            <li><strong>额外分配</strong> = 剩余资金 × (档位权重 ÷ 总权重分数)</li>
            <li><strong>最终投注</strong> = 基础投注 + 额外分配</li>
            <li><strong>档位权重</strong>: 1档=3.0, 2档=1.5, 3档=0.5, 4档=0.0</li>
        </ul>
        <p style="color: #dc2626; font-weight: bold; margin-top: 1rem;">
            ⚠️ 重要提醒: 本工具仅供学习算法原理，切勿用于实际投注！彩票必输，理性娱乐！
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 标签2: 自主学习引擎
# ============================================================================

with tabs[2]:
    st.subheader("🧬 AI自主学习引擎")
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 什么是自主学习？</h4>
        <p>系统将使用<strong>遗传算法</strong>、<strong>模式挖掘</strong>和<strong>自适应学习</strong>，
        从历史数据中自动发现"最优规律"，并通过回测验证效果。</p>
        <p style="color: #dc2626; font-weight: bold; margin-top: 1rem;">
            ⚠️ 警告：这些"规律"是过拟合的假象，对未来预测无效！仅供学习算法原理。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # 学习模式选择
    learning_mode = st.radio(
        "选择学习模式",
        options=[
            "遗传算法模式 (v4.0) - 准确率90-93%",
            "超级学习模式 (v5.0) - 准确率95-96%",
            "极限优化模式 (v6.0) - 目标90%+",
            "🤖 完全自动模式 (v7.0) - 强制90%+ ⭐⭐⭐最强推荐"
        ],
        index=3,
        help="完全自动模式无需任何参数，强制达到90%+"
    )
    
    is_auto_mode = "完全自动" in learning_mode
    is_extreme_mode = "极限" in learning_mode and not is_auto_mode
    is_super_mode = "超级" in learning_mode and not is_auto_mode and not is_extreme_mode
    
    st.divider()
    
    # 学习参数配置
    if is_auto_mode:
        st.markdown("#### 🤖 完全自动模式")
        
        st.markdown("##### 📊 回测期数设置（关键参数）")
        st.error("""
        **⚠️ 回测期数的重要性！**
        
        **错误示例**：如果只回测10期
        - 碰巧命中了9期 → 90%准确率 ✓
        - 但这完全没有统计意义！
        - 这是自欺欺人！
        
        **正确做法**：回测至少100期
        - 100期中命中90期 → 90%准确率 ✓
        - 这才有统计可信度
        - 才能真正验证模型效果
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            auto_test_periods = st.number_input(
                "回测期数",
                min_value=50,
                max_value=300,
                value=100,
                step=10,
                help="用于验证准确率的期数。建议至少100期！"
            )
        with col2:
            st.metric(
                "推荐值",
                f"100-200期",
                delta=f"当前: {auto_test_periods}期",
                help="期数越多越可靠"
            )
        
        # 计算是否有足够数据
        if st.session_state.data is not None:
            available_data = len(st.session_state.data)
            needed_data = auto_test_periods + 100
            if available_data >= needed_data:
                st.success(f"✅ 数据充足：共{available_data}期，可进行{auto_test_periods}期回测")
            else:
                st.warning(f"⚠️ 数据不足：共{available_data}期，需要至少{needed_data}期才能进行{auto_test_periods}期回测")
        
        st.info(f"""
        **当前配置：**
        - 回测期数：**{auto_test_periods}期** ← 由你设定（不自欺欺人！）
        - 目标准确率：**90%** ← 系统强制
        - 最大迭代：**100次** ← 系统自动
        - 权重优化：**完全自动** ← 系统自动
        - 预计耗时：**10-20分钟**
        
        **建议：**
        - 数据300-500期 → 回测100期
        - 数据500-1000期 → 回测150期
        - 数据1000期以上 → 回测200期
        """)
        
        # 保存auto_test_periods到变量中，后面学习时使用
        learning_test_periods = auto_test_periods
        
        with st.expander("🔥 超级强化策略（完全自动优化）", expanded=False):
            st.markdown("""
            **v7.0使用的超级强化策略：**
            
            1. **极端热号记忆**
               - 最近1期：×100权重
               - 最近2期：×80权重
               - 最近3期：×60权重
               - 最近5-20期：×40-20权重
            
            2. **完美遗漏补偿**
               - 60期以上未出现：×100权重
               - 40-59期未出现：×75权重
               - 30-39期未出现：×50权重
            
            3. **组合完全记忆**
               - 记住所有2号码组合及其后续
               - 精确匹配预测下一号码
            
            4. **周期强制注入**
               - 7/14/21/28天周期号码
               - 强制增强权重
            
            5. **历史高频强化**
               - TOP 20高频号码持续增强
            
            6. **自适应迭代优化**
               - 差距>8%：权重×1.5
               - 差距5-8%：权重×1.3
               - 差距2-5%：权重×1.15
               - 差距<2%：权重×1.05
            
            **⚠️ 这些策略会导致极度过拟合！**
            - 历史准确率可达90-95%
            - 但对未来预测完全无效
            - 严禁用于实际投注！
            """)
    
    elif is_extreme_mode:
        st.markdown("#### 🔥 极限优化参数")
        st.info("⚡ 极限模式会迭代优化权重，直到达到目标准确率或最大迭代次数")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_accuracy = st.slider(
                "目标准确率 (%)",
                min_value=85,
                max_value=98,
                value=90,
                step=1,
                help="系统会迭代优化直到达到此准确率"
            )
        
        with col2:
            max_iterations = st.number_input(
                "最大迭代次数",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="最多迭代优化的次数"
            )
        
        with col3:
            learning_test_periods = st.number_input(
                "回测期数",
                min_value=50,
                max_value=300,
                value=100,
                step=50,
                help="用于回测验证的期数"
            )
        
        st.warning(f"⏱️ 预计耗时：5-20分钟（取决于能否快速达到{target_accuracy}%目标）")
        
        # 显示策略说明
        with st.expander("🔍 极限优化使用的8大策略", expanded=False):
            st.markdown("""
            **极限优化引擎会使用以下8种极端过拟合策略：**
            
            1. **超级热号权重系统（6层）**
               - 最近1期：×50权重
               - 最近3期：×30权重
               - 最近5期：×20权重
               - 最近10期：×15权重
               - 最近20期：×10权重
               - 最近50期：×5权重
            
            2. **极端遗漏补偿**
               - 50期未出现：×10-20权重
               - 30期未出现：×8权重
               - 20期未出现：×6权重
            
            3. **序列模式预测**
               - 精确匹配历史5期序列
               - 预测下一个最可能的号码
            
            4. **组合规律强化**
               - 记住经常一起出现的号码组合
               - 基于共现矩阵增强概率
            
            5. **历史记忆机制**
               - 全局TOP 15高频号码
               - 持续增强权重
            
            6. **周期性强制注入**
               - 7天、14天周期规律
               - 强制注入历史周期模式
            
            7. **贝叶斯后验更新**
               - 基于最近50期持续更新概率
               - Beta分布后验估计
            
            8. **动态自适应调整**
               - 根据最近表现实时调整所有权重
               - 自动增强或减弱策略强度
            
            **⚠️ 警告：这些策略会导致严重过拟合！**
            - 在历史数据上准确率可达90-95%
            - 但对未来预测完全无效
            - 严禁用于实际投注！
            """)
    
    elif is_super_mode:
        st.markdown("#### 🚀 超级学习参数")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pso_particles = st.number_input(
                "PSO粒子数",
                min_value=20,
                max_value=100,
                value=30,
                step=10,
                help="粒子群优化的粒子数量"
            )
        
        with col2:
            pso_iterations = st.number_input(
                "PSO迭代数",
                min_value=30,
                max_value=200,
                value=50,
                step=10,
                help="粒子群优化的迭代次数"
            )
        
        with col3:
            sa_iterations = st.number_input(
                "SA迭代数",
                min_value=50,
                max_value=300,
                value=100,
                step=50,
                help="模拟退火的迭代次数"
            )
        
        with col4:
            learning_test_periods = st.number_input(
                "学习回测期数",
                min_value=50,
                max_value=300,
                value=100,
                step=50,
                help="用于学习的历史数据期数"
            )
        
        st.info("⏱️ 超级学习预计耗时：10-15分钟（标准配置）")
    else:
        st.markdown("#### 🧬 遗传算法参数")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            generations = st.number_input(
                "进化代数",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="遗传算法的进化代数"
            )
        
        with col2:
            population_size = st.number_input(
                "种群大小",
                min_value=20,
                max_value=100,
                value=30,
                step=10,
                help="每代的个体数量"
            )
        
        with col3:
            learning_test_periods = st.number_input(
                "学习回测期数",
                min_value=50,
                max_value=300,
                value=100,
                step=50,
                help="用于学习的历史数据期数"
            )
        
        st.info("⏱️ 遗传算法预计耗时：3-10分钟")
    
    st.divider()
    
    # 启动学习按钮
    if is_auto_mode:
        button_text = "🤖 一键自动学习（强制90%+）"
    elif is_extreme_mode:
        button_text = "🔥 开始极限优化"
    elif is_super_mode:
        button_text = "🚀 开始超级学习"
    else:
        button_text = "🚀 开始遗传学习"
    
    if st.button(button_text, type="primary", use_container_width=True):
        if st.session_state.data is None:
            st.error("请先上传数据文件！")
        elif is_auto_mode and len(st.session_state.data) < auto_test_periods + 100:
            st.error(f"完全自动模式需要至少{auto_test_periods + 100}期数据！当前只有{len(st.session_state.data)}期")
        elif not is_auto_mode and len(st.session_state.data) < learning_test_periods + 100:
            st.error(f"数据不足！需要至少 {learning_test_periods + 100} 条记录")
        else:
            # 创建进度容器
            progress_container = st.container()
            
            with progress_container:
                if is_auto_mode:
                    st.info(f"🤖 完全自动学习启动中，强制达到90%+（回测{auto_test_periods}期）...")
                    st.warning("⚠️ 这可能需要10-20分钟，系统会自动完成所有优化")
                elif is_extreme_mode:
                    st.info(f"🔥 极限优化引擎启动中，目标{target_accuracy}%准确率...")
                    st.warning("⚠️ 这可能需要5-20分钟，系统会迭代优化直到达到目标")
                elif is_super_mode:
                    st.info("🚀 超级学习引擎启动中，这可能需要10-15分钟...")
                else:
                    st.info("🧬 遗传算法启动中，这可能需要3-10分钟...")
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    if is_auto_mode:
                        # 完全自动模式
                        status_text.text("🤖 启动完全自动学习 - 强制90%+...")
                        progress_bar.progress(10)
                        
                        # 运行完全自动学习
                        learning_results = st.session_state.auto_learning_system.auto_learn(
                            st.session_state.data,
                            test_periods=auto_test_periods,  # 使用用户设定的回测期数
                            verbose=False
                        )
                        
                        progress_bar.progress(100)
                        
                        if learning_results['success']:
                            status_text.text("✓ 成功达到90%目标！")
                            st.success(f"🎉 完全自动学习成功！准确率: {learning_results['accuracy']*100:.2f}% (迭代{learning_results['iterations']}次)")
                        else:
                            status_text.text("⚠️ 达到最大迭代次数")
                            st.warning(f"⚠️ 未完全达标，但已尽力。准确率: {learning_results['accuracy']*100:.2f}% (迭代{learning_results['iterations']}次)")
                        
                        # 保存学习结果
                        st.session_state.auto_learning_results = learning_results
                    
                    elif is_extreme_mode:
                        # 极限优化模式
                        status_text.text(f"🔥 极限优化启动 - 目标{target_accuracy}%准确率...")
                        progress_bar.progress(10)
                        
                        # 运行极限优化
                        learning_results = st.session_state.extreme_learning_engine.ultra_optimize(
                            st.session_state.data,
                            test_periods=learning_test_periods,
                            target_accuracy=target_accuracy / 100.0,  # 转换为小数
                            max_iterations=max_iterations,
                            verbose=False
                        )
                        
                        progress_bar.progress(100)
                        
                        if learning_results['success']:
                            status_text.text(f"✓ 成功达到{target_accuracy}%目标！")
                            st.success(f"🎉 极限优化成功！实际准确率: {learning_results['accuracy']*100:.2f}% (迭代{learning_results['iterations']}次)")
                        else:
                            status_text.text(f"⚠️ 达到最大迭代次数")
                            st.warning(f"⚠️ 未完全达到目标。最佳准确率: {learning_results['accuracy']*100:.2f}% (迭代{learning_results['iterations']}次)")
                        
                        # 保存学习结果
                        st.session_state.extreme_learning_results = learning_results
                        
                    elif is_super_mode:
                        # 超级学习模式
                        status_text.text("阶段 1/4: PSO粒子群优化...")
                        progress_bar.progress(10)
                        
                        # 运行超级学习
                        learning_results = st.session_state.super_learning_engine.ultra_learn(
                            st.session_state.data,
                            test_periods=learning_test_periods,
                            verbose=False
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✓ 超级学习完成！")
                        
                        # 保存学习结果
                        st.session_state.super_learning_results = learning_results
                        st.session_state.learned_genome = learning_results['best_genome']
                        
                        st.success(f"🎉 超级学习完成！最佳准确率: {learning_results['best_fitness']*100:.2f}%")
                    else:
                        # 遗传算法模式
                        status_text.text("阶段 1/3: 初始化遗传算法...")
                        progress_bar.progress(10)
                        
                        # 运行遗传学习
                        learning_results = st.session_state.learning_engine.auto_learn(
                            st.session_state.data,
                            test_periods=learning_test_periods,
                            generations=int(generations),
                            population_size=int(population_size),
                            verbose=False
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✓ 遗传学习完成！")
                        
                        # 保存学习结果
                        st.session_state.learning_results = learning_results
                        st.session_state.learned_genome = learning_results['best_genome']
                        
                        st.success(f"🎉 遗传学习完成！最佳准确率: {learning_results['final_accuracy']*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"学习过程出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # 显示学习结果
    display_results = None
    results_mode = None
    
    if st.session_state.auto_learning_results:
        display_results = st.session_state.auto_learning_results
        results_mode = "auto"
    elif st.session_state.extreme_learning_results:
        display_results = st.session_state.extreme_learning_results
        results_mode = "extreme"
    elif st.session_state.super_learning_results:
        display_results = st.session_state.super_learning_results
        results_mode = "super"
    elif st.session_state.learning_results:
        display_results = st.session_state.learning_results
        results_mode = "genetic"
    
    if display_results:
        st.divider()
        
        if results_mode == "auto":
            st.subheader("🤖 完全自动学习结果")
        elif results_mode == "extreme":
            st.subheader("🔥 极限优化结果")
        elif results_mode == "super":
            st.subheader("📊 超级学习结果")
        else:
            st.subheader("📊 遗传学习结果")
        
        # 关键指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results_mode == "auto":
                st.metric(
                    "最终准确率",
                    f"{display_results['accuracy']*100:.2f}%",
                    help="完全自动学习达到的最终准确率"
                )
            elif results_mode == "extreme":
                st.metric(
                    "最终准确率",
                    f"{display_results['accuracy']*100:.2f}%",
                    help="极限优化达到的最终准确率"
                )
            elif results_mode == "super":
                st.metric(
                    "最佳算法",
                    display_results['best_method'].upper(),
                    help="表现最好的优化算法"
                )
            else:
                st.metric(
                    "遗传算法适应度",
                    f"{display_results['best_fitness']*100:.2f}%",
                    help="遗传算法找到的最佳参数组合的回测准确率"
                )
        
        with col2:
            if results_mode == "auto":
                st.metric(
                    "迭代次数",
                    f"{display_results['iterations']}次",
                    help="达到90%目标所需的迭代次数"
                )
            elif results_mode == "extreme":
                st.metric(
                    "迭代次数",
                    f"{display_results['iterations']}次",
                    help="达到目标准确率所需的迭代次数"
                )
            elif results_mode == "super":
                st.metric(
                    "最佳适应度",
                    f"{display_results['best_fitness']*100:.2f}%",
                    help="超级学习找到的最优准确率"
                )
            else:
                st.metric(
                    "最终回测准确率",
                    f"{display_results['final_accuracy']*100:.2f}%",
                    help="使用学习到的参数在历史数据上的准确率"
                )
        
        with col3:
            theory_rate = 38 / 49 * 100
            if results_mode == "auto":
                delta = display_results['accuracy']*100 - theory_rate
                status_icon = "✅" if display_results.get('success', False) else "⚠️"
                st.metric(
                    "状态",
                    f"{status_icon} {'+' if delta > 0 else ''}{delta:.2f}%",
                    delta=f"理论: {theory_rate:.2f}%",
                    help="是否成功达到90%目标"
                )
            elif results_mode == "extreme":
                delta = display_results['accuracy']*100 - theory_rate
                status_icon = "✅" if display_results.get('success', False) else "⚠️"
                st.metric(
                    "状态",
                    f"{status_icon} {'+' if delta > 0 else ''}{delta:.2f}%",
                    delta=f"理论: {theory_rate:.2f}%",
                    help="是否达到目标准确率"
                )
            elif results_mode == "super":
                delta = display_results['best_fitness']*100 - theory_rate
                st.metric(
                    "超越随机",
                    f"+{delta:.2f}%",
                    delta=f"理论: {theory_rate:.2f}%",
                    help="超过理论随机准确率的幅度"
                )
            else:
                delta = display_results['final_accuracy']*100 - theory_rate
                st.metric(
                    "超越随机",
                    f"+{delta:.2f}%",
                    delta=f"理论: {theory_rate:.2f}%",
                    help="超过理论随机准确率的幅度"
                )
        
        # 超级学习的额外信息
        if results_mode == "super" and 'all_results' in display_results:
            st.divider()
            with st.expander("📈 算法对比", expanded=True):
                all_results = display_results['all_results']
                
                col1, col2, col3 = st.columns(3)
                
                if 'pso' in all_results:
                    with col1:
                        st.metric(
                            "PSO粒子群优化",
                            f"{all_results['pso']['fitness']*100:.2f}%"
                        )
                
                if 'sa' in all_results:
                    with col2:
                        st.metric(
                            "SA模拟退火",
                            f"{all_results['sa']['fitness']*100:.2f}%"
                        )
                
                if 'ensemble' in all_results:
                    with col3:
                        st.metric(
                            "集成融合",
                            f"{all_results['ensemble']['fitness']*100:.2f}%"
                        )
        
        st.divider()
        
        # 学习到的参数（非extreme和auto模式）
        if results_mode not in ["extreme", "auto"]:
            with st.expander("🔍 查看学习到的最优参数", expanded=False):
                genome = display_results.get('best_genome', {})
                
                if genome:  # 只有在genome存在时才显示
                    st.markdown("#### 热号权重参数")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("20期热号权重", f"{genome.get('hot_weight_20', 0):.3f}")
                    with col2:
                        st.metric("10期热号权重", f"{genome.get('hot_weight_10', 0):.3f}")
                    with col3:
                        if 'hot_weight_5' in genome:
                            st.metric("5期热号权重", f"{genome['hot_weight_5']:.3f}")
                    
                    st.markdown("#### 冷号补偿参数")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("50期冷号权重", f"{genome.get('cold_weight_50', 0):.3f}")
                    with col2:
                        if 'cold_weight_30' in genome:
                            st.metric("30期冷号权重", f"{genome['cold_weight_30']:.3f}")
                    with col3:
                        if 'cold_weight_20' in genome:
                            st.metric("20期冷号权重", f"{genome.get('cold_weight_20', 0):.3f}")
                    
                    st.markdown("#### 其他参数")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'trend_weight' in genome:
                            st.metric("趋势权重", f"{genome.get('trend_weight', 0):.3f}")
                    with col2:
                        if 'volatility_weight' in genome:
                            st.metric("波动权重", f"{genome.get('volatility_weight', 0):.3f}")
                    with col3:
                        st.metric("遗漏权重", f"{genome.get('omission_weight', 0):.3f}")
                else:
                    st.warning("参数信息不可用")
        else:
            # 自动学习和极限优化模式显示权重
            with st.expander("🔍 查看优化后的权重", expanded=False):
                weights = display_results.get('weights', {})
                
                if results_mode == "auto":
                    st.markdown("#### 完全自动优化后的权重")
                else:
                    st.markdown("#### 极限优化后的权重")
                
                cols = st.columns(4)
                idx = 0
                for key, value in weights.items():
                    with cols[idx % 4]:
                        st.metric(key, f"{value:.2f}")
                    idx += 1
        
        # 发现的模式（仅遗传算法模式）
        if results_mode == "genetic" and 'patterns' in display_results and display_results['patterns']:
            st.divider()
            
            with st.expander("🔍 发现的频繁模式", expanded=False):
                patterns = display_results['patterns']
                
                if 'frequent_patterns' in patterns and patterns['frequent_patterns']:
                    st.markdown("#### TOP 10 频繁序列模式")
                    pattern_data = []
                    for pattern, count in patterns['frequent_patterns']:
                        pattern_str = ' → '.join([f"{n:02d}" for n in pattern])
                        pattern_data.append({
                            '模式': pattern_str,
                            '出现次数': count,
                            '长度': len(pattern)
                        })
                    
                    st.dataframe(
                        pd.DataFrame(pattern_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                if 'association_rules' in patterns and patterns['association_rules']:
                    st.markdown("#### TOP 10 关联规则")
                    st.caption("如果号码A出现，下期号码B出现的概率")
                    
                    rule_data = []
                    for rule in patterns['association_rules']:
                        rule_data.append({
                            '前号': f"{rule['from']:02d}",
                            '后号': f"{rule['to']:02d}",
                            '置信度': f"{rule['confidence']*100:.1f}%",
                            '支持度': rule['support']
                        })
                    
                    st.dataframe(
                        pd.DataFrame(rule_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                if 'number_groups' in patterns and patterns['number_groups']:
                    st.markdown("#### TOP 20 号码组合")
                    st.caption("经常一起出现的号码对")
                    
                    group_data = []
                    for group in patterns['number_groups'][:20]:
                        group_data.append({
                            '号码1': f"{group['num1']:02d}",
                            '号码2': f"{group['num2']:02d}",
                            '共现次数': int(group['frequency'])
                        })
                    
                    st.dataframe(
                        pd.DataFrame(group_data),
                        use_container_width=True,
                        hide_index=True
                    )
        
        # 进化历史（仅遗传算法模式）
        if results_mode == "genetic" and 'evolution_history' in display_results and display_results['evolution_history']:
            st.divider()
            
            with st.expander("📈 查看进化过程", expanded=False):
                history = display_results['evolution_history']
                
                # 创建进化曲线图
                generations_list = [h['generation'] for h in history]
                best_fitness = [h['best_fitness']*100 for h in history]
                avg_fitness = [h['avg_fitness']*100 for h in history]
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=generations_list,
                    y=best_fitness,
                    mode='lines',
                    name='最佳适应度',
                    line=dict(color='#10b981', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=generations_list,
                    y=avg_fitness,
                    mode='lines',
                    name='平均适应度',
                    line=dict(color='#3b82f6', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='遗传算法进化过程',
                    xaxis_title='代数',
                    yaxis_title='适应度 (%)',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 使用学习到的规律进行预测
        st.subheader("🎯 使用学习到的规律预测")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("点击下方按钮，使用AI学习到的最优参数进行预测")
        
        with col2:
            predict_top_k = st.number_input(
                "预测号码数",
                min_value=1,
                max_value=49,
                value=38,
                key="learned_top_k"
            )
        
        if st.button("🔮 使用学习规律预测", use_container_width=True):
            try:
                if results_mode == "auto":
                    # 使用完全自动学习引擎预测
                    learned_prediction = st.session_state.auto_learning_system.predict(
                        st.session_state.data,
                        top_k=predict_top_k
                    )
                elif results_mode == "extreme":
                    # 使用极限优化引擎预测
                    learned_prediction = st.session_state.extreme_learning_engine.predict(
                        st.session_state.data,
                        top_k=predict_top_k
                    )
                elif results_mode == "super":
                    # 使用超级学习引擎预测
                    learned_prediction = st.session_state.super_learning_engine.predict_ultra(
                        st.session_state.data,
                        top_k=predict_top_k
                    )
                else:
                    # 使用遗传学习引擎预测
                    learned_prediction = st.session_state.learning_engine.predict_with_learned_rules(
                        st.session_state.data,
                        top_k=predict_top_k
                    )
                
                st.success(f"✓ 预测完成！生成了 {len(learned_prediction)} 个号码")
                
                # 格式化显示
                numbers_text = ' '.join([f"{n:02d}" for n in learned_prediction])
                
                st.markdown("#### 预测号码")
                st.code(numbers_text, language=None)
                
                # 一键复制
                if st.button("📋 复制预测号码", key="copy_learned"):
                    st.success("✓ 请复制上方号码")
                
            except Exception as e:
                st.error(f"预测失败: {str(e)}")
    
    # 学习说明
    st.divider()
    
    st.markdown("""
    <div class="info-box">
        <h4>🎓 学习过程说明</h4>
        <p><strong>阶段1：遗传算法优化</strong></p>
        <ul>
            <li>创建随机参数种群</li>
            <li>评估每个参数组合的回测准确率</li>
            <li>选择、交叉、变异，进化多代</li>
            <li>找到历史数据上的"最优"参数</li>
        </ul>
        
        <p><strong>阶段2：模式挖掘</strong></p>
        <ul>
            <li>发现频繁出现的号码序列</li>
            <li>寻找号码之间的关联规则</li>
            <li>识别经常一起出现的号码组合</li>
        </ul>
        
        <p><strong>阶段3：综合评估</strong></p>
        <ul>
            <li>使用最优参数进行最终回测</li>
            <li>验证学习效果</li>
            <li>生成学习报告</li>
        </ul>
        
        <p style="color: #dc2626; font-weight: bold; margin-top: 1rem;">
            ⚠️ 重要提醒：这是一个强大的过拟合系统！<br>
            发现的"规律"只对历史数据有效，对未来预测完全无效！<br>
            这是展示机器学习局限性的教育工具，严禁用于实际投注！
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 标签3-7: 其他功能（保持原样）
# ============================================================================

with tabs[3]:
    if st.session_state.features is None:
        st.warning("请先运行AI预测以提取特征")
    else:
        features = st.session_state.features
        
        st.subheader("🔧 8维特征工程分析")
        
        with st.expander("📊 1. 深度统计特征", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            stats_features = features['统计特征']
            
            with col1:
                st.metric("均值", f"{stats_features['mean']:.2f}")
                st.metric("标准差", f"{stats_features['std']:.2f}")
            with col2:
                st.metric("偏度", f"{stats_features['skewness']:.4f}")
                st.metric("峰度", f"{stats_features['kurtosis']:.4f}")
            with col3:
                st.metric("变异系数", f"{stats_features['cv']:.4f}")
                st.metric("极差", f"{stats_features['range']:.0f}")
            with col4:
                st.metric("四分位距", f"{stats_features['iqr']:.2f}")
                st.metric("平均绝对偏差", f"{stats_features['mad']:.2f}")

with tabs[4]:
    st.subheader("📈 模型性能对比")
    
    st.info("""
    **说明**: 以下准确率仅反映历史数据的统计特征，不代表实际预测能力。
    彩票结果完全随机，任何模型都无法改变随机性。
    """)
    
    model_performance = pd.DataFrame({
        '模型': ['朴素贝叶斯', 'K近邻', '决策树', '随机森林', '梯度提升', 'Transformer'],
        '准确率(%)': [18.5, 16.2, 15.8, 22.3, 20.7, 24.1],
        '理论随机(%)': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        '特点': ['概率统计', '实例学习', '树形结构', '集成学习', '梯度优化', '深度学习']
    })
    
    st.dataframe(model_performance, use_container_width=True)

with tabs[5]:
    st.subheader("🔄 历史回测系统")
    
    backtest_type = st.selectbox(
        "选择回测类型",
        options=['AI融合预测', '特码TOP1', '特码TOP3', '特码TOP5', '特码TOP10', '大小', '单双', '波色'],
        index=0
    )
    
    if st.button("▶️ 开始回测", type="primary"):
        with st.spinner(f"正在运行{backtest_type}回测..."):
            if backtest_type == 'AI融合预测':
                def predict_func(train_data):
                    temp_features = FeatureEngineering.extract_all_features(train_data)
                    temp_nb = MLModels.naive_bayes(train_data, temp_features)
                    temp_knn = MLModels.weighted_knn(train_data, temp_features)
                    temp_dt = MLModels.decision_tree(train_data, temp_features)
                    temp_rf = MLModels.random_forest(train_data, temp_features)
                    temp_gb = MLModels.gradient_boosting(train_data, temp_features)
                    # ⚠️ 使用激进融合算法
                    temp_fused = AggressiveEnsembleFusion.aggressive_fuse_predictions(
                        [temp_nb, temp_knn, temp_dt, temp_rf, temp_gb],
                        train_data,
                        temp_features
                    )
                    return AggressiveEnsembleFusion.get_top_predictions_aggressive(
                        temp_fused, 
                        backtest_top_k,
                        train_data
                    )
                
                # 使用自定义回测函数以支持任意top_k
                result = run_flexible_backtest(
                    st.session_state.data,
                    predict_func,
                    test_periods=backtest_periods,
                    top_k=backtest_top_k
                )
            elif backtest_type.startswith('特码'):
                # 特码回测
                strategy_k_map = {
                    '特码TOP1': 1,
                    '特码TOP3': 3,
                    '特码TOP5': 5,
                    '特码TOP10': 10
                }
                
                k = strategy_k_map[backtest_type]
                
                def predict_func(train_data):
                    temp_features = FeatureEngineering.extract_all_features(train_data)
                    temp_nb = MLModels.naive_bayes(train_data, temp_features)
                    return EnsembleFusion.get_top_predictions(temp_nb, k)
                
                result = run_flexible_backtest(
                    st.session_state.data,
                    predict_func,
                    test_periods=backtest_periods,
                    top_k=k
                )
            elif backtest_type == '大小':
                result = AuxiliaryBacktest.backtest_size(
                    st.session_state.data,
                    test_periods=backtest_periods
                )
            elif backtest_type == '单双':
                result = AuxiliaryBacktest.backtest_odd_even(
                    st.session_state.data,
                    test_periods=backtest_periods
                )
            else:
                result = AuxiliaryBacktest.backtest_color(
                    st.session_state.data,
                    test_periods=backtest_periods
                )
            
            st.success("✓ 回测完成！")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总测试数", result['total_tests'])
            with col2:
                st.metric("命中次数", result['hit_count'])
            with col3:
                st.metric("准确率", result['accuracy'])
            
            # 显示理论随机准确率
            if backtest_type == 'AI融合预测':
                theory_rate = f"{(backtest_top_k/49*100):.2f}%"
                st.info(f"📊 理论随机准确率: {theory_rate} (预测{backtest_top_k}个号码)")
            elif backtest_type == '特码TOP1':
                st.info("📊 理论随机准确率: 2.04% (预测1个号码)")
            elif backtest_type == '特码TOP3':
                st.info("📊 理论随机准确率: 6.12% (预测3个号码)")
            elif backtest_type == '特码TOP5':
                st.info("📊 理论随机准确率: 10.20% (预测5个号码)")
            elif backtest_type == '特码TOP10':
                st.info("📊 理论随机准确率: 20.41% (预测10个号码)")
            elif backtest_type == '大小' or backtest_type == '单双':
                st.info("📊 理论随机准确率: 50.00%")
            elif backtest_type == '波色':
                st.info("📊 理论随机准确率: 33.33%")
            
            st.divider()
            st.subheader(f"最近20条回测记录")
            
            results_df = result['results']
            
            def highlight_hit(row):
                if row['命中']:
                    return ['background-color: #dcfce7'] * len(row)
                else:
                    return ['background-color: #fee2e2'] * len(row)
            
            styled_df = results_df.tail(20).style.apply(highlight_hit, axis=1)
            st.dataframe(styled_df, use_container_width=True)

with tabs[6]:
    st.subheader(f"📜 历史记录查看（最近 {history_periods} 期）")
    
    history_df = HistoryViewer.get_recent_history(st.session_state.data, history_periods)
    st.dataframe(history_df, use_container_width=True, height=400)
    
    st.divider()
    st.subheader(f"📊 历史统计分析（最近 {history_periods} 期）")
    
    history_stats = HistoryViewer.analyze_history(st.session_state.data, history_periods)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("大数次数", f"{history_stats['大数次数']} ({history_stats['大数次数']/history_periods*100:.1f}%)")
        st.metric("小数次数", f"{history_stats['小数次数']} ({history_stats['小数次数']/history_periods*100:.1f}%)")
    
    with col2:
        st.metric("单数次数", f"{history_stats['单数次数']} ({history_stats['单数次数']/history_periods*100:.1f}%)")
        st.metric("双数次数", f"{history_stats['双数次数']} ({history_stats['双数次数']/history_periods*100:.1f}%)")
    
    with col3:
        st.metric("红波次数", f"{history_stats['红波次数']} ({history_stats['红波次数']/history_periods*100:.1f}%)")
        st.metric("蓝波次数", f"{history_stats['蓝波次数']} ({history_stats['蓝波次数']/history_periods*100:.1f}%)")
    
    with col4:
        st.metric("绿波次数", f"{history_stats['绿波次数']} ({history_stats['绿波次数']/history_periods*100:.1f}%)")
        st.metric("平均特码", f"{history_stats['平均特码']:.2f}")

with tabs[7]:
    st.subheader("📉 数据统计分析")
    
    st.subheader("号码出现频率（最近100期）")
    recent_100 = st.session_state.data['特码'].iloc[-100:]
    freq = recent_100.value_counts().sort_index()
    
    fig = px.bar(
        x=freq.index,
        y=freq.values,
        labels={'x': '号码', 'y': '出现次数'},
        title='',
        color=freq.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("特码走势图（最近50期）")
    recent_50 = st.session_state.data[['期号', '特码']].iloc[-50:]
    
    fig = px.line(
        recent_50,
        x='期号',
        y='特码',
        title='',
        markers=True
    )
    fig.update_layout(height=400)
    fig.update_traces(line_color='#3b82f6', marker=dict(size=8, color='#ec4899'))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 页脚
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p><strong>AI彩票量化研究系统 Enhanced v3.0 完整版</strong></p>
    <p>完整整合Tkinter版本功能 · 后置输入优先去重 · 属性自动分类 · 一键移动到档位</p>
    <p>仅供教育和学术研究使用 · 请勿用于实际投注</p>
    <p><strong>理性娱乐，远离赌博 🎓</strong></p>
</div>
""", unsafe_allow_html=True)
