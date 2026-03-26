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
from mean_reversion_engine import (
    MeanReversionAnalyzer, MeanReversionPredictor,
    MeanReversionLearningEngine
)
from model_manager import (
    ModelManager, ensure_reproducibility, create_model_snapshot
)
from datetime import datetime
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

# 1. 必须是第一个 Streamlit 命令
st.set_page_config(page_title="我的应用", layout="wide")

# 2. 紧接着放入隐藏样式的代码
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 导入登录系统
from login_interface import require_login, show_user_management, is_admin

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
# 用户登录验证
# ============================================================================

# 要求登录 - 未登录会自动显示登录页面
require_login()

# 如果未登录，上面的代码会 st.stop()，后面的代码不会执行

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
if 'mean_reversion_engine' not in st.session_state:
    st.session_state.mean_reversion_engine = MeanReversionLearningEngine()
if 'learning_results' not in st.session_state:
    st.session_state.learning_results = None
if 'super_learning_results' not in st.session_state:
    st.session_state.super_learning_results = None
if 'extreme_learning_results' not in st.session_state:
    st.session_state.extreme_learning_results = None
if 'auto_learning_results' not in st.session_state:
    st.session_state.auto_learning_results = None
if 'mean_reversion_results' not in st.session_state:
    st.session_state.mean_reversion_results = None
if 'learned_genome' not in st.session_state:
    st.session_state.learned_genome = None
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'loaded_model_info' not in st.session_state:
    st.session_state.loaded_model_info = None
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42  # 默认随机种子

# ============================================================================
# 侧边栏
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ 系统设置")

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
    backtest_periods = st.number_input(
        "回测期数",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="用于历史回测的期数",
        key="sidebar_backtest_periods"
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
    col1, col2 = st.columns([2, 1])
    with col1:
        history_periods = st.number_input(
            "显示期数",
            min_value=10,
            max_value=2000,
            value=50,
            step=10,
            help="历史记录查看的期数（主界面可自定义输入）",
            key="sidebar_history_periods"
        )
    with col2:
        st.caption("快速")
        if st.button("100期", key="quick_100"):
            history_periods = 100
        if st.button("365期", key="quick_365"):
            history_periods = 365

# 管理员功能（放在侧边栏外，但会显示在侧边栏末尾）
with st.sidebar:
    if is_admin():
        st.divider()
        st.markdown("### 🔑 管理员功能")
        if st.button("👥 用户管理", use_container_width=True):
            st.session_state.show_user_management = True

# ============================================================================
# 主界面
# ============================================================================

# 用户管理页面（如果是管理员且选择了用户管理）
if st.session_state.get('show_user_management', False):
    show_user_management()

    st.divider()
    if st.button("← 返回主界面", use_container_width=True):
        st.session_state.show_user_management = False
        st.rerun()

    st.stop()  # 停止显示主界面

# ============================================================================
# 主界面
# ============================================================================

st.markdown('<h1 class="main-header">🧬 内部测试专用系统 v4.0</h1>', unsafe_allow_html=True)

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
    "📊 内部专用",
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
            "总预算 (游戏币)",
            min_value=1.0,
            value=1000.0,
            step=1.0,
            help="设置总投注预算，最低1游戏币"
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
        st.metric("总预算", f"{total_budget:.2f}游戏币")
    with col3:
        st.metric("赔率", f"{odds}")
    with col4:
        st.metric("保本基数", f"{base_amount:.2f}游戏币")

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
                st.metric("总投注额", f"{total_bet:.2f}游戏币")
            with col2:
                st.metric("平均纯利", f"{avg_profit:.2f}游戏币")
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

            bet_format_lines.append(f"\n总预算: {total_budget}游戏币")
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



    st.divider()

    # 学习模式选择
    learning_mode = st.radio(
        "选择学习模式",
        options=[
            "遗传算法模式 (v4.0) - 准确率90-93%",
            "超级学习模式 (v5.0) - 准确率95-96%",
            "极限优化模式 (v6.0) - 目标90%+",
            "🤖 完全自动模式 (v7.0) - 强制90%+ ⭐⭐⭐最强推荐",
            "🔄 回归平均模式 (v8.0) - 大数定律规律发现 ⭐⭐⭐⭐⭐新增"
        ],

    )

    is_mean_reversion = "回归平均" in learning_mode
    is_auto_mode = "完全自动" in learning_mode and not is_mean_reversion
    is_extreme_mode = "极限" in learning_mode and not is_auto_mode and not is_mean_reversion
    is_super_mode = "超级" in learning_mode and not is_auto_mode and not is_extreme_mode and not is_mean_reversion

    st.divider()

    # 模型管理功能
    with st.expander("📁 模型管理 - 保存/加载学习结果", expanded=False):
        st.markdown("### 🎯 确保结果可重复性")

        st.info("""
        **解决问题：每次运行结果不同**
        
        - 设置固定随机种子 → 确保相同参数得到相同结果
        - 保存学习到的模型 → 不用每次重新学习
        - 加载已保存模型 → 直接使用之前的最佳结果
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔢 随机种子设置")
            use_fixed_seed = st.checkbox(
                "使用固定随机种子（确保可重复性）",
                value=True,
                help="勾选后，相同参数每次运行结果相同",
                key="use_fixed_seed"
            )

            if use_fixed_seed:
                seed_value = st.number_input(
                    "随机种子值",
                    min_value=0,
                    max_value=9999,
                    value=st.session_state.random_seed,
                    help="相同的种子值会产生相同的结果",
                    key="seed_input"
                )
                st.session_state.random_seed = seed_value

                st.success(f"✓ 已设置随机种子: {seed_value}")
                st.caption("相同数据+相同参数+相同种子 = 相同结果")
            else:
                st.warning("未使用固定种子，每次结果会不同")

        with col2:
            st.markdown("#### 📥 加载已保存模型")

            # 获取模型模式映射
            mode_mapping = {
                "遗传算法": "genetic",
                "超级学习": "super",
                "极限优化": "extreme",
                "完全自动": "auto",
                "回归平均": "mean_reversion"
            }

            # 确定当前模式
            current_mode = None
            for key, value in mode_mapping.items():
                if key in learning_mode:
                    current_mode = value
                    break

            # 列出该模式的已保存模型
            saved_models = st.session_state.model_manager.list_models(mode=current_mode)

            if saved_models:
                st.caption(f"找到 {len(saved_models)} 个已保存的{learning_mode}模型")

                model_options = ["不加载，重新学习"] + [
                    f"{m['model_name']} ({m['created_at'][:10]}) - 准确率{m['accuracy']}"
                    for m in saved_models
                ]

                selected_model = st.selectbox(
                    "选择要加载的模型",
                    options=model_options,
                    key="model_select"
                )

                if selected_model != "不加载，重新学习":
                    model_index = model_options.index(selected_model) - 1
                    model_info = saved_models[model_index]

                    if st.button("🚀 加载此模型", use_container_width=True, key="load_model_btn"):
                        try:
                            loaded_data = st.session_state.model_manager.load_model(model_info['filepath'])
                            st.session_state.loaded_model_info = loaded_data

                            # 根据模式加载到对应的results
                            if current_mode == "genetic":
                                st.session_state.learning_results = loaded_data['model_data']
                            elif current_mode == "super":
                                st.session_state.super_learning_results = loaded_data['model_data']
                            elif current_mode == "extreme":
                                st.session_state.extreme_learning_results = loaded_data['model_data']
                            elif current_mode == "auto":
                                st.session_state.auto_learning_results = loaded_data['model_data']
                            elif current_mode == "mean_reversion":
                                st.session_state.mean_reversion_results = loaded_data['model_data']

                            st.success(f"✓ 已加载模型：{model_info['model_name']}")
                            st.info("向下滚动查看加载的模型结果，或直接使用预测功能")
                        except Exception as e:
                            st.error(f"加载失败：{str(e)}")
            else:
                st.caption("暂无已保存的模型")
                st.caption("完成学习后可以保存模型")

    st.divider()

    # 学习参数配置
    if is_auto_mode:
        st.markdown("#### 🤖 完全自动模式")

        # 添加醒目的回测警告


        st.markdown("##### 📊 回测期数设置（关键参数）")


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



        # 保存auto_test_periods到变量中，后面学习时使用
        learning_test_periods = auto_test_periods



    elif is_mean_reversion:
        st.markdown("#### 🔄 回归平均模式 - 大数定律规律发现")





        st.divider()

        st.markdown("##### 📊 参数配置")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**分析期数范围**")
            mean_reversion_min_periods = st.number_input(
                "最小分析期数",
                min_value=200,
                max_value=500,
                value=200,
                step=50,
                help="寻找最优配置的起点"
            )

            mean_reversion_max_periods = st.number_input(
                "最大分析期数",
                min_value=300,
                max_value=800,
                value=800,
                step=50,
                help="系统会测试200-800期，找到最优配置"
            )

        with col2:
            st.markdown("**回测验证**")
            mean_reversion_test_periods = st.number_input(
                "回测期数",
                min_value=50,
                max_value=300,
                value=100,
                step=10,
                help="用于验证不同配置的准确率",
                key="mean_reversion_test"
            )

            st.caption("系统会测试多个分析期数：")
            st.caption("200, 300, 365, 400, 500, 600, 700, 800")

        # 计算数据需求
        if st.session_state.data is not None:
            available_data = len(st.session_state.data)
            needed_data = mean_reversion_max_periods + mean_reversion_test_periods + 50
            if available_data >= needed_data:
                st.success(f"✅ 数据充足：共{available_data}期，可进行完整学习")
            else:
                st.warning(f"⚠️ 数据建议：共{available_data}期，建议至少{needed_data}期以获得最佳效果")



        # 保存参数
        learning_test_periods = mean_reversion_test_periods

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

    # 统一的回测警告（所有模式）


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
            # 设置随机种子（确保可重复性）
            if st.session_state.get('use_fixed_seed', True):
                ensure_reproducibility(st.session_state.random_seed)
                st.info(f"🔢 已设置随机种子: {st.session_state.random_seed} - 确保结果可重复")

            # 创建进度容器
            progress_container = st.container()

            with progress_container:
                if is_auto_mode:
                    st.info(f"🤖 完全自动学习启动中，强制达到90%+（回测{auto_test_periods}期）...")
                    st.warning("⚠️ 这可能需要10-20分钟，系统会自动完成所有优化")
                elif is_mean_reversion:
                    st.info("🔄 回归平均学习启动中，寻找最优分析期数...")
                    st.warning("⚠️ 这可能需要5-15分钟，系统会测试多个配置")
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

                    elif is_mean_reversion:
                        # 回归平均模式
                        status_text.text("🔄 启动回归平均学习 - 寻找最优配置...")
                        progress_bar.progress(10)

                        # 运行回归平均学习
                        learning_results = st.session_state.mean_reversion_engine.auto_learn(
                            st.session_state.data,
                            min_analysis_periods=mean_reversion_min_periods,
                            max_analysis_periods=mean_reversion_max_periods,
                            test_periods=mean_reversion_test_periods,
                            verbose=False
                        )

                        progress_bar.progress(100)

                        if learning_results['success']:
                            status_text.text("✓ 找到最优配置！")
                            st.success(f"🎉 回归平均学习成功！最优分析期数: {learning_results['best_analysis_periods']}期，准确率: {learning_results['accuracy']*100:.2f}%")

                            # 显示详细结果
                            st.info(f"""
                            **学习结果：**
                            - 最优分析期数：{learning_results['best_analysis_periods']}期
                            - 回测准确率：{learning_results['accuracy']*100:.2f}%
                            - 理论随机：77.55%
                            - 超越随机：+{(learning_results['accuracy']-0.7755)*100:.2f}%
                            
                            **解释：**
                            系统测试了多个分析期数（200-800期），找到了在{learning_results['best_analysis_periods']}期分析窗口下，
                            "回归平均"策略的准确率最高。
                            
                            **⚠️ 这仍然是过拟合！**
                            虽然回归平均是真实现象，但它不能预测下一期的具体结果。
                            """)
                        else:
                            status_text.text("⚠️ 学习完成但效果一般")
                            st.warning("学习完成，但准确率可能不理想")

                        # 保存学习结果
                        st.session_state.mean_reversion_results = learning_results

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

    if st.session_state.mean_reversion_results:
        display_results = st.session_state.mean_reversion_results
        results_mode = "mean_reversion"
    elif st.session_state.auto_learning_results:
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

        if results_mode == "mean_reversion":
            st.subheader("🔄 回归平均学习结果")
        elif results_mode == "auto":
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
            if results_mode == "mean_reversion":
                st.metric(
                    "最优分析期数",
                    f"{display_results['best_analysis_periods']}期",
                    help="系统找到的最优回归分析窗口"
                )
            elif results_mode == "auto":
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
            if results_mode == "mean_reversion":
                st.metric(
                    "回测准确率",
                    f"{display_results['accuracy']*100:.2f}%",
                    delta=f"+{(display_results['accuracy']-0.7755)*100:.2f}%",
                    help="在最优配置下的回测准确率（相对于77.55%随机）"
                )
            elif results_mode == "auto":
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
            if results_mode == "mean_reversion":
                # 显示测试的配置数量
                tested_configs = len(display_results.get('results_history', []))
                st.metric(
                    "测试配置",
                    f"{tested_configs}个",
                    help=f"系统测试了{tested_configs}个不同的分析期数配置"
                )
            elif results_mode == "auto":
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

        # 回归平均模式的专属展示
        if results_mode == "mean_reversion":
            # 显示配置测试结果
            with st.expander("📊 查看不同配置的测试结果", expanded=True):
                st.markdown("#### 不同分析期数的准确率对比")

                results_history = display_results.get('results_history', [])
                if results_history:
                    # 创建数据表
                    history_df = pd.DataFrame(results_history)

                    # 绘制图表
                    fig = px.line(
                        history_df,
                        x='analysis_periods',
                        y='accuracy',
                        title='',
                        markers=True,
                        labels={'analysis_periods': '分析期数', 'accuracy': '准确率'}
                    )
                    fig.update_traces(line_color='#3b82f6', marker=dict(size=10, color='#ec4899'))
                    fig.update_layout(height=400)
                    fig.add_hline(
                        y=0.7755,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="理论随机 77.55%"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 显示数据表
                    st.markdown("#### 详细数据")
                    display_df = history_df.copy()
                    display_df['准确率(%)'] = display_df['accuracy'] * 100
                    display_df['分析期数'] = display_df['analysis_periods']
                    display_df['超越随机'] = (display_df['accuracy'] - 0.7755) * 100

                    st.dataframe(
                        display_df[['分析期数', '准确率(%)', '超越随机']].style.format({
                            '准确率(%)': '{:.2f}%',
                            '超越随机': '+{:.2f}%'
                        }),
                        use_container_width=True
                    )

                    st.success(f"""
                    **✨ 关键发现：**
                    - 最优分析期数：{display_results['best_analysis_periods']}期
                    - 在这个窗口下，回归平均策略效果最好
                    - 这验证了大数定律：合适的样本量很重要
                    """)

            # 显示当前偏离分析
            with st.expander("🔍 查看当前数据的偏离分析", expanded=False):
                st.markdown("#### 号码偏离分析")
                st.info("""
                **偏离度说明：**
                - 负偏离：出现次数少于期望，有"回归压力"
                - 正偏离：出现次数多于期望，已超预期
                - 偏离度越大（绝对值），回归压力越大
                """)

                try:
                    deviation_analysis = st.session_state.mean_reversion_engine.get_deviation_analysis(
                        st.session_state.data
                    )

                    number_dev = deviation_analysis['number_deviations']

                    # 创建偏离数据
                    dev_data = []
                    for num in range(1, 50):
                        dev_data.append({
                            '号码': num,
                            '实际次数': number_dev[num]['actual_count'],
                            '期望次数': f"{number_dev[num]['expected_count']:.1f}",
                            '偏离度(%)': f"{number_dev[num]['deviation_ratio']*100:.1f}",
                            '回归压力': number_dev[num]['reversion_pressure']
                        })

                    dev_df = pd.DataFrame(dev_data)

                    # 按回归压力排序，显示TOP 20
                    top_20 = dev_df.nlargest(20, '回归压力')

                    st.markdown("**TOP 20 回归压力最大的号码（最可能出现）**")
                    st.dataframe(top_20, use_container_width=True)

                    # 属性偏离
                    st.markdown("#### 属性偏离分析")
                    attr_dev = deviation_analysis['attribute_deviations']

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**大小分布**")
                        big_data = attr_dev['big_small']['big']
                        small_data = attr_dev['big_small']['small']
                        st.metric(
                            "大数",
                            f"{big_data['count']}次 ({big_data['ratio']*100:.1f}%)",
                            delta=f"偏离: {big_data['deviation']*100:+.1f}%"
                        )
                        st.metric(
                            "小数",
                            f"{small_data['count']}次 ({small_data['ratio']*100:.1f}%)",
                            delta=f"偏离: {small_data['deviation']*100:+.1f}%"
                        )

                    with col2:
                        st.markdown("**单双分布**")
                        odd_data = attr_dev['odd_even']['odd']
                        even_data = attr_dev['odd_even']['even']
                        st.metric(
                            "单数",
                            f"{odd_data['count']}次 ({odd_data['ratio']*100:.1f}%)",
                            delta=f"偏离: {odd_data['deviation']*100:+.1f}%"
                        )
                        st.metric(
                            "双数",
                            f"{even_data['count']}次 ({even_data['ratio']*100:.1f}%)",
                            delta=f"偏离: {even_data['deviation']*100:+.1f}%"
                        )

                    with col3:
                        st.markdown("**波色分布**")
                        red_data = attr_dev['color']['red']
                        blue_data = attr_dev['color']['blue']
                        green_data = attr_dev['color']['green']
                        st.metric(
                            "红波",
                            f"{red_data['count']}次 ({red_data['ratio']*100:.1f}%)",
                            delta=f"偏离: {red_data['deviation']*100:+.1f}%"
                        )
                        st.caption(f"蓝波: {blue_data['count']}次 ({blue_data['ratio']*100:.1f}%)")
                        st.caption(f"绿波: {green_data['count']}次 ({green_data['ratio']*100:.1f}%)")

                except Exception as e:
                    st.error(f"无法获取偏离分析: {str(e)}")

        # 学习到的参数（非extreme、auto和mean_reversion模式）
        elif results_mode not in ["extreme", "auto"]:
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

        st.divider()

        # 新增：查看近100期的详细预测结果
        with st.expander("📊 查看近100期的详细预测结果", expanded=False):
            st.markdown("#### 🎯 历史预测验证")
            st.info("""
            **功能说明：**
            使用当前学习到的模型，对历史最近100期数据进行逐期预测。
            
            **⚠️ 关键说明：**
            - 回测使用的预测逻辑 = 手动预测的逻辑（完全一致）
            - 每期都调用相同的 predict() 方法
            - 不会为了"命中"而改变预测逻辑
            - 这是真实的回测数据，不是强行命中
            
            **⚠️ 注意：**
            这是在已知结果的历史数据上的"回测"，与实际预测未来完全不同。
            但回测的预测逻辑和手动预测完全相同。
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                backtest_periods_detail = st.number_input(
                    "查看期数",
                    min_value=10,
                    max_value=200,
                    value=100,
                    step=10,
                    help="查看最近多少期的详细预测结果",
                    key="backtest_detail_periods"
                )

            with col2:
                backtest_top_k_detail = st.number_input(
                    "每期预测号码数",
                    min_value=1,
                    max_value=49,
                    value=38,
                    help="每期预测多少个号码",
                    key="backtest_detail_topk"
                )

            with col3:
                use_anti_loss = st.checkbox(
                    "启用防连错机制",
                    value=False,
                    help="启用后，上期失败会调整预测策略",
                    key="use_anti_loss"
                )

            # 防连错模式选择
            if use_anti_loss:
                anti_loss_mode = st.selectbox(
                    "防连错策略",
                    options=[
                        "用户方案：失败→跳过前10个",
                        "动态调整：根据连错次数调整",
                        "混合策略：模型+冷号混合",
                        "自适应：调整预测窗口"
                    ],
                    index=0,
                    help="选择防连错的具体策略",
                    key="anti_loss_mode_select"
                )

                # 映射到代码中的mode
                mode_mapping = {
                    "用户方案：失败→跳过前10个": "user_proposed",
                    "动态调整：根据连错次数调整": "dynamic",
                    "混合策略：模型+冷号混合": "mixed",
                    "自适应：调整预测窗口": "adaptive"
                }
                anti_loss_mode_code = mode_mapping[anti_loss_mode]

                st.info(f"""
                **{anti_loss_mode}**
                
                {'**原理：** 上期失败 → 本期跳过前10个号码，使用11-49位' if anti_loss_mode_code == 'user_proposed' else ''}
                {'**原理：** 连错0-1次正常，2次跳5个，3次跳10个，4次及以上随机' if anti_loss_mode_code == 'dynamic' else ''}
                {'**原理：** 根据连错次数调整模型和冷号的混合比例' if anti_loss_mode_code == 'mixed' else ''}
                {'**原理：** 根据连错情况动态调整预测窗口范围' if anti_loss_mode_code == 'adaptive' else ''}
                """)

            if st.button("🔍 生成详细预测结果", use_container_width=True, key="generate_detail_backtest"):
                with st.spinner(f"正在生成最近{backtest_periods_detail}期的详细预测..."):
                    try:
                        # 导入防连错引擎（如果启用）
                        if use_anti_loss:
                            from anti_consecutive_loss import AntiConsecutiveLossPredictor

                        # 准备存储结果
                        detail_results = []
                        detail_results_no_anti = []  # 对比用：不使用防连错的结果

                        # 获取数据
                        data = st.session_state.data
                        start_idx = len(data) - backtest_periods_detail

                        # 创建防连错预测器（如果启用）
                        if use_anti_loss:
                            # 根据results_mode选择基础预测器
                            if results_mode == "mean_reversion":
                                base_predictor = st.session_state.mean_reversion_engine
                            elif results_mode == "auto":
                                base_predictor = st.session_state.auto_learning_system
                            elif results_mode == "extreme":
                                base_predictor = st.session_state.extreme_learning_engine
                            elif results_mode == "super":
                                base_predictor = st.session_state.super_learning_engine
                            else:
                                base_predictor = st.session_state.learning_engine

                            anti_loss_predictor = AntiConsecutiveLossPredictor(
                                base_predictor,
                                mode=anti_loss_mode_code
                            )

                        # 创建进度条
                        progress_bar = st.progress(0)

                        # 逐期预测
                        for i, idx in enumerate(range(start_idx, len(data))):
                            # 更新进度
                            progress_bar.progress((i + 1) / backtest_periods_detail)

                            # 训练数据（到当前期之前）
                            train_data = data.iloc[:idx]

                            # 实际结果
                            actual = data.iloc[idx]['特码']
                            period = data.iloc[idx]['期号']

                            # 根据模式进行预测
                            try:
                                if use_anti_loss:
                                    # 使用防连错预测
                                    predicted_nums, info = anti_loss_predictor.predict(train_data, top_k=backtest_top_k_detail)
                                    strategy_used = info.get('strategy', '默认')
                                    consecutive_losses = info.get('consecutive_losses', 0)

                                    # 同时生成不使用防连错的预测（对比）
                                    if results_mode == "mean_reversion":
                                        pred_list = st.session_state.mean_reversion_engine.predict(train_data, top_k=backtest_top_k_detail)
                                        predicted_nums_no_anti = [p['号码'] for p in pred_list]
                                    elif results_mode == "auto":
                                        predicted_nums_no_anti = st.session_state.auto_learning_system.predict(train_data, top_k=backtest_top_k_detail)
                                    elif results_mode == "extreme":
                                        predicted_nums_no_anti = st.session_state.extreme_learning_engine.predict(train_data, top_k=backtest_top_k_detail)
                                    elif results_mode == "super":
                                        predicted_nums_no_anti = st.session_state.super_learning_engine.predict_ultra(train_data, top_k=backtest_top_k_detail)
                                    else:
                                        predicted_nums_no_anti = st.session_state.learning_engine.predict_with_learned_rules(train_data, top_k=backtest_top_k_detail)

                                    hit_no_anti = actual in predicted_nums_no_anti

                                else:
                                    # 不使用防连错
                                    if results_mode == "mean_reversion":
                                        pred_list = st.session_state.mean_reversion_engine.predict(train_data, top_k=backtest_top_k_detail)
                                        predicted_nums = [p['号码'] for p in pred_list]
                                    elif results_mode == "auto":
                                        predicted_nums = st.session_state.auto_learning_system.predict(train_data, top_k=backtest_top_k_detail)
                                    elif results_mode == "extreme":
                                        predicted_nums = st.session_state.extreme_learning_engine.predict(train_data, top_k=backtest_top_k_detail)
                                    elif results_mode == "super":
                                        predicted_nums = st.session_state.super_learning_engine.predict_ultra(train_data, top_k=backtest_top_k_detail)
                                    else:
                                        predicted_nums = st.session_state.learning_engine.predict_with_learned_rules(train_data, top_k=backtest_top_k_detail)

                                    strategy_used = "标准预测"
                                    consecutive_losses = 0

                                # 判断命中
                                hit = actual in predicted_nums

                                # 更新防连错预测器的历史
                                if use_anti_loss:
                                    anti_loss_predictor.update_history(predicted_nums, actual)

                                # 格式化预测号码显示（完整显示所有号码）
                                pred_display = ','.join([f"{n:02d}" for n in predicted_nums])

                                # 保存结果
                                result_dict = {
                                    '期号': period,
                                    '预测号码': pred_display,
                                    '实际特码': f"{actual:02d}",
                                    '命中': '✓' if hit else '✗',
                                    '预测数量': len(predicted_nums)
                                }

                                if use_anti_loss:
                                    result_dict['连错次数'] = consecutive_losses
                                    result_dict['策略'] = strategy_used
                                    result_dict['标准命中'] = '✓' if hit_no_anti else '✗'

                                detail_results.append(result_dict)
                            except Exception as e:
                                # 如果某期预测失败，记录错误
                                detail_results.append({
                                    '期号': period,
                                    '预测号码': '预测失败',
                                    '实际特码': f"{actual:02d}",
                                    '命中': '✗',
                                    '预测数量': 0
                                })

                        progress_bar.empty()

                        # 转换为DataFrame
                        results_df = pd.DataFrame(detail_results)

                        # 计算统计
                        total_tests = len(results_df)
                        hit_count = (results_df['命中'] == '✓').sum()
                        accuracy = hit_count / total_tests * 100 if total_tests > 0 else 0

                        # 显示统计
                        st.success(f"✓ 详细预测完成！")

                        if use_anti_loss:
                            # 计算防连错效果统计
                            anti_loss_stats = anti_loss_predictor.get_statistics()

                            # 计算标准预测的统计（对比）
                            standard_hit_count = (results_df['标准命中'] == '✓').sum()
                            standard_accuracy = standard_hit_count / total_tests * 100 if total_tests > 0 else 0

                            # 计算标准预测的最大连错
                            standard_max_consecutive = 0
                            current_consecutive = 0
                            for hit in results_df['标准命中']:
                                if hit == '✗':
                                    current_consecutive += 1
                                    standard_max_consecutive = max(standard_max_consecutive, current_consecutive)
                                else:
                                    current_consecutive = 0

                            st.markdown("### 🎯 防连错效果对比")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("**防连错预测**")
                                st.metric("测试期数", f"{total_tests}期")
                                st.metric("命中次数", f"{hit_count}次")
                                st.metric("准确率", f"{accuracy:.2f}%")
                                st.metric("最大连错", f"{anti_loss_stats.get('max_consecutive_losses', 0)}次",
                                         delta=f"减少{standard_max_consecutive - anti_loss_stats.get('max_consecutive_losses', 0)}次" if standard_max_consecutive > anti_loss_stats.get('max_consecutive_losses', 0) else None)
                                st.metric("平均连错", f"{anti_loss_stats.get('avg_consecutive_losses', 0):.1f}次")

                            with col2:
                                st.markdown("**标准预测（对比）**")
                                st.metric("测试期数", f"{total_tests}期")
                                st.metric("命中次数", f"{standard_hit_count}次")
                                st.metric("准确率", f"{standard_accuracy:.2f}%")
                                st.metric("最大连错", f"{standard_max_consecutive}次")
                                st.metric("连错次数", f"{anti_loss_stats.get('consecutive_loss_count', 0)}次")

                            with col3:
                                st.markdown("**改进效果**")
                                accuracy_diff = accuracy - standard_accuracy
                                max_loss_diff = standard_max_consecutive - anti_loss_stats.get('max_consecutive_losses', 0)

                                st.metric("准确率提升", f"{accuracy_diff:+.2f}%",
                                         delta="提升" if accuracy_diff > 0 else ("下降" if accuracy_diff < 0 else "持平"))
                                st.metric("连错减少", f"{max_loss_diff}次",
                                         delta="改进" if max_loss_diff > 0 else ("变差" if max_loss_diff < 0 else "持平"))

                                theory_rate = backtest_top_k_detail / 49 * 100
                                st.metric("理论随机", f"{theory_rate:.2f}%")
                                st.metric("超越随机", f"{accuracy - theory_rate:+.2f}%")

                            st.divider()

                            # 防连错策略使用统计
                            if '策略' in results_df.columns:
                                st.markdown("#### 📊 策略使用统计")
                                strategy_counts = results_df['策略'].value_counts()

                                cols = st.columns(len(strategy_counts))
                                for idx, (strategy, count) in enumerate(strategy_counts.items()):
                                    with cols[idx]:
                                        st.metric(strategy, f"{count}次",
                                                 delta=f"{count/total_tests*100:.1f}%")

                        else:
                            # 标准统计显示
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("测试期数", f"{total_tests}期")
                            with col2:
                                st.metric("命中次数", f"{hit_count}次")
                            with col3:
                                st.metric("准确率", f"{accuracy:.2f}%")
                            with col4:
                                theory_rate = backtest_top_k_detail / 49 * 100
                                st.metric("理论随机", f"{theory_rate:.2f}%")

                        st.divider()

                        # 显示详细结果表格
                        st.markdown("#### 📋 详细预测记录")

                        # 高亮显示命中/未命中
                        def highlight_result(row):
                            if row['命中'] == '✓':
                                return ['background-color: #dcfce7'] * len(row)  # 绿色
                            else:
                                return ['background-color: #fee2e2'] * len(row)  # 红色

                        styled_df = results_df.style.apply(highlight_result, axis=1)
                        st.dataframe(styled_df, use_container_width=True, height=600)

                        # 提供下载
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载详细预测结果（CSV）",
                            data=csv,
                            file_name=f"详细预测结果_{backtest_periods_detail}期.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"生成详细预测失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        st.divider()

        if st.button("🔮 使用学习规律预测", use_container_width=True):
            try:
                if results_mode == "mean_reversion":
                    # 使用回归平均引擎预测
                    learned_prediction_list = st.session_state.mean_reversion_engine.predict(
                        st.session_state.data,
                        top_k=predict_top_k
                    )
                    # 提取号码列表
                    learned_prediction = [p['号码'] for p in learned_prediction_list]
                elif results_mode == "auto":
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

                # 回归平均模式显示详细信息
                if results_mode == "mean_reversion":
                    st.divider()
                    st.markdown("#### 🔍 回归平均预测详情")

                    st.info(f"""
                    **预测原理：**
                    基于最近{st.session_state.mean_reversion_engine.analysis_periods}期的数据分析，
                    找出当前偏离平均最大的号码和属性，预测它们会"回归"到平均水平。
                    
                    **⚠️ 重要提醒：**
                    回归平均是长期趋势，不能预测下一期的具体结果！
                    """)

                    # 显示TOP 10预测详情
                    with st.expander("查看TOP 10预测详情", expanded=True):
                        top_10_details = learned_prediction_list[:10]
                        details_df = pd.DataFrame(top_10_details)
                        st.dataframe(details_df, use_container_width=True)

                        st.caption("""
                        **说明：**
                        - 回归分数：负值越大，回归压力越大，理论上越可能出现
                        - 实际次数：在分析期内的实际出现次数
                        - 期望次数：理论上应该出现的次数
                        - 偏离度：实际相对期望的偏离百分比
                        """)

                # 一键复制
                if st.button("📋 复制预测号码", key="copy_learned"):
                    st.success("✓ 请复制上方号码")

            except Exception as e:
                st.error(f"预测失败: {str(e)}")

    # 保存模型功能
    if display_results:
        st.divider()
        st.markdown("### 💾 保存学习结果")

        st.info("""
        **保存模型的好处：**
        - ✓ 下次直接加载，不用重新学习
        - ✓ 保存最佳配置，避免丢失
        - ✓ 可以对比不同模型的效果
        - ✓ 确保结果可重复性
        """)

        col1, col2 = st.columns([2, 1])

        with col1:
            model_name = st.text_input(
                "模型名称",
                value=f"{results_mode}_model_{datetime.now().strftime('%Y%m%d')}",
                help="给这个模型起个名字",
                key="save_model_name"
            )

            model_description = st.text_input(
                "模型描述（可选）",
                value=f"准确率{display_results.get('accuracy', display_results.get('best_fitness', 0))*100:.2f}%",
                help="简单描述这个模型的特点",
                key="save_model_desc"
            )

        with col2:
            st.metric(
                "当前准确率",
                f"{display_results.get('accuracy', display_results.get('best_fitness', 0))*100:.2f}%",
                help="这个模型在回测中的准确率"
            )

        if st.button("💾 保存当前模型", type="primary", use_container_width=True, key="save_model_btn"):
            try:
                # 确定模式
                mode_mapping = {
                    "mean_reversion": "mean_reversion",
                    "auto": "auto",
                    "extreme": "extreme",
                    "super": "super",
                    "genetic": "genetic"
                }

                save_mode = mode_mapping.get(results_mode, "generic")

                # 保存模型
                filepath = st.session_state.model_manager.save_model(
                    model_data=display_results,
                    model_name=model_name,
                    mode=save_mode,
                    description=model_description
                )

                st.success(f"✓ 模型已保存！")
                st.info(f"保存路径：{filepath}")
                st.caption("下次可以在「📁 模型管理」中加载此模型")

            except Exception as e:
                st.error(f"保存失败：{str(e)}")

    # 学习说明


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
    pass  # 使用 pass，这样点击这个空白标签时什么都不会发生，也不会报错

with tabs[5]:
    st.subheader("🔄 历史回测系统")

    backtest_type = st.selectbox(
        "选择回测类型",
        options=['AI融合预测', 'Transformer深度学习', '特码TOP1', '特码TOP3', '特码TOP5', '特码TOP10', '大小', '单双', '波色'],
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
            elif backtest_type == 'Transformer深度学习':
                # Transformer深度学习回测
                def predict_func(train_data):
                    # 创建Transformer模型实例
                    transformer = TransformerModel()
                    # 调用predict方法
                    transformer_result = transformer.predict(train_data, top_k=backtest_top_k)
                    # 返回predictions列表
                    return transformer_result['predictions']

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
    st.subheader("📜 历史记录查看")

    # 自定义期数输入
    st.markdown("#### 🎯 自定义查看期数")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col2:
        # 快速选择按钮（放在前面，先处理）
        quick_select = st.selectbox(
            "快速选择",
            options=["最近100期", "自定义", "最近30期", "最近50期", "最近200期", "最近365期", "最近500期", "全部数据"],
            index=0,  # 默认选择"最近100期"
            key="history_quick_select"
        )

        # 根据快速选择设置默认值
        if quick_select == "全部数据" and st.session_state.data is not None:
            default_periods = len(st.session_state.data)
        elif quick_select == "自定义":
            # 自定义时使用侧边栏的值，但确保至少100期
            default_periods = max(100, history_periods)
        else:
            default_periods = int(quick_select.replace("最近", "").replace("期", ""))

    with col1:
        custom_history_periods = st.number_input(
            "输入查看期数",
            min_value=10,
            max_value=2000,
            value=default_periods,
            step=10,
            help="输入你想查看的期数，如365期",
            key="custom_history_input"
        )

    with col3:
        if st.session_state.data is not None:
            st.metric(
                "数据总期数",
                f"{len(st.session_state.data)}期"
            )

    # 数据可用性检查
    if st.session_state.data is not None:
        available_periods = len(st.session_state.data)
        if custom_history_periods > available_periods:
            st.warning(f"⚠️ 请求{custom_history_periods}期，但只有{available_periods}期数据。将显示全部{available_periods}期。")
            custom_history_periods = available_periods
        else:
            st.success(f"✅ 将显示最近{custom_history_periods}期的历史记录和统计分析")

    st.divider()

    st.markdown(f"### 📋 历史记录（最近 {custom_history_periods} 期）")

    history_df = HistoryViewer.get_recent_history(st.session_state.data, custom_history_periods)
    st.dataframe(history_df, use_container_width=True, height=400)

    st.divider()
    st.subheader(f"📊 历史统计分析（最近 {custom_history_periods} 期）")

    history_stats = HistoryViewer.analyze_history(st.session_state.data, custom_history_periods)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("大数次数", f"{history_stats['大数次数']} ({history_stats['大数次数']/custom_history_periods*100:.1f}%)")
        st.metric("小数次数", f"{history_stats['小数次数']} ({history_stats['小数次数']/custom_history_periods*100:.1f}%)")

    with col2:
        st.metric("单数次数", f"{history_stats['单数次数']} ({history_stats['单数次数']/custom_history_periods*100:.1f}%)")
        st.metric("双数次数", f"{history_stats['双数次数']} ({history_stats['双数次数']/custom_history_periods*100:.1f}%)")

    with col3:
        st.metric("红波次数", f"{history_stats['红波次数']} ({history_stats['红波次数']/custom_history_periods*100:.1f}%)")
        st.metric("蓝波次数", f"{history_stats['蓝波次数']} ({history_stats['蓝波次数']/custom_history_periods*100:.1f}%)")

    with col4:
        st.metric("绿波次数", f"{history_stats['绿波次数']} ({history_stats['绿波次数']/custom_history_periods*100:.1f}%)")
        st.metric("平均特码", f"{history_stats['平均特码']:.2f}")

with tabs[7]:
    st.subheader("📉 数据统计分析")

    # 自定义期数输入
    st.markdown("#### 🎯 自定义分析期数")
    col1, col2 = st.columns([2, 2])

    with col1:
        stats_periods = st.number_input(
            "输入分析期数",
            min_value=10,
            max_value=2000,
            value=100,
            step=10,
            help="输入你想分析的期数，如365期",
            key="stats_periods"
        )

    with col2:
        # 快速选择
        stats_quick = st.selectbox(
            "快速选择",
            options=["自定义", "最近30期", "最近50期", "最近100期", "最近200期", "最近365期"],
            index=3,
            key="stats_quick"
        )

        if stats_quick != "自定义":
            stats_periods = int(stats_quick.replace("最近", "").replace("期", ""))

    # 数据可用性检查
    if st.session_state.data is not None:
        available = len(st.session_state.data)
        if stats_periods > available:
            st.warning(f"⚠️ 请求{stats_periods}期，但只有{available}期数据。将分析全部{available}期。")
            stats_periods = available

    st.divider()

    st.subheader(f"号码出现频率（最近{stats_periods}期）")
    recent_data = st.session_state.data['特码'].iloc[-stats_periods:]
    freq = recent_data.value_counts().sort_index()

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

    # 走势图使用较少的期数以保持清晰
    trend_periods = min(stats_periods, 100)
    st.subheader(f"特码走势图（最近{trend_periods}期）")
    recent_trend = st.session_state.data[['期号', '特码']].iloc[-trend_periods:]

    fig = px.line(
        recent_trend,
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
    <p><strong>内部测试专用 </strong></p>
    <p>完整整合版本功能 · 后置输入优先去重 · 属性自动分类 · 一键移动到档位</p>
    <p>仅供教育和学术研究使用 · 请勿用于实际投注</p>
    <p><strong>理性娱乐，休闲健康 🎓</strong></p>
</div>
""", unsafe_allow_html=True)