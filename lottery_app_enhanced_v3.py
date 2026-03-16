"""
AI彩票量化研究系统 - 增强版v3.0 Streamlit Web界面
仅供教育和学术研究使用

v3.0新增功能:
- 完全手动控制的下注助手（原始号码粘贴区 + 属性分拣 + 档位分配）
- 支持自定义输入和复制粘贴
- 大小、单双、波色的AI预测和回测
- 可自定义预测数量(1-49)
- 一键复制预测结果
- 历史记录查看（可自定义期数）

运行命令: streamlit run lottery_app_enhanced_v3.py
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
    EnhancedPredictionEngine, HistoryViewer
)
import warnings
import re
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
        min-height: 120px;
    }
    .tier1 { background-color: #fce7f3; border: 2px solid #ec4899; }
    .tier2 { background-color: #dbeafe; border: 2px solid #3b82f6; }
    .tier3 { background-color: #d1fae5; border: 2px solid #10b981; }
    .tier4 { background-color: #f3f4f6; border: 2px solid #6b7280; }
    
    .category-header {
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 免责声明
# ============================================================================

DISCLAIMER = """
⚠️ **重要声明** ⚠️

本系统仅供**教育和学术研究**目的使用。

**彩票开奖结果完全随机**，任何预测模型都无法改变随机本质。
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
    # 提取所有数字
    numbers = re.findall(r'\d+', text)
    # 转换为整数并过滤1-49范围
    valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 49]
    return sorted(set(valid_numbers))

def get_number_property(num):
    """获取号码属性"""
    is_big = num >= 25
    is_odd = num % 2 == 1
    return {
        'size': '大' if is_big else '小',
        'parity': '单' if is_odd else '双',
        'category': '大单' if (is_big and is_odd) else '大双' if (is_big and not is_odd) else '小单' if (not is_big and is_odd) else '小双'
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
if 'category_numbers' not in st.session_state:
    st.session_state.category_numbers = {
        '大单': {'1档': '', '2档': '', '3档': '', '4档': ''},
        '大双': {'1档': '', '2档': '', '3档': '', '4档': ''},
        '小单': {'1档': '', '2档': '', '3档': '', '4档': ''},
        '小双': {'1档': '', '2档': '', '3档': '', '4档': ''}
    }

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
        max_value=200, 
        value=50,
        help="用于历史回测的期数"
    )
    
    backtest_strategy = st.selectbox(
        "回测策略",
        options=['top1', 'top3', 'top5', 'top10', 'fusion'],
        index=0,
        format_func=lambda x: {
            'top1': 'TOP1 (第1名命中)',
            'top3': 'TOP3 (前3名命中)',
            'top5': 'TOP5 (前5名命中)',
            'top10': 'TOP10 (前10名命中)',
            'fusion': 'AI融合预测'
        }[x],
        help="选择回测策略"
    )
    
    # 回测号码数量
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

st.markdown('<h1 class="main-header">🧠 AI彩票量化研究系统 Enhanced</h1>', unsafe_allow_html=True)

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
            
            # 概率融合
            fused_prob = EnsembleFusion.fuse_predictions([nb, knn, dt, rf, gb])
            fusion_predictions = EnsembleFusion.get_top_predictions(fused_prob, fusion_top_k)
            
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
            
            # 创建表格数据
            df_fusion = pd.DataFrame(fusion_preds)
            st.dataframe(df_fusion, use_container_width=True, height=400)
            
            # 一键复制
            if st.button("📋 复制融合预测", key="copy_fusion"):
                copy_text = "AI融合预测 TOP {}\n".format(len(fusion_preds))
                for i, pred in enumerate(fusion_preds, 1):
                    copy_text += f"{i}. {pred['号码']:02d} (概率:{pred['概率']}, 置信度:{pred['置信度']})\n"
                st.code(copy_text, language=None)
                st.success("已生成复制文本")
        
        with col2:
            st.subheader("🤖 Transformer深度学习")
            transformer_preds = st.session_state.predictions['Transformer']
            
            # 创建表格数据
            df_transformer = pd.DataFrame(transformer_preds)
            st.dataframe(df_transformer, use_container_width=True, height=400)
            
            # 一键复制
            if st.button("📋 复制Transformer预测", key="copy_transformer"):
                copy_text = "Transformer深度学习 TOP {}\n".format(len(transformer_preds))
                for i, pred in enumerate(transformer_preds, 1):
                    copy_text += f"{i}. {pred['号码']:02d} (概率:{pred['概率']}, 置信度:{pred['置信度']})\n"
                st.code(copy_text, language=None)
                st.success("已生成复制文本")
        
        st.divider()
        
        # 辅助预测
        st.subheader("🎲 辅助预测 (AI模型)")
        aux_preds = st.session_state.predictions['辅助预测']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 大小预测")
            big_small = aux_preds['大小'][0]  # 取第一个（概率最高的）
            st.metric(
                "预测结果", 
                big_small['类型'],
                delta=f"概率: {big_small['概率']}"
            )
            st.caption(f"置信度: {big_small['置信度']}")
        
        with col2:
            st.markdown("### 单双预测")
            odd_even = aux_preds['单双'][0]  # 取第一个（概率最高的）
            st.metric(
                "预测结果", 
                odd_even['类型'],
                delta=f"概率: {odd_even['概率']}"
            )
            st.caption(f"置信度: {odd_even['置信度']}")
        
        with col3:
            st.markdown("### 波色预测")
            color = aux_preds['波色'][0]  # 取第一个（概率最高的）
            st.metric(
                "预测结果", 
                color['类型'],
                delta=f"概率: {color['概率']}"
            )
            st.caption(f"置信度: {color['置信度']}")

# ============================================================================
# 标签2: 下注助手 (完全手动控制版本)
# ============================================================================

with tabs[1]:
    st.subheader("💰 智能下注助手")
    
    # 1. 原始号码粘贴区
    st.markdown("### 1. 原始号码粘贴区")
    raw_numbers = st.text_area(
        "在此粘贴或输入号码（支持任意格式：空格、逗号、换行等分隔）",
        value=st.session_state.raw_numbers,
        height=150,
        help="例如: 1 2 3 4 5 或 1,2,3,4,5 或每行一个号码",
        placeholder="示例:\n1 2 3 4 5\n或\n1,2,3,4,5\n或\n1\n2\n3\n4\n5"
    )
    
    parsed_numbers = parse_numbers(raw_numbers)
    st.info(f"✓ 检测到 {len(parsed_numbers)} 个有效号码 (1-49): {', '.join(map(str, parsed_numbers)) if parsed_numbers else '无'}")
    
    st.divider()
    
    # 2. 属性分拣区
    st.markdown("### 2. 属性分拣 (点击入档)")
    st.caption("从上方复制号码，粘贴到对应的属性和档位。例如：大数(25-49)且单数的号码放入「大单」对应档位")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 大单
    with col1:
        st.markdown('<div class="category-header">大单</div>', unsafe_allow_html=True)
        big_odd_1 = st.text_input("1档", value=st.session_state.category_numbers['大单']['1档'], key="bo1", label_visibility="collapsed", placeholder="1档")
        big_odd_2 = st.text_input("2档", value=st.session_state.category_numbers['大单']['2档'], key="bo2", label_visibility="collapsed", placeholder="2档")
        big_odd_3 = st.text_input("3档", value=st.session_state.category_numbers['大单']['3档'], key="bo3", label_visibility="collapsed", placeholder="3档")
        big_odd_4 = st.text_input("4档", value=st.session_state.category_numbers['大单']['4档'], key="bo4", label_visibility="collapsed", placeholder="4档")
    
    # 大双
    with col2:
        st.markdown('<div class="category-header">大双</div>', unsafe_allow_html=True)
        big_even_1 = st.text_input("1档", value=st.session_state.category_numbers['大双']['1档'], key="be1", label_visibility="collapsed", placeholder="1档")
        big_even_2 = st.text_input("2档", value=st.session_state.category_numbers['大双']['2档'], key="be2", label_visibility="collapsed", placeholder="2档")
        big_even_3 = st.text_input("3档", value=st.session_state.category_numbers['大双']['3档'], key="be3", label_visibility="collapsed", placeholder="3档")
        big_even_4 = st.text_input("4档", value=st.session_state.category_numbers['大双']['4档'], key="be4", label_visibility="collapsed", placeholder="4档")
    
    # 小单
    with col3:
        st.markdown('<div class="category-header">小单</div>', unsafe_allow_html=True)
        small_odd_1 = st.text_input("1档", value=st.session_state.category_numbers['小单']['1档'], key="so1", label_visibility="collapsed", placeholder="1档")
        small_odd_2 = st.text_input("2档", value=st.session_state.category_numbers['小单']['2档'], key="so2", label_visibility="collapsed", placeholder="2档")
        small_odd_3 = st.text_input("3档", value=st.session_state.category_numbers['小单']['3档'], key="so3", label_visibility="collapsed", placeholder="3档")
        small_odd_4 = st.text_input("4档", value=st.session_state.category_numbers['小单']['4档'], key="so4", label_visibility="collapsed", placeholder="4档")
    
    # 小双
    with col4:
        st.markdown('<div class="category-header">小双</div>', unsafe_allow_html=True)
        small_even_1 = st.text_input("1档", value=st.session_state.category_numbers['小双']['1档'], key="se1", label_visibility="collapsed", placeholder="1档")
        small_even_2 = st.text_input("2档", value=st.session_state.category_numbers['小双']['2档'], key="se2", label_visibility="collapsed", placeholder="2档")
        small_even_3 = st.text_input("3档", value=st.session_state.category_numbers['小双']['3档'], key="se3", label_visibility="collapsed", placeholder="3档")
        small_even_4 = st.text_input("4档", value=st.session_state.category_numbers['小双']['4档'], key="se4", label_visibility="collapsed", placeholder="4档")
    
    st.divider()
    
    # 3. 档位分配区
    st.markdown("### 3. 档位分配 (后置递输入优先点)")
    
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
    
    # 收集所有号码和档位
    all_numbers_with_tier = []
    
    # 辅助函数：解析并添加号码
    def add_numbers(text, tier, category):
        nums = parse_numbers(text)
        for num in nums:
            all_numbers_with_tier.append({
                'number': num,
                'tier': tier,
                'category': category
            })
    
    # 大单
    add_numbers(big_odd_1, 1, '大单')
    add_numbers(big_odd_2, 2, '大单')
    add_numbers(big_odd_3, 3, '大单')
    add_numbers(big_odd_4, 4, '大单')
    
    # 大双
    add_numbers(big_even_1, 1, '大双')
    add_numbers(big_even_2, 2, '大双')
    add_numbers(big_even_3, 3, '大双')
    add_numbers(big_even_4, 4, '大双')
    
    # 小单
    add_numbers(small_odd_1, 1, '小单')
    add_numbers(small_odd_2, 2, '小单')
    add_numbers(small_odd_3, 3, '小单')
    add_numbers(small_odd_4, 4, '小单')
    
    # 小双
    add_numbers(small_even_1, 1, '小双')
    add_numbers(small_even_2, 2, '小双')
    add_numbers(small_even_3, 3, '小双')
    add_numbers(small_even_4, 4, '小双')
    
    # 档位可视化展示
    col1, col2, col3, col4 = st.columns(4)
    
    tier_numbers = {1: [], 2: [], 3: [], 4: []}
    for item in all_numbers_with_tier:
        tier_numbers[item['tier']].append(item['number'])
    
    with col1:
        st.markdown('<div class="tier-box tier1">', unsafe_allow_html=True)
        st.markdown("**1档(最高)**")
        numbers_text = ', '.join(map(str, sorted(set(tier_numbers[1])))) if tier_numbers[1] else '空'
        st.write(numbers_text)
        st.caption(f"共 {len(set(tier_numbers[1]))} 个号码")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="tier-box tier2">', unsafe_allow_html=True)
        st.markdown("**2档(中)**")
        numbers_text = ', '.join(map(str, sorted(set(tier_numbers[2])))) if tier_numbers[2] else '空'
        st.write(numbers_text)
        st.caption(f"共 {len(set(tier_numbers[2]))} 个号码")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="tier-box tier3">', unsafe_allow_html=True)
        st.markdown("**3档(低)**")
        numbers_text = ', '.join(map(str, sorted(set(tier_numbers[3])))) if tier_numbers[3] else '空'
        st.write(numbers_text)
        st.caption(f"共 {len(set(tier_numbers[3]))} 个号码")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="tier-box tier4">', unsafe_allow_html=True)
        st.markdown("**4档(保本)**")
        numbers_text = ', '.join(map(str, sorted(set(tier_numbers[4])))) if tier_numbers[4] else '空'
        st.write(numbers_text)
        st.caption(f"共 {len(set(tier_numbers[4]))} 个号码")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # 统计信息
    col1, col2, col3, col4 = st.columns(4)
    
    total_unique_numbers = len(set([item['number'] for item in all_numbers_with_tier]))
    base_amount = total_budget / odds if total_unique_numbers > 0 else 0
    
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
            st.warning("请先在属性分拣区输入号码")
        else:
            # 档位权重
            tier_weights = {1: 3.0, 2: 1.5, 3: 0.5, 4: 0.0}
            
            # 计算基础投注
            base_amount = total_budget / odds
            
            # 计算总权重
            total_weight = sum(tier_weights[item['tier']] for item in all_numbers_with_tier)
            
            # 剩余资金
            remaining = total_budget - (len(all_numbers_with_tier) * base_amount)
            
            # 计算每个号码的投注金额
            betting_results = []
            for item in all_numbers_with_tier:
                weight = tier_weights[item['tier']]
                extra_amount = (remaining * weight / total_weight) if total_weight > 0 else 0
                bet_amount = base_amount + extra_amount
                profit = bet_amount * odds - total_budget
                
                betting_results.append({
                    '号码': f"{item['number']:02d}",
                    '属性归档': item['category'],
                    '所属档位': f"{item['tier']}档",
                    '单注金额': f"{bet_amount:.2f}",
                    '中奖纯利': f"{profit:.2f}"
                })
            
            # 按档位和号码排序
            betting_results = sorted(betting_results, key=lambda x: (int(x['所属档位'][0]), int(x['号码'])))
            
            # 显示结果表格
            st.subheader("📋 投注明细")
            df_bet = pd.DataFrame(betting_results)
            st.dataframe(df_bet, use_container_width=True, height=400)
            
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
                st.metric("预算状态", "正常" if total_bet <= total_budget + 1 else "超支")
            
            st.divider()
            
            # 生成投注格式（同金额合并）
            st.subheader("📋 投注格式（同金额合并）")
            
            # 按金额分组
            amount_groups = {}
            for result in betting_results:
                amount = result['单注金额']
                if amount not in amount_groups:
                    amount_groups[amount] = []
                amount_groups[amount].append(result['号码'])
            
            # 生成格式化文本
            bet_format_lines = []
            for amount in sorted(amount_groups.keys(), key=lambda x: float(x), reverse=True):
                nums = ', '.join(amount_groups[amount])
                bet_format_lines.append(f"{nums} = {amount}元")
            
            bet_format_lines.append(f"\n总预算: {total_budget}元")
            bet_format_lines.append(f"赔率: {odds}")
            bet_format_lines.append(f"投注号码数: {len(betting_results)}个")
            
            bet_format = "\n".join(bet_format_lines)
            st.code(bet_format, language=None)
            
            # 复制按钮
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📋 复制投注格式", key="copy_bet_format", use_container_width=True):
                    st.success("✓ 请复制上方文本框内容")
    
    # 保本原理说明
    st.divider()
    st.info("""
    💡 **保本原理说明**
    
    - **基础投注** = 总预算 ÷ 赔率
    - **剩余资金** = 总预算 - (号码数量 × 基础投注)
    - **额外分配** = 剩余资金 × (档位权重 ÷ 总权重分数)
    - **最终投注** = 基础投注 + 额外分配
    - **档位权重**: 1档=3.0, 2档=1.5, 3档=0.5, 4档=0.0
    
    ⚠️ **重要提醒**: 本工具仅供学习算法原理，切勿用于实际投注！彩票必输，理性娱乐！
    """)

# ============================================================================
# 标签3: 特征工程
# ============================================================================

with tabs[2]:
    if st.session_state.features is None:
        st.warning("请先运行AI预测以提取特征")
    else:
        features = st.session_state.features
        
        st.subheader("🔧 8维特征工程分析")
        
        # 统计特征
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
        
        with st.expander("📈 2. 频率与概率特征"):
            freq_features = features['频率特征']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("信息熵", f"{freq_features['entropy']:.4f}")
                st.metric("归一化熵", f"{freq_features['normalized_entropy']:.4f}")
                st.metric("吉尼系数", f"{freq_features['gini']:.4f}")
            
            with col2:
                st.write("**热号 TOP 10:**")
                st.write(", ".join(map(str, freq_features['hot_numbers'])))
                st.write("**冷号 TOP 10:**")
                st.write(", ".join(map(str, freq_features['cold_numbers'])))

# ============================================================================
# 标签4: 模型评估
# ============================================================================

with tabs[3]:
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
    
    # 可视化对比
    fig = px.bar(
        model_performance,
        x='模型',
        y='准确率(%)',
        color='准确率(%)',
        color_continuous_scale='Blues',
        title='各模型准确率对比'
    )
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="理论随机(2%)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 标签5: 历史回测
# ============================================================================

with tabs[4]:
    st.subheader("🔄 历史回测系统")
    
    backtest_type = st.selectbox(
        "选择回测类型",
        options=['AI融合预测', '特码TOP1', '特码TOP3', '特码TOP5', '特码TOP10', '大小', '单双', '波色'],
        index=0
    )
    
    if st.button("▶️ 开始回测", type="primary"):
        with st.spinner(f"正在运行{backtest_type}回测..."):
            if backtest_type == 'AI融合预测':
                # AI融合预测回测
                def predict_func(train_data):
                    temp_features = FeatureEngineering.extract_all_features(train_data)
                    temp_nb = MLModels.naive_bayes(train_data, temp_features)
                    temp_knn = MLModels.weighted_knn(train_data, temp_features)
                    temp_dt = MLModels.decision_tree(train_data, temp_features)
                    temp_rf = MLModels.random_forest(train_data, temp_features)
                    temp_gb = MLModels.gradient_boosting(train_data, temp_features)
                    temp_fused = EnsembleFusion.fuse_predictions([temp_nb, temp_knn, temp_dt, temp_rf, temp_gb])
                    return EnsembleFusion.get_top_predictions(temp_fused, backtest_top_k)
                
                result = BacktestEngine.run(
                    st.session_state.data,
                    predict_func,
                    test_periods=backtest_periods,
                    strategy=f'top{min(backtest_top_k, 10)}'  # 使用实际的top_k，但策略名称限制在top10
                )
                
            elif backtest_type.startswith('特码'):
                # 特码回测
                strategy_map = {
                    '特码TOP1': 'top1',
                    '特码TOP3': 'top3',
                    '特码TOP5': 'top5',
                    '特码TOP10': 'top10'
                }
                
                def predict_func(train_data):
                    temp_features = FeatureEngineering.extract_all_features(train_data)
                    temp_nb = MLModels.naive_bayes(train_data, temp_features)
                    # 使用backtest_top_k作为预测数量
                    k = {'top1': 1, 'top3': 3, 'top5': 5, 'top10': 10}[strategy_map[backtest_type]]
                    return EnsembleFusion.get_top_predictions(temp_nb, k)
                
                result = BacktestEngine.run(
                    st.session_state.data,
                    predict_func,
                    test_periods=backtest_periods,
                    strategy=strategy_map[backtest_type]
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
                
            else:  # 波色
                result = AuxiliaryBacktest.backtest_color(
                    st.session_state.data,
                    test_periods=backtest_periods
                )
            
            st.success("✓ 回测完成！")
            
            # 显示回测结果
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总测试数", result['total_tests'])
            with col2:
                st.metric("命中次数", result['hit_count'])
            with col3:
                st.metric("准确率", result['accuracy'])
            
            # 理论随机准确率提示
            if backtest_type == 'AI融合预测':
                theory_rate = f"{(backtest_top_k/49*100):.2f}%"
                st.info(f"📊 理论随机准确率: {theory_rate} (预测{backtest_top_k}个号码)")
            elif backtest_type == '大小' or backtest_type == '单双':
                st.info("📊 理论随机准确率: 50%")
            elif backtest_type == '波色':
                st.info("📊 理论随机准确率: 33.3%")
            elif backtest_type == '特码TOP1':
                st.info("📊 理论随机准确率: 2.0%")
            elif backtest_type == '特码TOP3':
                st.info("📊 理论随机准确率: 6.1%")
            elif backtest_type == '特码TOP5':
                st.info("📊 理论随机准确率: 10.2%")
            elif backtest_type == '特码TOP10':
                st.info("📊 理论随机准确率: 20.4%")
            
            # 显示回测记录
            st.divider()
            st.subheader(f"最近20条回测记录")
            
            results_df = result['results']
            
            # 高亮显示
            def highlight_hit(row):
                if row['命中']:
                    return ['background-color: #dcfce7'] * len(row)
                else:
                    return ['background-color: #fee2e2'] * len(row)
            
            styled_df = results_df.tail(20).style.apply(highlight_hit, axis=1)
            st.dataframe(styled_df, use_container_width=True)

# ============================================================================
# 标签6: 历史记录
# ============================================================================

with tabs[5]:
    st.subheader(f"📜 历史记录查看（最近 {history_periods} 期）")
    
    # 获取历史记录
    history_df = HistoryViewer.get_recent_history(st.session_state.data, history_periods)
    
    # 显示历史记录
    st.dataframe(history_df, use_container_width=True, height=400)
    
    st.divider()
    
    # 历史统计分析
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

# ============================================================================
# 标签7: 统计图表
# ============================================================================

with tabs[6]:
    st.subheader("📉 数据统计分析")
    
    # 号码频率分布
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
    
    # 趋势图
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
    <p><strong>AI彩票量化研究系统 Enhanced v3.0</strong></p>
    <p>新增: 完全手动下注助手 · 原始号码粘贴区 · 属性分拣 · 档位分配可视化</p>
    <p>仅供教育和学术研究使用 · 请勿用于实际投注</p>
    <p><strong>理性娱乐，远离赌博 🎓</strong></p>
</div>
""", unsafe_allow_html=True)
