import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="MBA Dashboard - Apriori Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .business-rec {
        background: #fff4e6;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# HELPER FUNCTIONS WITH CACHING
# ===================================================================

@st.cache_data
def load_and_prepare_data(filepath):
    """Load and prepare data with comprehensive preprocessing"""
    try:
        df = pd.read_csv(filepath)
        
        # Preprocessing
        for col in df.columns:
            if df[col].dtypes == 'float64':
                df[col].fillna(-1, inplace=True)
        
        df['days_since_prior_order'] = df['days_since_prior_order'].astype(np.int64)
        
        # Add temporal features
        if 'order_dow' in df.columns:
            df['is_weekend'] = df['order_dow'].isin([0, 6])
            df['day_name'] = df['order_dow'].map({
                0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 
                3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'
            })
        
        if 'order_hour_of_day' in df.columns:
            def get_time_period(hour):
                if 6 <= hour < 12: return 'Morning'
                elif 12 <= hour < 18: return 'Afternoon'
                elif 18 <= hour < 23: return 'Evening'
                else: return 'Night'
            df['time_period'] = df['order_hour_of_day'].apply(get_time_period)
        
        # Create transactions
        transactions_list = df.groupby(['user_id', 'department'])['product_name'].unique().apply(list).tolist()
        
        # One-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions_list).transform(transactions_list)
        basket_data = pd.DataFrame(te_ary, columns=te.columns_)
        
        product_names = basket_data.columns.tolist()
        
        return df, basket_data, product_names, transactions_list
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def run_algorithm(basket_data, algorithm, min_support):
    """Run Apriori or FP-Growth with timing"""
    start_time = time.time()
    
    if algorithm == "Apriori":
        frequent_itemsets = apriori(basket_data, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(basket_data, min_support=min_support, use_colnames=True)
    
    execution_time = time.time() - start_time
    return frequent_itemsets, execution_time

@st.cache_data
def generate_rules(frequent_itemsets, metric, min_threshold):
    """Generate association rules"""
    try:
        if len(frequent_itemsets) == 0:
            return pd.DataFrame()
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        return rules
    except:
        return pd.DataFrame()

def format_frozenset(fs):
    """Convert frozenset to readable string"""
    return ', '.join(list(fs))

def calculate_business_metrics(rules, df):
    """Calculate business-oriented metrics"""
    metrics = {}
    
    if not rules.empty:
        metrics['avg_confidence'] = rules['confidence'].mean()
        metrics['avg_lift'] = rules['lift'].mean()
        metrics['strong_rules'] = len(rules[rules['lift'] > 2])
        metrics['total_rules'] = len(rules)
        
        # Calculate potential revenue impact (simplified)
        avg_basket_value = df.groupby('order_id')['product_id'].count().mean()
        metrics['avg_basket_size'] = avg_basket_value
        
    return metrics

# ===================================================================
# LOAD DATA
# ===================================================================

st.markdown('<h1 class="main-header">🛒 Market Basket Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis Perbandingan Algoritma Apriori dan FP-Growth untuk Sistem Informasi Ritel</p>', unsafe_allow_html=True)

DATA_FILE = "ECommerce_consumer behaviour.csv"

with st.spinner("📂 Loading data dan preprocessing..."):
    df_original, basket_data, product_names, transactions_list = load_and_prepare_data(DATA_FILE)

if df_original is None:
    st.stop()

# Quick stats
total_transactions = basket_data.shape[0]
total_products = basket_data.shape[1]
total_records = len(df_original)
total_users = df_original['user_id'].nunique()
total_departments = df_original['department'].nunique()

# ===================================================================
# SIDEBAR CONTROLS
# ===================================================================

st.sidebar.markdown("# ⚙️ Control Panel")
st.sidebar.markdown("---")

# Algorithm Selection
st.sidebar.markdown("### 🧮 Algorithm Selection")
algo_col1, algo_col2 = st.sidebar.columns(2)
with algo_col1:
    run_apriori = st.checkbox("**Apriori** (Primary)", value=True, help="Main algorithm for analysis")
with algo_col2:
    run_fpgrowth = st.checkbox("FP-Growth (Compare)", value=False, help="Comparison algorithm")

if not run_apriori and not run_fpgrowth:
    st.sidebar.warning("⚠️ Select at least one algorithm!")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Mining Parameters")

min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.001,
    max_value=0.05,
    value=0.005,
    step=0.001,
    format="%.3f",
    help="Minimum frequency threshold for itemsets"
)

st.sidebar.markdown("### 📏 Rule Parameters")

metric = st.sidebar.selectbox(
    "Evaluation Metric",
    ['lift', 'confidence', 'support'],
    help="Primary metric for filtering rules"
)

if metric == 'lift':
    min_threshold = st.sidebar.slider("Min Lift", 1.0, 10.0, 1.0, 0.1)
elif metric == 'confidence':
    min_threshold = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.2, 0.05)
else:
    min_threshold = st.sidebar.slider("Min Support", 0.001, 0.1, 0.005, 0.001)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Apriori sebagai algoritma utama, FP-Growth sebagai pembanding kinerja")

# ===================================================================
# RUN ALGORITHMS
# ===================================================================

results = {}

if run_apriori:
    with st.spinner("⚙️ Running Apriori Algorithm..."):
        progress_bar = st.progress(0)
        freq_apriori, time_apriori = run_algorithm(basket_data, "Apriori", min_support)
        progress_bar.progress(50)
        rules_apriori = generate_rules(freq_apriori, metric, min_threshold)
        progress_bar.progress(100)
        progress_bar.empty()
        
        results['Apriori'] = {
            'freq': freq_apriori,
            'rules': rules_apriori,
            'time': time_apriori,
            'metrics': calculate_business_metrics(rules_apriori, df_original)
        }

if run_fpgrowth:
    with st.spinner("⚙️ Running FP-Growth Algorithm..."):
        progress_bar = st.progress(0)
        freq_fpgrowth, time_fpgrowth = run_algorithm(basket_data, "FP-Growth", min_support)
        progress_bar.progress(50)
        rules_fpgrowth = generate_rules(freq_fpgrowth, metric, min_threshold)
        progress_bar.progress(100)
        progress_bar.empty()
        
        results['FP-Growth'] = {
            'freq': freq_fpgrowth,
            'rules': rules_fpgrowth,
            'time': time_fpgrowth,
            'metrics': calculate_business_metrics(rules_fpgrowth, df_original)
        }

# ===================================================================
# TABS STRUCTURE
# ===================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Executive Summary",
    "🔬 Apriori Deep Dive",
    "📈 Algorithm Comparison",
    "🛍️ Business Intelligence",
    "🤔 Smart Recommender",
    "⏰ Temporal Analysis",
    "🏬 Department Intelligence",
    "🕸️ Visual Analytics",
    "📑 Report Generator",
])

# ===================================================================
# TAB 1: EXECUTIVE SUMMARY
# ===================================================================

with tab1:
    st.header("📊 Executive Summary Dashboard")
    
    # Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📦 Total Records", f"{total_records:,}")
    col2.metric("🛒 Transactions", f"{total_transactions:,}")
    col3.metric("👥 Unique Users", f"{total_users:,}")
    col4.metric("🏷️ Products", f"{total_products:,}")
    col5.metric("🏪 Departments", f"{total_departments:,}")
    
    st.markdown("---")
    
    # Algorithm Performance
    st.subheader("⚡ Algorithm Performance")
    
    perf_cols = st.columns(len(results))
    for idx, (algo_name, data) in enumerate(results.items()):
        with perf_cols[idx]:
            st.markdown(f"### {'🎯 ' if algo_name == 'Apriori' else '📊 '}{algo_name}")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("⏱️ Time", f"{data['time']:.2f}s")
            metric_col2.metric("📦 Itemsets", f"{len(data['freq']):,}")
            
            metric_col3, metric_col4 = st.columns(2)
            metric_col3.metric("📋 Rules", f"{len(data['rules']):,}")
            if data['metrics']:
                metric_col4.metric("💪 Strong Rules", f"{data['metrics'].get('strong_rules', 0):,}")
    
    st.markdown("---")
    
    # Business Insights
    st.subheader("💼 Key Business Insights")
    
    if 'Apriori' in results and not results['Apriori']['rules'].empty:
        rules = results['Apriori']['rules']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Top Product Associations")
            top_rules = rules.nlargest(5, 'lift').copy()
            top_rules['antecedents'] = top_rules['antecedents'].apply(format_frozenset)
            top_rules['consequents'] = top_rules['consequents'].apply(format_frozenset)
            
            for idx, row in top_rules.iterrows():
                st.markdown(f"""
                <div class="business-rec">
                    <strong>IF</strong> customer buys: <strong>{row['antecedents']}</strong><br>
                    <strong>THEN</strong> they likely buy: <strong>{row['consequents']}</strong><br>
                    <small>Confidence: {row['confidence']:.1%} | Lift: {row['lift']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📈 Business Recommendations")
            
            # Generate actionable recommendations
            strong_associations = rules[rules['lift'] > 3].nlargest(3, 'confidence')
            
            if not strong_associations.empty:
                st.markdown("""
                <div class="insight-box">
                    <strong>🎁 Bundling Opportunities:</strong><br>
                    Create product bundles based on high-lift associations to increase basket size.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="insight-box">
                    <strong>🏪 Store Layout Optimization:</strong><br>
                    Place frequently associated products near each other to encourage impulse buying.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="insight-box">
                    <strong>💰 Promotional Strategy:</strong><br>
                    Offer discounts on antecedents to drive sales of higher-margin consequents.
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Product Popularity
    st.subheader("🏆 Top 10 Most Popular Products")
    
    product_freq = basket_data.sum().sort_values(ascending=False).head(10)
    
    fig = go.Figure(go.Bar(
        x=product_freq.values,
        y=product_freq.index,
        orientation='h',
        marker=dict(
            color=product_freq.values,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Frequency")
        ),
        text=product_freq.values,
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Product Frequency Distribution",
        xaxis_title="Number of Transactions",
        yaxis_title="Product",
        height=450,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 2: APRIORI DEEP DIVE
# ===================================================================

with tab2:
    st.header("🔬 Apriori Algorithm - Deep Analysis")
    
    if 'Apriori' not in results:
        st.warning("⚠️ Please enable Apriori algorithm in the sidebar")
    else:
        apriori_data = results['Apriori']
        rules = apriori_data['rules']
        freq = apriori_data['freq']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱️ Execution Time", f"{apriori_data['time']:.3f} sec")
        col2.metric("📦 Frequent Itemsets", f"{len(freq):,}")
        col3.metric("📋 Association Rules", f"{len(rules):,}")
        
        if apriori_data['metrics']:
            col4.metric("💪 Avg Lift", f"{apriori_data['metrics']['avg_lift']:.2f}")
        
        st.markdown("---")
        
        # Rules Explorer
        st.subheader("🔍 Association Rules Explorer")
        
        if not rules.empty:
            # Filters
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, key='conf_apriori')
            with filter_col2:
                min_lift = st.slider("Min Lift", 1.0, float(rules['lift'].max()), 1.0, 0.5, key='lift_apriori')
            with filter_col3:
                min_supp = st.slider("Min Support", 0.0, float(rules['support'].max()), 0.0, 0.001, key='supp_apriori', format="%.3f")
            with filter_col4:
                search_product = st.text_input("🔎 Search Product", key='search_apriori')
            
            # Apply filters
            filtered_rules = rules[
                (rules['confidence'] >= min_conf) &
                (rules['lift'] >= min_lift) &
                (rules['support'] >= min_supp)
            ].copy()
            
            if search_product:
                filtered_rules = filtered_rules[
                    filtered_rules['antecedents'].astype(str).str.contains(search_product, case=False) |
                    filtered_rules['consequents'].astype(str).str.contains(search_product, case=False)
                ]
            
            st.info(f"📊 Showing **{len(filtered_rules)}** rules (filtered from {len(rules)} total)")
            
            if not filtered_rules.empty:
                # Display options
                sort_by = st.selectbox("Sort by", ['lift', 'confidence', 'support'], key='sort_apriori')
                filtered_rules = filtered_rules.sort_values(by=sort_by, ascending=False)
                
                # Format for display
                display_rules = filtered_rules.head(50).copy()
                display_rules['antecedents'] = display_rules['antecedents'].apply(format_frozenset)
                display_rules['consequents'] = display_rules['consequents'].apply(format_frozenset)
                
                # Display table
                st.dataframe(
                    display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].style.format({
                        'support': '{:.2%}',
                        'confidence': '{:.2%}',
                        'lift': '{:.2f}'
                    }).background_gradient(subset=['lift'], cmap='RdYlGn'),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = filtered_rules.to_csv(index=False)
                st.download_button(
                    label="📥 Download Rules (CSV)",
                    data=csv,
                    file_name=f"apriori_rules_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.markdown("---")
                
                # Visualizations
                st.subheader("📊 Rule Metrics Visualization")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Scatter plot - Convert frozensets to strings first
                    scatter_data = filtered_rules.head(100).copy()
                    scatter_data['antecedents_str'] = scatter_data['antecedents'].apply(format_frozenset)
                    scatter_data['consequents_str'] = scatter_data['consequents'].apply(format_frozenset)
                    
                    fig = px.scatter(
                        scatter_data,
                        x='support',
                        y='confidence',
                        size='lift',
                        color='lift',
                        hover_data=['antecedents_str', 'consequents_str'],
                        title="Support vs Confidence (sized by Lift)",
                        color_continuous_scale='Viridis',
                        labels={'support': 'Support', 'confidence': 'Confidence'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_col2:
                    # Lift distribution
                    fig = px.histogram(
                        filtered_rules,
                        x='lift',
                        nbins=30,
                        title="Lift Distribution",
                        labels={'lift': 'Lift Value', 'count': 'Frequency'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Association Heatmap
                st.markdown("---")
                st.subheader("🔥 Association Heatmap (Top Products)")
                
                # Get top products
                top_n = st.slider("Number of top products", 5, 20, 10, key='heatmap_n')
                top_products = basket_data.sum().nlargest(top_n).index.tolist()
                
                # Create lift matrix
                lift_matrix = pd.DataFrame(index=top_products, columns=top_products, dtype=float)
                
                for _, row in filtered_rules.iterrows():
                    ant = list(row['antecedents'])
                    cons = list(row['consequents'])
                    
                    if len(ant) == 1 and len(cons) == 1:
                        if ant[0] in top_products and cons[0] in top_products:
                            lift_matrix.loc[ant[0], cons[0]] = row['lift']
                
                lift_matrix = lift_matrix.fillna(0).astype(float)
                
                fig = px.imshow(
                    lift_matrix,
                    labels=dict(x="Consequent", y="Antecedent", color="Lift"),
                    title=f"Product Association Heatmap (Top {top_n} Products)",
                    color_continuous_scale='RdYlGn',
                    aspect='auto'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("No rules match the current filters")
        else:
            st.warning("No association rules generated. Try lowering the thresholds.")
        
        # Itemset Analysis
        st.markdown("---")
        st.subheader("📦 Frequent Itemsets Analysis")
        
        if not freq.empty:
            # Itemset size distribution
            freq['itemset_size'] = freq['itemsets'].apply(len)
            size_dist = freq['itemset_size'].value_counts().sort_index()
            
            fig = px.bar(
                x=size_dist.index,
                y=size_dist.values,
                labels={'x': 'Itemset Size', 'y': 'Count'},
                title="Distribution of Itemset Sizes",
                color=size_dist.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 3: ALGORITHM COMPARISON
# ===================================================================

with tab3:
    st.header("📈 Algorithm Performance Comparison")
    
    if len(results) < 2:
        st.info("ℹ️ Enable both Apriori and FP-Growth in the sidebar for comparison")
        
        if len(results) == 1:
            algo_name = list(results.keys())[0]
            st.markdown(f"### Currently running: **{algo_name}**")
            
            data = results[algo_name]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ Execution Time", f"{data['time']:.3f}s")
            col2.metric("📦 Itemsets Found", f"{len(data['freq']):,}")
            col3.metric("📋 Rules Generated", f"{len(data['rules']):,}")
    else:
        # Comprehensive comparison
        st.markdown("### ⚡ Performance Metrics Comparison")
        
        comparison_data = []
        for algo, data in results.items():
            comparison_data.append({
                'Algorithm': algo,
                'Execution Time (s)': data['time'],
                'Frequent Itemsets': len(data['freq']),
                'Association Rules': len(data['rules']),
                'Avg Confidence': data['metrics'].get('avg_confidence', 0),
                'Avg Lift': data['metrics'].get('avg_lift', 0),
                'Strong Rules (Lift>2)': data['metrics'].get('strong_rules', 0)
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Display table
        st.dataframe(
            comp_df.style.highlight_max(axis=0, subset=['Frequent Itemsets', 'Association Rules', 'Strong Rules (Lift>2)'], color='lightgreen')
            .highlight_min(axis=0, subset=['Execution Time (s)'], color='lightgreen')
            .format({
                'Execution Time (s)': '{:.3f}',
                'Avg Confidence': '{:.2%}',
                'Avg Lift': '{:.2f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Visual comparisons
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Execution time
            fig = px.bar(
                comp_df,
                x='Algorithm',
                y='Execution Time (s)',
                title='⏱️ Execution Time Comparison',
                color='Algorithm',
                text='Execution Time (s)'
            )
            fig.update_traces(texttemplate='%{text:.3f}s', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Rules generated
            fig = px.bar(
                comp_df,
                x='Algorithm',
                y='Association Rules',
                title='📋 Association Rules Generated',
                color='Algorithm',
                text='Association Rules'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Performance insights
        st.subheader("💡 Performance Insights")
        
        fastest = comp_df.loc[comp_df['Execution Time (s)'].idxmin(), 'Algorithm']
        slowest = comp_df.loc[comp_df['Execution Time (s)'].idxmax(), 'Algorithm']
        time_diff = comp_df['Execution Time (s)'].max() - comp_df['Execution Time (s)'].min()
        speedup = comp_df['Execution Time (s)'].max() / comp_df['Execution Time (s)'].min()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Fastest Algorithm", fastest)
        col2.metric("🐌 Slowest Algorithm", slowest)
        col3.metric("⏱️ Time Difference", f"{time_diff:.3f}s")
        
        st.info(f"🚀 **{fastest}** is **{speedup:.2f}x** faster than **{slowest}**")
        
        # Validation check
        st.markdown("---")
        st.subheader("✅ Results Validation")
        
        algo_names = list(results.keys())
        freq1 = results[algo_names[0]]['freq']
        freq2 = results[algo_names[1]]['freq']
        
        if len(freq1) == len(freq2):
            st.success(f"✅ Both algorithms found the same number of itemsets: **{len(freq1)}**")
            
            # Check if identical
            freq1_sorted = freq1.sort_values('support').reset_index(drop=True)
            freq2_sorted = freq2.sort_values('support').reset_index(drop=True)
            
            freq1_sorted['itemsets'] = freq1_sorted['itemsets'].apply(lambda x: tuple(sorted(x)))
            freq2_sorted['itemsets'] = freq2_sorted['itemsets'].apply(lambda x: tuple(sorted(x)))
            
            if freq1_sorted['itemsets'].equals(freq2_sorted['itemsets']):
                st.success("✅ **Itemsets are IDENTICAL** between both algorithms!")
            else:
                st.warning("⚠️ Itemsets differ slightly")
        else:
            st.warning(f"⚠️ Different itemset counts: {algo_names[0]}: {len(freq1)}, {algo_names[1]}: {len(freq2)}")
        
        # Support distribution comparison
        st.markdown("---")
        st.subheader("📊 Support Distribution Comparison")
        
        fig = go.Figure()
        for algo, data in results.items():
            if not data['freq'].empty:
                fig.add_trace(go.Histogram(
                    x=data['freq']['support'],
                    name=algo,
                    opacity=0.7,
                    nbinsx=30
                ))
        
        fig.update_layout(
            title='Support Distribution of Frequent Itemsets',
            xaxis_title='Support',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 4: BUSINESS INTELLIGENCE
# ===================================================================

with tab4:
    st.header("🛍️ Business Intelligence & Actionable Insights")
    
    if 'Apriori' not in results:
        st.warning("⚠️ Enable Apriori algorithm to view business insights")
    else:
        rules = results['Apriori']['rules']
        
        if rules.empty:
            st.warning("No rules available. Try lowering the thresholds.")
        else:
            # Business subsections
            bi_tab1, bi_tab2, bi_tab3, bi_tab4 = st.tabs([
                "💰 Cross-Selling Strategy",
                "🏪 Product Placement",
                "📦 Inventory Intelligence",
                "📊 Performance Analytics"
            ])
            
            # ===== CROSS-SELLING STRATEGY =====
            with bi_tab1:
                st.subheader("💰 Cross-Selling & Up-Selling Opportunities")
                
                # High-confidence rules for cross-selling
                cross_sell_rules = rules[
                    (rules['confidence'] >= 0.3) & 
                    (rules['lift'] > 1.5)
                ].sort_values('lift', ascending=False).head(20)
                
                if not cross_sell_rules.empty:
                    st.markdown("#### 🎯 Top Cross-Selling Recommendations")
                    
                    for idx, row in cross_sell_rules.head(10).iterrows():
                        ant = format_frozenset(row['antecedents'])
                        cons = format_frozenset(row['consequents'])
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="business-rec">
                                <strong>🛒 Bundle Opportunity:</strong><br>
                                <strong>Base Product:</strong> {ant}<br>
                                <strong>Recommend:</strong> {cons}<br>
                                <small>
                                    • Success Rate: {row['confidence']:.1%}<br>
                                    • Lift: {row['lift']:.2f}x more likely<br>
                                    • Appears in {row['support']:.2%} of transactions
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Simple ROI indicator
                            roi_score = row['confidence'] * row['lift']
                            st.metric("ROI Score", f"{roi_score:.2f}", help="Confidence × Lift")
                    
                    st.markdown("---")
                    
                    # Bundle suggestions
                    st.markdown("#### 🎁 Product Bundle Suggestions")
                    
                    # Find itemsets with 2+ items
                    multi_itemsets = results['Apriori']['freq'][
                        results['Apriori']['freq']['itemsets'].apply(len) >= 2
                    ].sort_values('support', ascending=False).head(10)
                    
                    if not multi_itemsets.empty:
                        bundle_col1, bundle_col2 = st.columns(2)
                        
                        for idx, (i, row) in enumerate(multi_itemsets.iterrows()):
                            items = ', '.join(list(row['itemsets']))
                            
                            with bundle_col1 if idx % 2 == 0 else bundle_col2:
                                st.markdown(f"""
                                <div class="insight-box">
                                    <strong>Bundle #{idx+1}</strong><br>
                                    {items}<br>
                                    <small>Support: {row['support']:.2%}</small>
                                </div>
                                """, unsafe_allow_html=True)
                
                else:
                    st.info("No strong cross-selling opportunities found with current parameters")
            
            # ===== PRODUCT PLACEMENT =====
            with bi_tab2:
                st.subheader("🏪 Store Layout Optimization")
                
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Layout Strategy:</strong><br>
                    Place frequently associated products near each other to maximize impulse purchases and customer convenience.
                </div>
                """, unsafe_allow_html=True)
                
                # Get strong associations
                strong_assoc = rules[rules['lift'] > 2].copy()
                
                if not strong_assoc.empty:
                    # Product proximity recommendations
                    st.markdown("#### 📍 Product Proximity Recommendations")
                    
                    proximity_rules = strong_assoc.nlargest(15, 'lift')
                    
                    for idx, row in proximity_rules.iterrows():
                        ant = format_frozenset(row['antecedents'])
                        cons = format_frozenset(row['consequents'])
                        
                        st.markdown(f"""
                        <div class="business-rec">
                            <strong>Place Near Each Other:</strong><br>
                            📦 {ant} ↔️ {cons}<br>
                            <small>Association Strength: {row['lift']:.2f}x | Confidence: {row['confidence']:.1%}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Department-level recommendations
                    st.markdown("#### 🏬 Department-Level Insights")
                    
                    dept_summary = df_original.groupby('department').agg({
                        'order_id': 'nunique',
                        'product_name': 'nunique'
                    }).reset_index()
                    dept_summary.columns = ['Department', 'Orders', 'Products']
                    dept_summary = dept_summary.sort_values('Orders', ascending=False).head(10)
                    
                    fig = px.bar(
                        dept_summary,
                        x='Orders',
                        y='Department',
                        orientation='h',
                        title='Top 10 Departments by Order Volume',
                        color='Orders',
                        color_continuous_scale='Blues',
                        text='Orders'
                    )
                    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("No strong associations found for layout optimization")
            
            # ===== INVENTORY INTELLIGENCE =====
            with bi_tab3:
                st.subheader("📦 Inventory Management Intelligence")
                
                # Product velocity analysis
                product_freq = basket_data.sum().sort_values(ascending=False)
                
                # Categorize products (ABC Analysis)
                total_freq = product_freq.sum()
                cumsum_freq = product_freq.cumsum() / total_freq
                
                category_a = product_freq[cumsum_freq <= 0.7].index.tolist()
                category_b = product_freq[(cumsum_freq > 0.7) & (cumsum_freq <= 0.9)].index.tolist()
                category_c = product_freq[cumsum_freq > 0.9].index.tolist()
                
                col1, col2, col3 = st.columns(3)
                
                col1.markdown(f"""
                <div style="background: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;">
                    <h4 style="color: #155724; margin: 0;">Category A</h4>
                    <h2 style="color: #155724; margin: 0;">{len(category_a)}</h2>
                    <small>Fast-Moving (70% of sales)</small>
                </div>
                """, unsafe_allow_html=True)
                
                col2.markdown(f"""
                <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffc107;">
                    <h4 style="color: #856404; margin: 0;">Category B</h4>
                    <h2 style="color: #856404; margin: 0;">{len(category_b)}</h2>
                    <small>Medium-Moving (20% of sales)</small>
                </div>
                """, unsafe_allow_html=True)
                
                col3.markdown(f"""
                <div style="background: #f8d7da; padding: 1rem; border-radius: 10px; border-left: 5px solid #dc3545;">
                    <h4 style="color: #721c24; margin: 0;">Category C</h4>
                    <h2 style="color: #721c24; margin: 0;">{len(category_c)}</h2>
                    <small>Slow-Moving (10% of sales)</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Restock recommendations based on associations
                st.markdown("#### 🔄 Smart Restock Recommendations")
                
                st.info("💡 When restocking Category A products, also consider restocking their associated products to avoid stockouts")
                
                restock_recommendations = []
                for prod_a in category_a[:5]:  # Top 5 fast-movers
                    related_rules = rules[
                        (rules['antecedents'].apply(lambda x: prod_a in x)) |
                        (rules['consequents'].apply(lambda x: prod_a in x))
                    ].sort_values('lift', ascending=False).head(3)
                    
                    if not related_rules.empty:
                        for _, row in related_rules.iterrows():
                            ant = format_frozenset(row['antecedents'])
                            cons = format_frozenset(row['consequents'])
                            
                            st.markdown(f"""
                            <div class="business-rec">
                                <strong>⚡ Restock Alert:</strong><br>
                                {ant} → {cons}<br>
                                <small>Lift: {row['lift']:.2f} | Stock both together</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Product velocity chart
                st.markdown("#### 📊 Product Velocity Analysis")
                
                top_30 = product_freq.head(30)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=list(range(len(top_30))),
                    y=top_30.values,
                    text=top_30.index,
                    hovertemplate='<b>%{text}</b><br>Frequency: %{y}<extra></extra>',
                    marker=dict(
                        color=top_30.values,
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title='Top 30 Products by Velocity',
                    xaxis_title='Product Rank',
                    yaxis_title='Transaction Frequency',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ===== PERFORMANCE ANALYTICS =====
            with bi_tab4:
                st.subheader("📊 Business Performance Analytics")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                avg_basket_size = df_original.groupby('order_id')['product_id'].count().mean()
                col1.metric("🛒 Avg Basket Size", f"{avg_basket_size:.2f} items")
                
                total_orders = df_original['order_id'].nunique()
                col2.metric("📦 Total Orders", f"{total_orders:,}")
                
                products_per_user = df_original.groupby('user_id')['product_name'].nunique().mean()
                col3.metric("👤 Products per User", f"{products_per_user:.1f}")
                
                if results['Apriori']['metrics']:
                    strong_rules = results['Apriori']['metrics']['strong_rules']
                    col4.metric("💪 Strong Associations", f"{strong_rules:,}")
                
                st.markdown("---")
                
                # Market basket value analysis
                st.markdown("#### 💵 Market Basket Value Analysis")
                
                basket_size_dist = df_original.groupby('order_id')['product_id'].count()
                
                fig = px.histogram(
                    basket_size_dist,
                    nbins=30,
                    title='Distribution of Basket Sizes',
                    labels={'value': 'Number of Items', 'count': 'Number of Orders'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Top performing associations
                st.markdown("#### 🏆 Top Performing Associations")
                
                top_performers = rules.nlargest(10, 'lift').copy()
                top_performers['antecedents'] = top_performers['antecedents'].apply(format_frozenset)
                top_performers['consequents'] = top_performers['consequents'].apply(format_frozenset)
                top_performers['performance_score'] = top_performers['confidence'] * top_performers['lift']
                
                fig = px.bar(
                    top_performers,
                    x='performance_score',
                    y=top_performers['antecedents'] + ' → ' + top_performers['consequents'],
                    orientation='h',
                    title='Top 10 Associations by Performance Score (Confidence × Lift)',
                    labels={'performance_score': 'Performance Score', 'y': 'Association Rule'},
                    color='performance_score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 5: SMART RECOMMENDER
# ===================================================================

with tab5:
    st.header("🤔 Smart Product Recommendation Engine")
    
    if 'Apriori' not in results:
        st.warning("⚠️ Enable Apriori algorithm for recommendations")
    else:
        rules = results['Apriori']['rules']
        
        if rules.empty:
            st.warning("No rules available for recommendations")
        else:
            st.markdown("""
            <div class="insight-box">
                <strong>💡 How it works:</strong> Select products to simulate a shopping cart. 
                The system will recommend products based on association rules discovered by the Apriori algorithm.
            </div>
            """, unsafe_allow_html=True)
            
            # Cart simulation
            st.subheader("🛒 Simulate Shopping Cart")
            
            selected_products = st.multiselect(
                "Select products in cart:",
                options=sorted(product_names),
                help="Select one or more products"
            )
            
            if selected_products:
                st.markdown(f"**Cart contains {len(selected_products)} item(s):** {', '.join(selected_products)}")
                
                # Find recommendations
                selected_set = frozenset(selected_products)
                
                # Exact match recommendations
                exact_recommendations = rules[
                    rules['antecedents'] == selected_set
                ].sort_values('confidence', ascending=False)
                
                # Partial match recommendations (if no exact match)
                partial_recommendations = rules[
                    rules['antecedents'].apply(lambda x: x.issubset(selected_set) and x != selected_set)
                ].sort_values('lift', ascending=False)
                
                if not exact_recommendations.empty:
                    st.success(f"✅ Found {len(exact_recommendations)} recommendation(s) based on your exact cart")
                    
                    st.subheader("🎯 Recommended Products")
                    
                    for idx, row in exact_recommendations.head(10).iterrows():
                        cons = format_frozenset(row['consequents'])
                        
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"### ➡️ {cons}")
                        with col2:
                            st.metric("Confidence", f"{row['confidence']:.1%}")
                        with col3:
                            st.metric("Lift", f"{row['lift']:.2f}x")
                        with col4:
                            st.metric("Support", f"{row['support']:.2%}")
                        
                        st.markdown("---")
                    
                    # Visualization
                    st.subheader("📊 Recommendation Strength Visualization")
                    
                    rec_viz = exact_recommendations.head(10).copy()
                    rec_viz['product'] = rec_viz['consequents'].apply(format_frozenset)
                    rec_viz['score'] = rec_viz['confidence'] * rec_viz['lift']
                    
                    fig = px.bar(
                        rec_viz,
                        x='score',
                        y='product',
                        orientation='h',
                        title='Top 10 Recommendations (Confidence × Lift)',
                        color='lift',
                        color_continuous_scale='Greens',
                        labels={'score': 'Recommendation Score', 'product': 'Product'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif not partial_recommendations.empty:
                    st.info(f"💡 Found {len(partial_recommendations)} recommendation(s) based on items in your cart")
                    
                    st.subheader("🎯 Suggested Products")
                    
                    for idx, row in partial_recommendations.head(10).iterrows():
                        ant = format_frozenset(row['antecedents'])
                        cons = format_frozenset(row['consequents'])
                        
                        st.markdown(f"""
                        <div class="business-rec">
                            <strong>Based on:</strong> {ant}<br>
                            <strong>Suggest:</strong> {cons}<br>
                            <small>Confidence: {row['confidence']:.1%} | Lift: {row['lift']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.warning("🔍 No recommendations found for this combination")
                    st.info("💡 Try selecting fewer products or adjusting the parameters in the sidebar")
                    
                    # Show popular alternatives
                    st.subheader("🌟 Popular Products You Might Like")
                    
                    top_products = basket_data.sum().sort_values(ascending=False).head(10)
                    
                    for idx, (prod, freq) in enumerate(top_products.items(), 1):
                        if prod not in selected_products:
                            st.markdown(f"""
                            <div class="insight-box">
                                <strong>#{idx}. {prod}</strong><br>
                                <small>Purchased in {freq:,} transactions ({freq/total_transactions:.1%})</small>
                            </div>
                            """, unsafe_allow_html=True)
            
            else:
                st.info("👆 Select products above to get recommendations")
                
                # Show trending products
                st.subheader("🔥 Trending Products")
                
                top_10 = basket_data.sum().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=top_10.index,
                    y=top_10.values,
                    title='Top 10 Most Popular Products',
                    labels={'x': 'Product', 'y': 'Frequency'},
                    color=top_10.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 6: TEMPORAL ANALYSIS
# ===================================================================

with tab6:
    st.header("⏰ Temporal Pattern Analysis")
    
    st.markdown("""
    <div class="insight-box">
        <strong>📅 Time-Based Insights:</strong> Understand how shopping patterns vary across different time periods.
    </div>
    """, unsafe_allow_html=True)
    
    # Time period selector
    time_analysis = st.radio(
        "Select Analysis Type:",
        ['Overview', 'Weekday vs Weekend', 'Day of Week Analysis', 'Hour of Day Analysis'],
        horizontal=True
    )
    
    if time_analysis == 'Overview':
        # Overall temporal patterns
        col1, col2 = st.columns(2)
        
        with col1:
            if 'order_dow' in df_original.columns:
                day_dist = df_original['order_dow'].value_counts().sort_index()
                day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
                
                fig = px.bar(
                    x=[day_names[i] for i in day_dist.index],
                    y=day_dist.values,
                    title='Orders by Day of Week',
                    labels={'x': 'Day', 'y': 'Number of Orders'},
                    color=day_dist.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'order_hour_of_day' in df_original.columns:
                hour_dist = df_original['order_hour_of_day'].value_counts().sort_index()
                
                fig = px.line(
                    x=hour_dist.index,
                    y=hour_dist.values,
                    title='Orders by Hour of Day',
                    labels={'x': 'Hour', 'y': 'Number of Orders'},
                    markers=True
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    elif time_analysis == 'Weekday vs Weekend':
        if 'is_weekend' in df_original.columns:
            col1, col2 = st.columns(2)
            
            for is_weekend, label, col in [(False, 'Weekday', col1), (True, 'Weekend', col2)]:
                with col:
                    df_period = df_original[df_original['is_weekend'] == is_weekend]
                    
                    st.subheader(f"{'📅' if not is_weekend else '🎉'} {label}")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Orders", f"{df_period['order_id'].nunique():,}")
                    metric_col2.metric("Users", f"{df_period['user_id'].nunique():,}")
                    
                    # Top products
                    top_prods = df_period['product_name'].value_counts().head(5)
                    
                    st.markdown("**Top 5 Products:**")
                    for prod, count in top_prods.items():
                        st.text(f"• {prod}: {count:,}")
        else:
            st.info("Weekend information not available in dataset")
    
    elif time_analysis == 'Day of Week Analysis':
        if 'day_name' in df_original.columns:
            day_stats = df_original.groupby('day_name').agg({
                'order_id': 'nunique',
                'user_id': 'nunique',
                'product_id': 'count'
            }).reset_index()
            day_stats.columns = ['Day', 'Orders', 'Users', 'Items']
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats['Day'] = pd.Categorical(day_stats['Day'], categories=day_order, ordered=True)
            day_stats = day_stats.sort_values('Day')
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Orders by Day', 'Users by Day')
            )
            
            fig.add_trace(
                go.Bar(x=day_stats['Day'], y=day_stats['Orders'], name='Orders', marker_color='#1f77b4'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=day_stats['Day'], y=day_stats['Users'], name='Users', marker_color='#ff7f0e'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Hour of Day Analysis
        if 'order_hour_of_day' in df_original.columns and 'order_dow' in df_original.columns:
            # Heatmap
            heatmap_data = df_original.groupby(['order_dow', 'order_hour_of_day'])['order_id'].nunique().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='order_dow', columns='order_hour_of_day', values='order_id')
            
            day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            heatmap_pivot.index = [day_names[i] for i in heatmap_pivot.index]
            
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
                title="Order Heatmap: Day vs Hour",
                color_continuous_scale='YlOrRd',
                aspect='auto'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak hours analysis
            st.markdown("---")
            st.subheader("📊 Peak Hours Analysis")
            
            hour_stats = df_original.groupby('order_hour_of_day')['order_id'].nunique().sort_values(ascending=False)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🔥 Peak Hour", f"{hour_stats.index[0]}:00", f"{hour_stats.values[0]:,} orders")
            col2.metric("📈 2nd Peak", f"{hour_stats.index[1]}:00", f"{hour_stats.values[1]:,} orders")
            col3.metric("📊 3rd Peak", f"{hour_stats.index[2]}:00", f"{hour_stats.values[2]:,} orders")

# ===================================================================
# TAB 7: DEPARTMENT INTELLIGENCE
# ===================================================================

with tab7:
    st.header("🏬 Department-Specific Intelligence")
    
    departments = ['All Departments'] + sorted(df_original['department'].unique().tolist())
    selected_dept = st.selectbox("🔍 Select Department for Analysis:", departments)
    
    if selected_dept != 'All Departments':
        df_dept = df_original[df_original['department'] == selected_dept]
        
        st.subheader(f"📊 Analysis for: **{selected_dept}**")
        
        # Department metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏷️ Products", df_dept['product_name'].nunique())
        col2.metric("🛒 Orders", df_dept['order_id'].nunique())
        col3.metric("👥 Users", df_dept['user_id'].nunique())
        col4.metric("📦 Total Items", len(df_dept))
        
        st.markdown("---")
        
        # Run department-specific analysis
        with st.spinner(f"Analyzing {selected_dept}..."):
            dept_transactions = df_dept.groupby(['user_id'])['product_name'].unique().apply(list).tolist()
            
            if len(dept_transactions) > 0:
                te_dept = TransactionEncoder()
                te_ary_dept = te_dept.fit(dept_transactions).transform(dept_transactions)
                basket_dept = pd.DataFrame(te_ary_dept, columns=te_dept.columns_)
                
                freq_dept, time_dept = run_algorithm(basket_dept, 'Apriori', min_support)
                rules_dept = generate_rules(freq_dept, metric, min_threshold)
                
                st.success(f"✅ Analysis complete in {time_dept:.2f}s | Found {len(rules_dept)} rules")
                
                if not rules_dept.empty:
                    # Top rules for this department
                    st.subheader("🔝 Top Association Rules")
                    
                    top_dept_rules = rules_dept.nlargest(10, 'lift').copy()
                    top_dept_rules['antecedents'] = top_dept_rules['antecedents'].apply(format_frozenset)
                    top_dept_rules['consequents'] = top_dept_rules['consequents'].apply(format_frozenset)
                    
                    st.dataframe(
                        top_dept_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].style.format({
                            'support': '{:.2%}',
                            'confidence': '{:.2%}',
                            'lift': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Department-specific recommendations
                    st.markdown("---")
                    st.subheader("💡 Department Strategies")
                    
                    st.markdown(f"""
                    <div class="business-rec">
                        <strong>🎯 Cross-Selling Strategy:</strong><br>
                        Promote complementary products within {selected_dept} based on strong associations.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="business-rec">
                        <strong>🏪 Layout Optimization:</strong><br>
                        Arrange products with high lift values near each other in the {selected_dept} section.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No strong associations found. Try lowering thresholds.")
            else:
                st.error("No transactions found for this department")
    
    else:
        # Overview of all departments
        st.subheader("📊 Department Performance Overview")
        
        dept_summary = df_original.groupby('department').agg({
            'order_id': 'nunique',
            'product_name': 'nunique',
            'user_id': 'nunique'
        }).reset_index()
        dept_summary.columns = ['Department', 'Orders', 'Products', 'Users']
        dept_summary = dept_summary.sort_values('Orders', ascending=False)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                dept_summary.head(15),
                x='Orders',
                y='Department',
                orientation='h',
                title='Top 15 Departments by Orders',
                color='Orders',
                color_continuous_scale='Blues',
                text='Orders'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.treemap(
                dept_summary.head(20),
                path=['Department'],
                values='Orders',
                title='Department Distribution (Top 20)',
                color='Orders',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# TAB 8: VISUAL ANALYTICS
# ===================================================================

with tab8:
    st.header("🕸️ Advanced Visual Analytics")
    
    if 'Apriori' not in results:
        st.warning("⚠️ Enable Apriori algorithm for visualizations")
    else:
        rules = results['Apriori']['rules']
        
        if rules.empty:
            st.warning("No rules available for visualization")
        else:
            viz_type = st.radio(
                "Select Visualization Type:",
                ['Network Graph', 'Association Heatmap', 'Sunburst Chart', 'Parallel Coordinates'],
                horizontal=True
            )
            
            # ===== NETWORK GRAPH =====
            if viz_type == 'Network Graph':
                st.subheader("🕸️ Product Association Network")
                
                num_rules = st.slider("Number of top rules to visualize:", 5, min(100, len(rules)), 30)
                
                top_rules = rules.nlargest(num_rules, 'lift')
                
                # Create graph
                G = nx.DiGraph()
                
                for _, row in top_rules.iterrows():
                    ant = format_frozenset(row['antecedents'])
                    cons = format_frozenset(row['consequents'])
                    
                    G.add_node(ant, node_type='antecedent')
                    G.add_node(cons, node_type='consequent')
                    G.add_edge(ant, cons, weight=row['lift'], confidence=row['confidence'])
                
                # Layout
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                
                # Create edges
                edge_trace = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    edge_trace.append(
                        go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=edge[2]['weight']/2, color='#888'),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
                
                # Create nodes
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_color = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    degree = G.degree(node)
                    node_text.append(f"{node}<br>Connections: {degree}")
                    node_size.append(20 + degree * 5)
                    node_color.append(degree)
                
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=[n[:15] + '...' if len(n) > 15 else n for n in G.nodes()],
                    textposition="top center",
                    hovertext=node_text,
                    hoverinfo='text',
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        colorscale='YlOrRd',
                        showscale=True,
                        colorbar=dict(title="Connections"),
                        line=dict(width=2, color='white')
                    ),
                    textfont=dict(size=10)
                )
                
                # Create figure
                fig = go.Figure(
                    data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title=f'Product Association Network (Top {num_rules} Rules)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700,
                        plot_bgcolor='#f8f9fa'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📊 Network: {G.number_of_nodes()} products, {G.number_of_edges()} associations")
            
            # ===== ASSOCIATION HEATMAP =====
            elif viz_type == 'Association Heatmap':
                st.subheader("🔥 Product Association Heatmap")
                
                top_n = st.slider("Number of top products:", 10, 30, 15)
                top_products = basket_data.sum().nlargest(top_n).index.tolist()
                
                # Create lift matrix
                lift_matrix = pd.DataFrame(0.0, index=top_products, columns=top_products)
                
                for _, row in rules.iterrows():
                    ant_list = list(row['antecedents'])
                    cons_list = list(row['consequents'])
                    
                    if len(ant_list) == 1 and len(cons_list) == 1:
                        if ant_list[0] in top_products and cons_list[0] in top_products:
                            lift_matrix.loc[ant_list[0], cons_list[0]] = row['lift']
                
                fig = px.imshow(
                    lift_matrix,
                    labels=dict(x="Consequent Product", y="Antecedent Product", color="Lift"),
                    title=f"Association Strength Heatmap (Top {top_n} Products)",
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    text_auto='.2f'
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Higher lift (green) = stronger association")
            
            # ===== SUNBURST CHART =====
            elif viz_type == 'Sunburst Chart':
                st.subheader("☀️ Hierarchical Product View")
                
                # Create hierarchical data
                dept_prod_df = df_original.groupby(['department', 'product_name']).size().reset_index(name='count')
                dept_prod_df = dept_prod_df.nlargest(100, 'count')
                
                fig = px.sunburst(
                    dept_prod_df,
                    path=['department', 'product_name'],
                    values='count',
                    title='Product Distribution by Department',
                    color='count',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            # ===== PARALLEL COORDINATES =====
            else:  # Parallel Coordinates
                st.subheader("📊 Multi-Metric Analysis")
                
                # Prepare data
                parallel_data = rules.head(50).copy()
                parallel_data['rule_id'] = range(len(parallel_data))
                
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(
                            color=parallel_data['lift'],
                            colorscale='Viridis',
                            showscale=True,
                            cmin=parallel_data['lift'].min(),
                            cmax=parallel_data['lift'].max()
                        ),
                        dimensions=[
                            dict(label='Support', values=parallel_data['support']),
                            dict(label='Confidence', values=parallel_data['confidence']),
                            dict(label='Lift', values=parallel_data['lift']),
                            dict(label='Leverage', values=parallel_data['leverage'] if 'leverage' in parallel_data else parallel_data['support'])
                        ]
                    )
                )
                
                fig.update_layout(
                    title='Parallel Coordinates Plot of Rule Metrics',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Each line represents one association rule. Use this to identify patterns across multiple metrics.")

# ===================================================================
# TAB 9: REPORT GENERATOR
# ===================================================================

with tab9:
    st.header("📑 Report Generator & Export")
    
    st.markdown("""
    <div class="insight-box">
        <strong>📊 Generate comprehensive reports</strong> for stakeholders, presentations, or documentation.
    </div>
    """, unsafe_allow_html=True)
    
    report_type = st.radio(
        "Select Report Type:",
        ['Executive Summary', 'Technical Report', 'Business Recommendations'],
        horizontal=True
    )
    
    # ===== EXECUTIVE SUMMARY REPORT =====
    if report_type == 'Executive Summary':
        st.subheader("📊 Executive Summary Report")
        
        if 'Apriori' in results:
            rules = results['Apriori']['rules']
            
            # Generate report content
            report_content = f"""
# Market Basket Analysis - Executive Summary

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total Records: {total_records:,}
- Unique Transactions: {total_transactions:,}
- Unique Users: {total_users:,}
- Products Analyzed: {total_products:,}
- Departments: {total_departments:,}

## Analysis Results
- Algorithm Used: **Apriori**
- Execution Time: {results['Apriori']['time']:.2f} seconds
- Frequent Itemsets Found: {len(results['Apriori']['freq']):,}
- Association Rules Generated: {len(rules):,}

## Key Findings
"""
            if not rules.empty:
                top_5 = rules.nlargest(5, 'lift')
                report_content += "\n### Top 5 Strongest Associations:\n\n"
                for idx, row in top_5.iterrows():
                    ant = format_frozenset(row['antecedents'])
                    cons = format_frozenset(row['consequents'])
                    report_content += f"{idx+1}. **{ant}** → **{cons}**\n"
                    report_content += f"   - Confidence: {row['confidence']:.1%}\n"
                    report_content += f"   - Lift: {row['lift']:.2f}\n"
                    report_content += f"   - Support: {row['support']:.2%}\n\n"
                
                report_content += "\n## Business Recommendations\n\n"
                report_content += "1. **Cross-Selling**: Implement bundling strategies based on high-confidence rules\n"
                report_content += "2. **Store Layout**: Optimize product placement using lift values\n"
                report_content += "3. **Promotions**: Design targeted campaigns using association insights\n"
            
            st.markdown(report_content)
            
            # Download button
            st.download_button(
                label="📥 Download Executive Summary (Markdown)",
                data=report_content,
                file_name=f"executive_summary_{time.strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        else:
            st.warning("Enable Apriori algorithm to generate report")
    
    # ===== TECHNICAL REPORT =====
    elif report_type == 'Technical Report':
        st.subheader("🔬 Technical Analysis Report")
        
        if results:
            tech_report = f"""
# Market Basket Analysis - Technical Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Analyst:** Market Basket Analysis System

## Methodology

### Algorithm: Apriori
The Apriori algorithm was used to discover frequent itemsets and generate association rules.

### Parameters:
- Minimum Support: {min_support}
- Minimum {metric.capitalize()}: {min_threshold}
- Metric Used: {metric}

### Dataset Statistics:
- Total Records: {total_records:,}
- Transaction Count: {total_transactions:,}
- Product Count: {total_products:,}
- User Count: {total_users:,}

## Results
"""
            
            for algo_name, data in results.items():
                tech_report += f"\n### {algo_name} Algorithm\n"
                tech_report += f"- Execution Time: {data['time']:.4f} seconds\n"
                tech_report += f"- Frequent Itemsets: {len(data['freq']):,}\n"
                tech_report += f"- Association Rules: {len(data['rules']):,}\n"
                
                if data['metrics']:
                    tech_report += f"- Average Confidence: {data['metrics']['avg_confidence']:.2%}\n"
                    tech_report += f"- Average Lift: {data['metrics']['avg_lift']:.2f}\n"
                    tech_report += f"- Strong Rules (Lift>2): {data['metrics']['strong_rules']:,}\n"
            
            tech_report += "\n## Conclusion\n"
            tech_report += "The analysis successfully identified significant product associations that can be leveraged for business strategies.\n"
            
            st.markdown(tech_report)
            
            st.download_button(
                label="📥 Download Technical Report (Markdown)",
                data=tech_report,
                file_name=f"technical_report_{time.strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    # ===== BUSINESS RECOMMENDATIONS =====
    else:
        st.subheader("💼 Business Recommendations Report")
        
        if 'Apriori' in results:
            rules = results['Apriori']['rules']
            
            if not rules.empty:
                biz_report = f"""
# Business Action Plan - Market Basket Analysis

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Overview
This report provides actionable business recommendations based on association rule mining analysis.

## Key Opportunities

### 1. Cross-Selling & Bundling
"""
                # Top bundling opportunities
                top_bundles = rules[rules['confidence'] >= 0.3].nlargest(10, 'lift')
                
                for idx, row in top_bundles.iterrows():
                    ant = format_frozenset(row['antecedents'])
                    cons = format_frozenset(row['consequents'])
                    biz_report += f"\n**Opportunity #{idx+1}:**\n"
                    biz_report += f"- Bundle: {ant} + {cons}\n"
                    biz_report += f"- Success Probability: {row['confidence']:.1%}\n"
                    biz_report += f"- Strength: {row['lift']:.2f}x above random\n"
                    biz_report += f"- Action: Create promotional bundle or place together\n"
                
                biz_report += "\n### 2. Store Layout Optimization\n"
                biz_report += "Place strongly associated products in proximity to encourage impulse purchases.\n"
                
                biz_report += "\n### 3. Inventory Management\n"
                biz_report += "Ensure associated products are stocked together to avoid missed cross-selling opportunities.\n"
                
                biz_report += "\n### 4. Promotional Strategy\n"
                biz_report += "Offer discounts on antecedent products to drive sales of higher-margin consequent products.\n"
                
                biz_report += "\n## Implementation Timeline\n"
                biz_report += "- Week 1-2: Test top 3 bundling recommendations\n"
                biz_report += "- Week 3-4: Adjust store layout based on associations\n"
                biz_report += "- Month 2: Implement inventory synchronization\n"
                biz_report += "- Month 3: Launch targeted promotional campaigns\n"
                
                st.markdown(biz_report)
                
                st.download_button(
                    label="📥 Download Business Recommendations (Markdown)",
                    data=biz_report,
                    file_name=f"business_recommendations_{time.strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            else:
                st.warning("No rules available for recommendations")
    
    st.markdown("---")
    
    # Export data section
    st.subheader("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Apriori' in results and not results['Apriori']['rules'].empty:
            rules_export = results['Apriori']['rules'].copy()
            rules_export['antecedents'] = rules_export['antecedents'].apply(format_frozenset)
            rules_export['consequents'] = rules_export['consequents'].apply(format_frozenset)
            
            csv_rules = rules_export.to_csv(index=False)
            st.download_button(
                label="📥 Export Rules (CSV)",
                data=csv_rules,
                file_name=f"association_rules_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'Apriori' in results and not results['Apriori']['freq'].empty:
            freq_export = results['Apriori']['freq'].copy()
            freq_export['itemsets'] = freq_export['itemsets'].apply(format_frozenset)
            
            csv_freq = freq_export.to_csv(index=False)
            st.download_button(
                label="📥 Export Itemsets (CSV)",
                data=csv_freq,
                file_name=f"frequent_itemsets_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        config = {
            'min_support': min_support,
            'metric': metric,
            'min_threshold': min_threshold, 
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        config_json = json.dumps(config, indent=2)
        
        st.download_button(
            label="📥 Export Config (JSON)",
            data=config_json,
            file_name=f"analysis_config_{time.strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
# ===================================================================
# FOOTER
# ===================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
    <h3 style='color: #1f77b4; margin-bottom: 1rem;'>Market Basket Analysis Dashboard</h3>
    <p><strong>Algoritma Utama:</strong> Apriori (Primary) | FP-Growth (Comparison)</p>
    <p><strong>Powered by:</strong> Streamlit • MLxtend • Plotly • NetworkX</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        🎓 <em>Sistem Informasi Ritel Berbasis Analisis Asosiasi Produk</em>
    </p>
    <p style='font-size: 0.8rem; color: #999;'>
        © 2025 Market Basket Analysis System | Data-Driven Decision Making
    </p>
</div>
""", unsafe_allow_html=True)