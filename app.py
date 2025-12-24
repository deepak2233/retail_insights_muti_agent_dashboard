"""
Retail Insights AI - Enterprise Analytics Platform
Advanced Multi-Agent GenAI System with Modern Business UI
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import get_orchestrator, reset_orchestrator
from utils.data_layer import get_data_layer
from utils.memory import get_memory, reset_memory
from config import settings

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Retail Insights AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# MODERN BUSINESS CSS
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    /* Hero Header */
    .hero-section {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='80' cy='20' r='40' fill='rgba(255,255,255,0.1)'/%3E%3Ccircle cx='90' cy='80' r='30' fill='rgba(255,255,255,0.05)'/%3E%3C/svg%3E");
        background-size: cover;
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
        margin-top: 0.3rem;
        position: relative;
        z-index: 1;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.2);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.25rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
    }
    
    .kpi-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: white;
        margin: 0.2rem 0;
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }
    
    .kpi-trend {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    
    .trend-up {
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
    }
    
    .trend-down {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: white;
        margin: 0;
    }
    
    .section-desc {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
    }
    
    /* AI Chat Interface */
    .chat-container {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
    }
    
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .quick-btn {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .quick-btn:hover {
        background: rgba(99, 102, 241, 0.3);
        transform: scale(1.02);
    }
    
    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        max-width: 90%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .ai-message {
        background: linear-gradient(145deg, #334155 0%, #1e293b 100%);
        color: rgba(255,255,255,0.9);
        border: 1px solid rgba(255,255,255,0.1);
        border-bottom-left-radius: 4px;
    }
    
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.75rem;
    }
    
    .confidence-bar {
        height: 4px;
        background: rgba(255,255,255,0.1);
        border-radius: 2px;
        margin-top: 0.75rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.5s ease;
    }
    
    .conf-high { background: linear-gradient(90deg, #4ade80, #22c55e); }
    .conf-med { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
    .conf-low { background: linear-gradient(90deg, #f87171, #ef4444); }
    
    /* Chart Cards */
    .chart-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .chart-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
        font-weight: 600;
        color: white;
    }
    
    /* Data Table */
    .data-table {
        background: transparent;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.4) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        padding: 0.5rem;
        border-radius: 14px;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255,255,255,0.6);
        border-radius: 10px;
        padding: 0.6rem 1.25rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.6) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 1rem;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 2rem;
    }
    
    /* Responsive */
    @media (max-width: 992px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    @media (max-width: 576px) {
        .kpi-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
def init_session():
    """Initialize session state"""
    defaults = {
        'messages': [],
        'orchestrator': None,
        'data_layer': None,
        'initialized': False,
        'pending_question': '',
        'uploaded_data': None,
        'eval_history': []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def check_api_key():
    """Check if API key is available from secrets or environment"""
    from config import get_secret
    
    # Check for Google API key
    api_key = get_secret("GOOGLE_API_KEY")
    if api_key:
        return True, "google", api_key
    
    # Check for OpenAI API key
    api_key = get_secret("OPENAI_API_KEY")
    if api_key:
        return True, "openai", api_key
    
    return False, None, None


def load_system():
    """Load data and initialize agents"""
    try:
        # First check if API key is available
        has_key, provider, _ = check_api_key()
        if not has_key:
            st.warning("⚠️ No API key found. Please configure in Streamlit Cloud Secrets or sidebar.")
            return False
        
        st.session_state.data_layer = get_data_layer()
        st.session_state.orchestrator = get_orchestrator()
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.info("💡 Make sure GOOGLE_API_KEY is set in Streamlit Cloud Secrets")
        return False


# ============================================================================
# COMPONENTS
# ============================================================================
def render_hero():
    """Render hero header section"""
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">📊 Retail Insights AI</h1>
            <p class="hero-subtitle">Enterprise Analytics Platform • Multi-Agent AI • Real-time Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.initialized:
            st.markdown("""
            <div style="display:flex;justify-content:flex-end;padding-top:1rem;">
                <div class="status-pill">
                    <span class="status-dot"></span>
                    Online
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_kpis():
    """Render KPI dashboard cards"""
    if not st.session_state.initialized:
        return
    
    try:
        stats = st.session_state.data_layer.get_summary_stats()
        o = stats.get("overall", {})
        
        revenue = o.get('total_revenue', 0)
        orders = o.get('total_orders', 0)
        aov = o.get('avg_order_value', 0)
        cancelled = o.get('cancelled_orders', 0)
        cancel_rate = (cancelled / orders * 100) if orders > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">₹{revenue/10000000:.2f} Cr</div>
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-trend trend-up">📈 Sales Data</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📦</div>
                <div class="kpi-value">{orders:,}</div>
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-trend trend-up">✓ Active</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🛒</div>
                <div class="kpi-value">₹{aov:,.0f}</div>
                <div class="kpi-label">Avg Order Value</div>
                <div class="kpi-trend trend-up">Per Order</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">{cancel_rate:.1f}%</div>
                <div class="kpi-label">Cancellation Rate</div>
                <div class="kpi-trend trend-down">{cancelled:,} orders</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"KPI Error: {e}")


def render_ai_chat():
    """Render AI chat interface"""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🤖</div>
        <div>
            <h2 class="section-title">AI Insights Assistant</h2>
            <p class="section-desc">Ask questions in natural language about your retail data</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialized:
        st.warning("⏳ Initializing system...")
        return
    
    # Quick action buttons
    st.markdown("**Quick Insights:**")
    questions = {
        "💰 Revenue Analysis": "What is the total revenue and how is it distributed across categories?",
        "🏆 Top Performers": "Which are the top 5 states and categories by revenue?",
        "📈 Trends": "Show the monthly revenue and order trend",
        "📊 B2B vs B2C": "Compare B2B and B2C sales performance in detail",
        "❌ Cancellations": "Analyze the cancellation rate by category and state",
        "📦 Orders": "What is the average order value by fulfillment type?"
    }
    
    cols = st.columns(len(questions))
    for i, (label, q) in enumerate(questions.items()):
        with cols[i]:
            if st.button(label, key=f"q_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Ask your question",
            value=st.session_state.get('pending_question', ''),
            placeholder="e.g., What is the revenue breakdown by state?",
            label_visibility="collapsed"
        )
    with col2:
        submit = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    # Process question
    if submit and user_input:
        with st.spinner("🧠 Analyzing..."):
            try:
                answer = st.session_state.orchestrator.process_query(user_input)
                
                confidence = 85
                if st.session_state.orchestrator.evaluation:
                    es = st.session_state.orchestrator.get_evaluation_summary()
                    confidence = es.get('overall', 0.85) * 100
                
                st.session_state.messages.append({
                    "q": user_input,
                    "a": answer,
                    "time": datetime.now().strftime("%H:%M"),
                    "conf": confidence
                })
                st.session_state.pending_question = ''
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display messages
    if st.session_state.messages:
        st.markdown("<hr style='border-color:rgba(255,255,255,0.1);margin:1.5rem 0;'>", unsafe_allow_html=True)
        
        for msg in reversed(st.session_state.messages[-5:]):
            conf = msg.get('conf', 85)
            conf_class = 'conf-high' if conf >= 80 else 'conf-med' if conf >= 60 else 'conf-low'
            
            st.markdown(f"""
            <div class="message-bubble user-message">
                <div class="message-header">
                    <span>You</span>
                    <span>{msg['time']}</span>
                </div>
                {msg['q']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="message-bubble ai-message">
                <div class="message-header">
                    <span>🤖 AI Assistant</span>
                    <span>Confidence: {conf:.0f}%</span>
                </div>
                <div style="line-height:1.6;">{msg['a']}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {conf_class}" style="width:{conf}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat", key="clear"):
            st.session_state.messages = []
            if st.session_state.orchestrator:
                st.session_state.orchestrator.clear_memory()
            st.rerun()


def render_analytics():
    """Render analytics dashboard"""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div>
            <h2 class="section-title">Analytics Dashboard</h2>
            <p class="section-desc">Visual insights from your retail data</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialized:
        return
    
    try:
        stats = st.session_state.data_layer.get_summary_stats()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-card"><div class="chart-header">🏆 Top 10 States by Revenue</div>', unsafe_allow_html=True)
            if stats.get("top_states"):
                df = pd.DataFrame(stats["top_states"]).head(10)
                fig = px.bar(
                    df, x='state', y='revenue',
                    color='revenue',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=60),
                    xaxis=dict(tickangle=45)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-card"><div class="chart-header">📦 Revenue by Category</div>', unsafe_allow_html=True)
            if stats.get("by_category"):
                df = pd.DataFrame(stats["by_category"])
                fig = px.pie(
                    df.head(8), values='revenue', names='category',
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation='v', yanchor='middle', y=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Trend chart
        st.markdown('<div class="chart-card"><div class="chart-header">📈 Revenue & Profit Trend</div>', unsafe_allow_html=True)
        if stats.get("monthly_trend"):
            df = pd.DataFrame(stats["monthly_trend"])
            df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['period'], y=df['revenue'],
                name='Revenue',
                mode='lines+markers',
                line=dict(color='#8b5cf6', width=3),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.2)'
            ))
            fig.add_trace(go.Scatter(
                x=df['period'], y=df['profit'],
                name='Profit',
                mode='lines+markers',
                line=dict(color='#4ade80', width=3)
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=280,
                margin=dict(l=20, r=20, t=10, b=20),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data tables
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 State Performance**")
            if stats.get("top_states"):
                df = pd.DataFrame(stats["top_states"]).head(10)
                df['revenue'] = df['revenue'].apply(lambda x: f"₹{x:,.0f}")
                df['orders'] = df['orders'].apply(lambda x: f"{x:,}")
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**📋 Category Performance**")
            if stats.get("by_category"):
                df = pd.DataFrame(stats["by_category"])
                df['revenue'] = df['revenue'].apply(lambda x: f"₹{x:,.0f}")
                df['orders'] = df['orders'].apply(lambda x: f"{x:,}")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
    except Exception as e:
        st.error(f"Analytics Error: {e}")


def render_data_upload():
    """Render data upload section"""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📁</div>
        <div>
            <h2 class="section-title">Data Upload</h2>
            <p class="section-desc">Upload new CSV data and refresh analytics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Upload your retail sales data (CSV format)**
        
        Expected columns:
        - `order_id` - Unique order identifier
        - `date` - Order date
        - `category` - Product category
        - `state` - Customer state
        - `amount` - Order amount
        - `quantity` - Quantity ordered
        - `status` - Order status
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your retail data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
                
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("**Column Info:**")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Sample': [str(df[c].iloc[0])[:30] if len(df) > 0 else '' for c in df.columns]
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)
                
                if st.button("🔄 Load This Data", type="primary"):
                    # Save to data folder
                    save_path = "data/uploaded_data.csv"
                    df.to_csv(save_path, index=False)
                    st.session_state.uploaded_data = save_path
                    
                    # Reload system with new data
                    reset_orchestrator()
                    reset_memory()
                    st.session_state.initialized = False
                    st.session_state.data_layer = None
                    st.session_state.orchestrator = None
                    st.success("✅ Data loaded! Refreshing system...")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with col2:
        st.markdown("**Current Data:**")
        if st.session_state.initialized:
            stats = st.session_state.data_layer.get_summary_stats()
            o = stats.get("overall", {})
            st.metric("Records", f"{o.get('total_orders', 0):,}")
            st.metric("Date Range", o.get('date_range', 'N/A'))
            st.metric("Revenue", f"₹{o.get('total_revenue', 0)/10000000:.2f} Cr")
        else:
            st.info("No data loaded")
        
        st.markdown("---")
        st.markdown("**Sample Data:**")
        if st.button("📥 Download Sample CSV"):
            sample_data = """order_id,date,category,state,amount,quantity,status
ORD001,2024-01-15,Electronics,Maharashtra,15000,2,Shipped
ORD002,2024-01-16,Clothing,Karnataka,3500,3,Delivered
ORD003,2024-01-17,Home,Delhi,8000,1,Shipped
ORD004,2024-01-18,Electronics,Tamil Nadu,22000,1,Delivered
ORD005,2024-01-19,Clothing,Gujarat,4500,2,Cancelled"""
            st.download_button(
                "Download",
                data=sample_data,
                file_name="sample_retail_data.csv",
                mime="text/csv"
            )


def render_evaluation_dashboard():
    """Render evaluation metrics dashboard"""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <div>
            <h2 class="section-title">Evaluation Metrics</h2>
            <p class="section-desc">AI quality metrics and performance analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialized:
        st.warning("⚠️ System not initialized")
        return
    
    orchestrator = st.session_state.orchestrator
    
    # Get evaluation summary
    if orchestrator and orchestrator.evaluation:
        eval_summary = orchestrator.get_evaluation_summary()
        
        # Main metrics
        st.markdown("### 📊 Overall Quality Metrics")
        
        cols = st.columns(5)
        metrics = [
            ("🎯 Accuracy", eval_summary.get('accuracy', 0), "SQL query correctness"),
            ("✅ Faithfulness", eval_summary.get('faithfulness', 0), "Response grounded in data"),
            ("🔍 Relevance", eval_summary.get('relevance', 0), "Answer addresses question"),
            ("📋 Completeness", eval_summary.get('completeness', 0), "Full answer provided"),
            ("⭐ Overall", eval_summary.get('overall', 0), "Combined quality score")
        ]
        
        for i, (name, value, desc) in enumerate(metrics):
            with cols[i]:
                score = value * 100
                color = "#4ade80" if score >= 80 else "#fbbf24" if score >= 60 else "#f87171"
                st.markdown(f"""
                <div style="background:linear-gradient(145deg,#1e293b,#334155);padding:1rem;border-radius:12px;text-align:center;border:1px solid rgba(255,255,255,0.1);">
                    <div style="font-size:1.5rem;font-weight:700;color:{color};">{score:.1f}%</div>
                    <div style="font-size:0.9rem;color:white;margin:0.3rem 0;">{name}</div>
                    <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f"<br>*Based on {eval_summary.get('total_evaluations', 0)} evaluated queries*", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed metrics table
        st.markdown("### 📋 Metric Definitions")
        
        metric_defs = pd.DataFrame([
            {"Metric": "Accuracy", "Description": "How well the generated SQL matches expected query patterns", "Target": "≥ 85%"},
            {"Metric": "Faithfulness", "Description": "Percentage of response statements grounded in actual data", "Target": "≥ 90%"},
            {"Metric": "Relevance", "Description": "How directly the response addresses the user's question", "Target": "≥ 85%"},
            {"Metric": "Completeness", "Description": "Whether the response provides a full, comprehensive answer", "Target": "≥ 80%"},
            {"Metric": "Overall", "Description": "Weighted average: 25% Accuracy + 30% Faithfulness + 25% Relevance + 20% Completeness", "Target": "≥ 80%"}
        ])
        st.dataframe(metric_defs, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Query history with scores
        st.markdown("### 📜 Query Evaluation History")
        
        if st.session_state.messages:
            history_data = []
            for i, msg in enumerate(st.session_state.messages):
                history_data.append({
                    "#": i + 1,
                    "Question": msg['q'][:50] + "..." if len(msg['q']) > 50 else msg['q'],
                    "Confidence": f"{msg.get('conf', 0):.0f}%",
                    "Time": msg.get('time', 'N/A')
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
        else:
            st.info("No queries yet. Ask questions in the AI Assistant tab to see evaluation metrics.")
        
        st.markdown("---")
        
        # Evaluation parameters
        st.markdown("### ⚙️ Evaluation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Query Evaluation Weights:**")
            st.markdown("""
            | Parameter | Weight |
            |-----------|--------|
            | SQL Accuracy | 25% |
            | Faithfulness | 30% |
            | Relevance | 25% |
            | Completeness | 20% |
            """)
        
        with col2:
            st.markdown("**Confidence Thresholds:**")
            st.markdown("""
            | Level | Score |
            |-------|-------|
            | 🟢 High | ≥ 80% |
            | 🟡 Medium | 60-79% |
            | 🔴 Low | < 60% |
            """)
    else:
        st.info("Evaluation metrics will appear after you make queries in the AI Assistant tab.")


def render_reports():
    """Render executive reports section"""
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📝</div>
        <div>
            <h2 class="section-title">Executive Reports</h2>
            <p class="section-desc">AI-generated comprehensive business insights</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialized:
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Generate a comprehensive AI-powered executive report including:
        
        - **📊 Revenue Analysis** — Total sales, growth patterns, and distribution
        - **🗺️ Regional Performance** — State-wise breakdown and top performers  
        - **📦 Category Insights** — Product category analysis and trends
        - **⚙️ Operational Metrics** — Fulfillment efficiency and cancellations
        - **💡 Strategic Recommendations** — Data-driven action items
        """)
    
    with col2:
        if st.button("🚀 Generate Report", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is generating executive report..."):
                try:
                    summary = st.session_state.orchestrator.generate_summary()
                    st.markdown("---")
                    st.markdown(summary)
                    
                    st.download_button(
                        "📥 Download Report",
                        data=summary,
                        file_name=f"executive_report_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


def render_system_panel():
    """Render system status panel"""
    with st.expander("⚙️ System Configuration", expanded=False):
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown("**🔧 LLM**")
            st.info(f"{settings.llm_provider.upper()}")
            model = settings.gemini_model if settings.llm_provider == 'google' else settings.openai_model
            st.info(f"{model}")
        
        with cols[1]:
            st.markdown("**📊 Data**")
            if st.session_state.initialized:
                stats = st.session_state.data_layer.get_summary_stats()
                o = stats.get("overall", {})
                st.success(f"{o.get('total_orders', 0):,} records")
                st.info(f"{o.get('date_range', 'N/A')}")
        
        with cols[2]:
            st.markdown("**🧠 Session**")
            st.info(f"{len(st.session_state.messages)} queries")
            if st.session_state.orchestrator and st.session_state.orchestrator.evaluation:
                es = st.session_state.orchestrator.get_evaluation_summary()
                st.info(f"Quality: {es.get('overall', 0)*100:.0f}%")
        
        with cols[3]:
            st.markdown("**🎛️ Actions**")
            if st.button("🔄 Reset", use_container_width=True):
                reset_orchestrator()
                reset_memory()
                for k in ['initialized', 'orchestrator', 'data_layer', 'messages']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()


def render_footer():
    """Render app footer"""
    st.markdown("""
    <div class="app-footer">
        <p><strong>Retail Insights AI</strong> — Enterprise Analytics Platform</p>
        <p>Multi-Agent AI • LangChain • LangGraph • DuckDB • Gemini</p>
        <p style="margin-top:0.5rem;">Developed by <strong>Deepak Yadav</strong> | dk.yadav125566@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    init_session()
    
    # Sidebar for API key configuration (fallback)
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        has_key, provider, _ = check_api_key()
        
        if has_key:
            st.success(f"✅ Using {provider.upper()} API")
        else:
            st.warning("⚠️ No API key configured")
            st.markdown("**Option 1:** Set in Streamlit Cloud Secrets")
            st.code("GOOGLE_API_KEY = 'your-key-here'", language="toml")
            
            st.markdown("**Option 2:** Enter below (session only)")
            api_key_input = st.text_input("Google API Key", type="password", key="api_key_input")
            if api_key_input:
                import os
                os.environ["GOOGLE_API_KEY"] = api_key_input
                st.success("✅ API key set for this session")
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Get API Key:**")
        st.markdown("[Google AI Studio](https://aistudio.google.com/app/apikey)")
    
    # Auto-initialize
    if not st.session_state.initialized:
        with st.spinner("🚀 Starting Retail Insights AI..."):
            load_system()
    
    # Render components
    render_hero()
    render_kpis()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI Assistant", "📊 Analytics", "📁 Data Upload", "📈 Evaluation", "📝 Reports"])
    
    with tab1:
        render_ai_chat()
    
    with tab2:
        render_analytics()
    
    with tab3:
        render_data_upload()
    
    with tab4:
        render_evaluation_dashboard()
    
    with tab5:
        render_reports()
    
    render_system_panel()
    render_footer()


if __name__ == "__main__":
    main()
