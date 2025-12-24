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
        'pending_question': ''
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_system():
    """Load data and initialize agents"""
    try:
        st.session_state.data_layer = get_data_layer()
        st.session_state.orchestrator = get_orchestrator()
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
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
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    init_session()
    
    # Auto-initialize
    if not st.session_state.initialized:
        with st.spinner("🚀 Starting Retail Insights AI..."):
            load_system()
    
    # Render components
    render_hero()
    render_kpis()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🤖 AI Assistant", "📊 Analytics", "📝 Reports"])
    
    with tab1:
        render_ai_chat()
    
    with tab2:
        render_analytics()
    
    with tab3:
        render_reports()
    
    render_system_panel()
    render_footer()


if __name__ == "__main__":
    main()
