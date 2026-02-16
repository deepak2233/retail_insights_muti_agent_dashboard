"""
Retail Insights AI - Enterprise Analytics Platform
Advanced Multi-Agent GenAI System with Modern Business UI
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
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
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODERN BUSINESS CSS
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Professional Dark SaaS Theme */
    :root {
        --primary: #818cf8;
        --secondary: #6366f1;
        --bg-main: #0f172a;
        --card-bg: #1e293b;
        --border: #334155;
        --text-main: #e2e8f0;
        --text-muted: #94a3b8;
        --accent-glow: rgba(129, 140, 248, 0.15);
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-main);
        color: var(--text-main);
    }

    /* Dark Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-muted) !important;
    }

    /* KPI Cards - Dark & Modern */
    .kpi-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: var(--primary);
        box-shadow: 0 4px 16px var(--accent-glow);
    }

    .kpi-value { color: #f1f5f9; }
    .kpi-label { color: var(--text-muted); }
    .kpi-trend { color: var(--text-muted); font-size: 0.8rem; }
    .trend-up { color: #4ade80; }
    .trend-down { color: #f87171; }

    /* Chat Messages - Dark bubbles */
    .chat-window {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
    }

    .user-message {
        background: var(--primary);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }

    .ai-message {
        background: #334155;
        color: var(--text-main);
        border: 1px solid var(--border);
        margin-right: auto;
        border-bottom-left-radius: 2px;
    }

    .message-meta {
        font-size: 0.75rem;
        margin-bottom: 0.25rem;
        color: var(--text-muted);
        display: block;
    }

    /* Buttons & Inputs */
    .stButton button {
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }

    /* Section Headers */
    .section-title {
        color: #f1f5f9 !important;
    }
    .section-desc {
        color: var(--text-muted) !important;
    }

    /* Chart Cards */
    .chart-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .chart-header {
        color: var(--text-muted);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Footer */
    .app-footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        color: var(--text-muted);
        border-top: 1px solid var(--border);
    }
    .app-footer p { margin: 0.2rem 0; }

    /* Status Pill */
    .status-pill {
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .status-dot {
        width: 8px; height: 8px;
        background: #4ade80;
        border-radius: 50%;
        display: inline-block;
    }

    /* Streamlit widget overrides for dark mode */
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); }
    .stTabs [aria-selected="true"] { color: var(--primary) !important; }
    .stExpander { border-color: var(--border) !important; }

    /* Pipeline Badges */
    .pipeline-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
    .badge {
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .badge-cache    { background: rgba(74,222,128,0.15); color: #4ade80; }
    .badge-spell    { background: rgba(96,165,250,0.15); color: #60a5fa; }
    .badge-guard    { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .badge-dup      { background: rgba(148,163,184,0.18); color: #94a3b8; }
    .badge-topic    { background: rgba(192,132,252,0.15); color: #c084fc; }
    .badge-circuit  { background: rgba(248,113,113,0.15); color: #f87171; }
    .badge-intent   { background: rgba(129,140,248,0.12); color: #818cf8; }
    .badge-time     { background: rgba(148,163,184,0.10); color: #94a3b8; }
    .badge-entity   { background: rgba(45,212,191,0.15); color: #2dd4bf; }
</style>
""", unsafe_allow_html=True)

# Custom Sidebar Styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
def init_session():
    """Initialize session state"""
    defaults = {
        'messages': [{"role": "assistant", "content": "Hello! I'm your Retail Insights Assistant. How can I help you analyze your data today?", "time": datetime.now().strftime("%H:%M")}],
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
    from config import get_secret, settings
    
    # Check based on configured provider first
    provider = get_secret("LLM_PROVIDER", settings.llm_provider)
    
    if provider == "groq":
        # Check for dedicated Groq API key
        api_key = get_secret("GROQ_API_KEY", settings.groq_api_key)
        if api_key:
            return True, "groq", api_key
            
    if provider == "openai":
        # Check for OpenAI/OpenRouter API key
        api_key = get_secret("OPENAI_API_KEY", settings.openai_api_key)
        if api_key:
            return True, "openai", api_key
    
    # Check for Google API key
    api_key = get_secret("GOOGLE_API_KEY", settings.google_api_key)
    if api_key:
        return True, "google", api_key
    
    # Fallback search if no provider set or provider-specific key missing
    for p in ["groq", "openai", "google"]:
        env_key = f"{p.upper()}_API_KEY"
        api_key = get_secret(env_key)
        if api_key:
            return True, p, api_key
    
    return False, None, None


def load_system():
    """Load data and initialize agents"""
    try:
        # First check if API key is available
        has_key, provider, _ = check_api_key()
        if not has_key:
            st.warning("No API key found. Please configure in Streamlit Cloud Secrets or sidebar.")
            return False
        
        st.session_state.data_layer = get_data_layer()
        st.session_state.orchestrator = get_orchestrator()
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.info("Make sure GOOGLE_API_KEY is set in Streamlit Cloud Secrets")
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
            <h1 class="hero-title">Retail Insights AI</h1>
            <p class="hero-subtitle">Enterprise Analytics Platform | Multi-Agent AI | Real-time Insights</p>
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
                <div class="kpi-value">₹{revenue/10000000:.2f} Cr</div>
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-trend trend-up">Sales Data</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{orders:,}</div>
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-trend trend-up">Active</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">₹{aov:,.0f}</div>
                <div class="kpi-label">Avg Order Value</div>
                <div class="kpi-trend trend-up">Per Order</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{cancel_rate:.1f}%</div>
                <div class="kpi-label">Cancellation Rate</div>
                <div class="kpi-trend trend-down">{cancelled:,} orders</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"KPI Error: {e}")


def _build_pipeline_badges_html(flags: dict, timings: dict = None) -> str:
    """Build HTML badge strip from pipeline flags."""
    badges = []
    if flags.get("cache_hit"):
        badges.append('<span class="badge badge-cache">Cache Hit</span>')
    if flags.get("spell_corrected"):
        orig = flags.get("original_query", "")
        norm = flags.get("normalized_query", "")
        if orig != norm:
            badges.append(f'<span class="badge badge-spell">Auto-corrected: "{orig[:30]}" -> "{norm[:30]}"</span>')
        else:
            badges.append('<span class="badge badge-spell">Preprocessed</span>')
    if flags.get("duplicate_detected"):
        badges.append('<span class="badge badge-dup">Duplicate — reused answer</span>')
    if flags.get("guardrail_blocked"):
        gtype = flags.get("guardrail_type", "blocked")
        badges.append(f'<span class="badge badge-guard">Guardrail: {gtype}</span>')
    if flags.get("topic_shift"):
        badges.append('<span class="badge badge-topic">Topic Shift Detected</span>')
    if flags.get("circuit_breaker_open"):
        badges.append('<span class="badge badge-circuit">Circuit Breaker — Fallback</span>')
    if flags.get("entity_context_used"):
        badges.append('<span class="badge badge-entity">Entity Memory Used</span>')
    intent = flags.get("intent", "")
    if intent:
        badges.append(f'<span class="badge badge-intent">Intent: {intent}</span>')
    conf = flags.get("confidence", 0)
    if conf > 0:
        badges.append(f'<span class="badge badge-intent">Conf: {conf*100:.0f}%</span>')
    if timings and timings.get("total_ms"):
        badges.append(f'<span class="badge badge-time">{timings["total_ms"]:.0f}ms</span>')
    if not badges:
        return ""
    return '<div class="pipeline-badges">' + "".join(badges) + '</div>'


def render_ai_chat():
    """Render modern AI chat interface with pipeline visibility"""
    st.markdown("""
    <div class="section-header">
        <div>
            <h2 class="section-title">AI Insights Assistant</h2>
            <p class="section-desc">Production pipeline with guardrails, memory, caching, and spell correction</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.initialized:
        st.warning("Initializing system...")
        return

    # Clear chat button
    col_e1, col_e2 = st.columns([5, 1])
    with col_e2:
        if st.button("Clear Chat", use_container_width=True, type="primary"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Chat cleared. How can I help you now?",
                "time": datetime.now().strftime("%H:%M"),
            }]
            if st.session_state.orchestrator:
                st.session_state.orchestrator.clear_memory()
            st.rerun()

    # Display message history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            # Chart if available
            if msg.get("chart") and msg["chart"].get("figure"):
                st.plotly_chart(msg["chart"]["figure"], use_container_width=True, key=f"hist_chart_{idx}")
            st.markdown(msg["content"])
            # Pipeline badges
            if msg.get("pipeline_flags"):
                badges_html = _build_pipeline_badges_html(
                    msg["pipeline_flags"], msg.get("timings")
                )
                if badges_html:
                    st.markdown(badges_html, unsafe_allow_html=True)
                # Collapsible SQL
                if msg.get("sql_query"):
                    with st.expander("SQL Query", expanded=False):
                        st.code(msg["sql_query"], language="sql")
            elif msg.get("time"):
                st.caption(msg["time"])

    # Suggestion buttons
    st.markdown("**Suggestions:**")
    questions = {
        "Revenue Analysis": "What is the total revenue and how is it distributed across categories?",
        "Top States": "Which are the top 5 states by revenue?",
        "Cancellation Rate": "Analyze the cancellation rate by category",
        "Hello!": "Hello, what can you do?",
        "Off-topic Test": "What is the weather today?",
    }

    rows = st.columns(len(questions))
    for i, (label, q) in enumerate(questions.items()):
        if rows[i].button(label, key=f"q_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    # Chat input
    prompt = st.chat_input("Ask about sales, trends, or specific reports...")

    # Handle pending question from buttons
    if st.session_state.get('pending_question'):
        prompt = st.session_state.pending_question
        st.session_state.pending_question = ''

    if prompt:
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "time": datetime.now().strftime("%H:%M"),
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response via metadata-rich pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing through production pipeline..."):
                try:
                    report_content = st.session_state.get('report_content')
                    orch = st.session_state.orchestrator
                    result = orch.process_query_with_metadata(prompt, report_content)

                    answer = result.get("answer", "I couldn't generate a response.")
                    flags = result.get("pipeline_flags", {})
                    timings = result.get("timings", {})
                    sql_query = result.get("sql_query")
                    chart_data = result.get("chart_data")

                    # Display chart first if available
                    if chart_data and chart_data.get("figure"):
                        st.plotly_chart(
                            chart_data["figure"], use_container_width=True,
                            key=f"chart_{len(st.session_state.messages)}",
                        )

                    st.markdown(answer)

                    # Pipeline badges
                    badges_html = _build_pipeline_badges_html(flags, timings)
                    if badges_html:
                        st.markdown(badges_html, unsafe_allow_html=True)

                    # SQL query expander
                    if sql_query:
                        with st.expander("SQL Query", expanded=False):
                            st.code(sql_query, language="sql")

                    # Confidence
                    confidence = flags.get("confidence", 0) * 100
                    if confidence == 0 and orch.evaluation:
                        es = orch.get_evaluation_summary()
                        confidence = es.get('overall', 0.85) * 100

                    # Store message with all metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "time": datetime.now().strftime("%H:%M"),
                        "conf": confidence,
                        "chart": chart_data,
                        "pipeline_flags": flags,
                        "timings": timings,
                        "sql_query": sql_query,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


def render_analytics():
    """Render analytics dashboard"""
    st.markdown("""
    <div class="section-header">
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
        overall = stats.get("overall", {})
        revenue = overall.get("total_revenue", 0)
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-card"><div class="chart-header">Top 10 States by Revenue</div>', unsafe_allow_html=True)
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
                    font_color='#e2e8f0',
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=60),
                    xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.08)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-card"><div class="chart-header">Revenue by Category</div>', unsafe_allow_html=True)
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
                    font_color='#e2e8f0',
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation='v', yanchor='middle', y=0.5, font=dict(color='#e2e8f0'))
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Status and Fulfillment
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('<div class="chart-card"><div class="chart-header">Order Status Distribution</div>', unsafe_allow_html=True)
            if stats.get("by_status"):
                df_status = pd.DataFrame(stats["by_status"])
                fig_status = px.bar(
                    df_status, x='status', y='orders',
                    color='status',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_status.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=300,
                    margin=dict(l=20, r=20, t=10, b=40),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
                )
                st.plotly_chart(fig_status, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="chart-card"><div class="chart-header">Revenue by Fulfillment Method</div>', unsafe_allow_html=True)
            # Sample fulfillment data if not in stats
            fulfillment_data = stats.get("by_fulfillment", [{"method": "B2C", "revenue": revenue * 0.7}, {"method": "B2B", "revenue": revenue * 0.3}])
            df_fill = pd.DataFrame(fulfillment_data)
            fig_fill = px.pie(
                df_fill, values='revenue', names='method',
                hole=0.4,
                color_discrete_sequence=['#6366f1', '#a855f7']
            )
            fig_fill.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                height=300,
                margin=dict(l=20, r=20, t=10, b=10),
                legend=dict(font=dict(color='#e2e8f0'))
            )
            st.plotly_chart(fig_fill, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Trend chart
        st.markdown('<div class="chart-card"><div class="chart-header">Revenue and Profit Trend</div>', unsafe_allow_html=True)
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
                font_color='#e2e8f0',
                height=280,
                margin=dict(l=20, r=20, t=10, b=20),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='#e2e8f0')),
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.15)')
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Tables Section
        st.markdown("### Regional and Category Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**State Performance**")
            if stats.get("top_states"):
                df = pd.DataFrame(stats["top_states"]).head(10)
                df['revenue'] = df['revenue'].apply(lambda x: f"₹{x:,.0f}")
                df['orders'] = df['orders'].apply(lambda x: f"{x:,}")
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Category Performance**")
            if stats.get("by_category"):
                df = pd.DataFrame(stats["by_category"])
                df['revenue'] = df['revenue'].apply(lambda x: f"₹{x:,.0f}")
                df['orders'] = df['orders'].apply(lambda x: f"{x:,}")
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Raw Data Explorer
        with st.expander("Deep Dive: Raw Data Explorer"):
            df_full = st.session_state.data_layer.get_raw_data(500)
            st.markdown(f"Displaying sample of records from the database.")
            st.dataframe(df_full, use_container_width=True)
                
    except Exception as e:
        st.error(f"Analytics Error: {e}")


def render_data_upload():
    """Render data upload section"""
    st.markdown("""
    <div class="section-header">
        <div>
            <h2 class="section-title">Data Upload</h2>
            <p class="section-desc">Upload new CSV data and refresh analytics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Upload retail data or summarized reports**
        
        Supported formats:
        - **Structured Data**: CSV, Excel (.xlsx), JSON
        - **Summarized Reports**: Text (.txt), JSON, or Excel
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json', 'txt'],
            help="Upload sales data or a business report"
        )
        
        if uploaded_file is not None:
            file_ext = Path(uploaded_file.name).suffix.lower()
            try:
                # Handle structured data (CSV, XLSX, JSON)
                if file_ext in ['.csv', '.xlsx', '.json']:
                    if file_ext == '.csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_ext == '.xlsx':
                        df = pd.read_excel(uploaded_file)
                    else:
                        df = pd.read_json(uploaded_file)
                        
                    st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
                    st.markdown("**Data Preview:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("Load as Main Dataset", type="primary"):
                        # Save and refresh
                        save_path = f"data/uploaded_data{file_ext}"
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.session_state.data_layer.load_file(save_path)
                        reset_orchestrator()
                        reset_memory()
                        st.success("Data loaded! Refreshing system...")
                        st.rerun()
                
                # Handle text reports
                elif file_ext == '.txt':
                    content = uploaded_file.read().decode("utf-8")
                    st.info("Uploaded a text-based report.")
                    st.text_area("Report Content Preview", content[:500] + "...", height=150)
                    
                    if st.button("Use as Context for AI", type="primary"):
                        st.session_state.report_content = content
                        st.success("Report added to AI context!")
                        if st.session_state.orchestrator:
                            # We'll need to update orchestrator to handle this
                            pass
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
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
        if st.button("Download Sample CSV"):
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
        <div>
            <h2 class="section-title">Evaluation Metrics</h2>
            <p class="section-desc">AI quality metrics and performance analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialized:
        st.warning("System not initialized")
        return
    
    orchestrator = st.session_state.orchestrator
    
    # Get evaluation summary
    if orchestrator and orchestrator.evaluation:
        eval_summary = orchestrator.get_evaluation_summary()
        
        # Main metrics
        st.markdown("### Overall Quality Metrics")
        
        cols = st.columns(5)
        metrics = [
            ("Accuracy", eval_summary.get('accuracy', 0), "SQL query correctness"),
            ("Faithfulness", eval_summary.get('faithfulness', 0), "Response grounded in data"),
            ("Relevance", eval_summary.get('relevance', 0), "Answer addresses question"),
            ("Completeness", eval_summary.get('completeness', 0), "Full answer provided"),
            ("Overall", eval_summary.get('overall', 0), "Combined quality score")
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
        st.markdown("### Query Evaluation History")
        
        if st.session_state.messages:
            history_data = []
            for i in range(len(st.session_state.messages)):
                msg = st.session_state.messages[i]
                if msg["role"] == "assistant" and "conf" in msg:
                    # Try to find the preceding user question
                    question = "Unknown Question"
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        question = st.session_state.messages[i-1]["content"]
                    
                    history_data.append({
                        "#": len(history_data) + 1,
                        "Question": question[:50] + "..." if len(question) > 50 else question,
                        "Confidence": f"{msg.get('conf', 0):.0f}%",
                        "Time": msg.get('time', 'N/A')
                    })
            
            if history_data:
                st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
            else:
                st.info("No analytical queries evaluated yet.")
        else:
            st.info("No queries yet. Ask questions in the AI Assistant tab to see evaluation metrics.")
        
        st.markdown("---")
        
        # Evaluation parameters
        st.markdown("### Evaluation Parameters")
        
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
            | High | ≥ 80% |
            | Medium | 60-79% |
            | Low | < 60% |
            """)
    else:
        st.info("Evaluation metrics will appear after you make queries in the AI Assistant tab.")


def render_reports():
    """Render executive reports section"""
    st.markdown("""
    <div class="section-header">
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
        
        - **Revenue Analysis** — Total sales, growth patterns, and distribution
        - **Regional Performance** — State-wise breakdown and top performers  
        - **Category Insights** — Product category analysis and trends
        - **Operational Metrics** — Fulfillment efficiency and cancellations
        - **Strategic Recommendations** — Data-driven action items
        """)
    
    with col2:
        if st.button("Generate Report", type="primary", use_container_width=True):
            with st.spinner("AI is generating executive report..."):
                try:
                    summary = st.session_state.orchestrator.generate_summary()
                    st.markdown("---")
                    st.markdown(summary)
                    
                    st.download_button(
                        "Download Report",
                        data=summary,
                        file_name=f"executive_report_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


def render_architecture():
    """Render actual production pipeline architecture"""
    st.markdown("""
    <div class="section-header">
        <div>
            <h2 class="section-title">Production Pipeline Architecture</h2>
            <p class="section-desc">Live LangGraph pipeline with all production features</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["LangGraph Pipeline", "Production Features", "Scalability Design", "Cost Analysis"])

    with tabs[0]:
        st.markdown("### Live Agent Pipeline (LangGraph)")
        st.markdown("This is the **actual pipeline** running in this application:")

        # Mermaid flowchart of the real LangGraph pipeline
        mermaid_diagram = """
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({startOnLoad:true, theme:'dark', themeVariables:{
            primaryColor:'#1e3a5f', primaryTextColor:'#e2e8f0', primaryBorderColor:'#60a5fa',
            lineColor:'#94a3b8', secondaryColor:'#312e81', tertiaryColor:'#164e63',
            fontSize:'14px'
        }});</script>
        <div class="mermaid" style="background:transparent;">
flowchart TD
    A["User Query"] --> B["Preprocess Node"]
    B -->|"Injection / Edge Case"| EC["Handle Edge Case"]
    B -->|"Cache Hit"| POST["Postprocess Node"]
    B -->|"Clean Query"| RI["Route Intent Node"]

    RI -->|"analytics"| QA["Query Agent"]
    RI -->|"greeting / capability"| POST
    RI -->|"out_of_scope / off-topic"| POST
    RI -->|"duplicate"| POST
    RI -->|"appreciation"| POST

    QA -->|"NL to SQL + Entity Context"| EA["Extract Agent"]
    EA -->|"Success"| VA["Validate Agent"]
    EA -->|"Error + retries left"| QA
    EA -->|"Max retries reached"| ERR["Error Handler"]

    VA -->|"Pass"| FE["Fact Extractor"]
    VA -->|"Fail"| ERR

    FE --> RA["Response Agent"]
    RA --> POST

    POST --> DONE["Response + Pipeline Metadata"]

    EC --> DONE2["Safe Response"]
    ERR --> DONE3["Error Response"]

    style B fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    style RI fill:#312e81,stroke:#818cf8,color:#e2e8f0
    style QA fill:#164e63,stroke:#22d3ee,color:#e2e8f0
    style EA fill:#164e63,stroke:#22d3ee,color:#e2e8f0
    style VA fill:#164e63,stroke:#22d3ee,color:#e2e8f0
    style FE fill:#164e63,stroke:#22d3ee,color:#e2e8f0
    style RA fill:#164e63,stroke:#22d3ee,color:#e2e8f0
    style POST fill:#14532d,stroke:#4ade80,color:#e2e8f0
    style EC fill:#78350f,stroke:#fbbf24,color:#e2e8f0
    style ERR fill:#7f1d1d,stroke:#f87171,color:#e2e8f0
    style DONE fill:#0f172a,stroke:#94a3b8,color:#e2e8f0
    style DONE2 fill:#0f172a,stroke:#94a3b8,color:#e2e8f0
    style DONE3 fill:#0f172a,stroke:#94a3b8,color:#e2e8f0
        </div>
        """
        import streamlit.components.v1 as components
        components.html(mermaid_diagram, height=700, scrolling=True)

        st.markdown("---")
        st.markdown("#### Node Descriptions")
        node_desc = {
            "Preprocess": "Spell correction, abbreviation expansion, synonym mapping, injection detection, cache lookup",
            "Route Intent": "Fast regex + LLM classification → analytics, greeting, capability, off-topic, duplicate, appreciation",
            "Query Agent": "NL → SQL translation with entity memory context, circuit breaker protection",
            "Extract Agent": "Executes SQL on DuckDB, with retry on error (up to 3x with error learning)",
            "Validate Agent": "Confidence scoring and data quality checks",
            "Fact Extractor": "Extracts verifiable facts from query results to prevent hallucination",
            "Response Agent": "Generates grounded natural language response, cross-checks against facts",
            "Postprocess": "Updates memory, caches response, runs evaluation, logs timing",
        }
        for node, desc in node_desc.items():
            st.markdown(f"- **{node}**: {desc}")

    with tabs[1]:
        st.markdown("### Production Features (All Active)")
        features = pd.DataFrame([
            {"Feature": "Query Preprocessor", "Status": "Active", "Description": "Spell correction, abbreviation expansion, synonym mapping, whitespace normalization"},
            {"Feature": "Injection Detection", "Status": "Active", "Description": "Blocks SQL injection, XSS, and prompt injection attempts"},
            {"Feature": "Response Cache (LRU + TTL)", "Status": "Active", "Description": "Caches responses with 5-minute TTL, skips full pipeline on cache hit"},
            {"Feature": "Semantic Duplicate Detection", "Status": "Active", "Description": "Detects similar questions using SequenceMatcher (>85% threshold)"},
            {"Feature": "Topic-Shift Detection", "Status": "Active", "Description": "Router agent detects when user switches topics mid-conversation"},
            {"Feature": "Entity Memory & Decay", "Status": "Active", "Description": "Tracks mentioned entities with decay weights for context-aware SQL"},
            {"Feature": "Circuit Breaker", "Status": "Active", "Description": "Opens after 3 LLM failures, auto-resets after 60s cooldown"},
            {"Feature": "Guardrails (Router Agent)", "Status": "Active", "Description": "Detects greetings, off-topic, out-of-scope, and redirects constructively"},
            {"Feature": "Fact Extraction", "Status": "Active", "Description": "Extracts verifiable facts from data to prevent hallucination"},
            {"Feature": "Grounded Response Validation", "Status": "Active", "Description": "Cross-checks response against extracted facts for faithfulness"},
            {"Feature": "Per-Node Timing", "Status": "Active", "Description": "Tracks latency for each pipeline node (avg, p95, max)"},
            {"Feature": "Retry with Error Learning", "Status": "Active", "Description": "Up to 3 retries with previous errors fed back to improve SQL"},
            {"Feature": "Conversation Compaction", "Status": "Active", "Description": "Auto-summarizes older turns when memory exceeds threshold"},
            {"Feature": "Session Analytics", "Status": "Active", "Description": "Tracks latency, error rate, cache hit rate, p95 per session"},
        ])
        st.dataframe(features, use_container_width=True, hide_index=True)

        # Live pipeline monitor
        st.markdown("### Live Node Timings")
        orch = st.session_state.orchestrator
        if orch:
            timings = orch.get_node_timings()
            if timings:
                timing_data = []
                for node, stats in timings.items():
                    timing_data.append({
                        "Node": node,
                        "Calls": stats["count"],
                        "Avg (ms)": f"{stats['avg_ms']:.0f}",
                        "P95 (ms)": f"{stats['p95_ms']:.0f}",
                        "Max (ms)": f"{stats['max_ms']:.0f}",
                    })
                st.dataframe(pd.DataFrame(timing_data), use_container_width=True, hide_index=True)
            else:
                st.info("No timing data yet. Ask some questions in the AI Assistant tab.")

    with tabs[2]:
        st.markdown("### Scalability Design (100GB+)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Data Layer")
            st.markdown("""
            - **Batch ETL**: Apache Spark for massive CSV/Parquet
            - **Streaming**: Kafka + Spark Structured Streaming
            - **Data Format**: Delta Lake / Iceberg for ACID
            - **Partitioning**: year/month/region multi-level
            """)
            st.markdown("#### Storage")
            st.markdown("""
            - **Raw**: Parquet/ORC on Cloud Object Storage
            - **Analytical**: Snowflake/BigQuery for complex JOINs
            - **Semantic**: FAISS/Pinecone for RAG retrieval
            """)
        with col2:
            st.markdown("#### Optimization")
            st.markdown("""
            - **Query Routing**: Route by data size
            - **Redis Cache**: 80% faster repetitive queries
            - **Materialized Views**: Pre-aggregated dashboards
            - **Connection Pooling**: DuckDB thread safety
            """)
            st.markdown("#### Resilience")
            st.markdown("""
            - **Circuit Breaker**: Auto-fallback on LLM failure
            - **Graceful Degradation**: Optional modules fail safely
            - **Rate Limiting**: Per-user request throttling
            - **Retry with Learning**: Feed errors back to LLM
            """)

    with tabs[3]:
        st.markdown("#### Estimated Monthly Cost (100GB Dataset)")
        cost_data = [
            {"Component": "Storage (S3)", "Cost": "$2.30", "Notes": "Standard tier"},
            {"Component": "Data Warehouse", "Cost": "$72.00", "Notes": "Snowflake X-Small"},
            {"Component": "Vector DB", "Cost": "$70.00", "Notes": "Pinecone Starter"},
            {"Component": "LLM API", "Cost": "$100-500", "Notes": "GPT-4 / Groq usage"},
            {"Component": "Compute", "Cost": "$30.00", "Notes": "t3.medium instance"},
        ]
        st.table(cost_data)
        st.success("**Projected Total**: ~$300 - $700/month depending on traffic")

def render_system_panel():
    """Render system status panel with pipeline monitor"""
    with st.expander("System Configuration", expanded=False):
        cols = st.columns(4)

        with cols[0]:
            st.markdown("**LLM**")
            st.info(f"{settings.llm_provider.upper()}")
            model = settings.gemini_model if settings.llm_provider == 'google' else settings.openai_model
            st.info(f"{model}")

        with cols[1]:
            st.markdown("**Data**")
            if st.session_state.initialized:
                try:
                    stats = st.session_state.data_layer.get_summary_stats()
                    o = stats.get("overall", {})
                    st.success(f"{o.get('total_orders', 0):,} records")
                    st.info(f"{o.get('date_range', 'N/A')}")
                except Exception:
                    st.info("Data loaded")

        with cols[2]:
            st.markdown("**Session**")
            st.info(f"{len(st.session_state.messages)} messages")
            if st.session_state.orchestrator and st.session_state.orchestrator.evaluation:
                es = st.session_state.orchestrator.get_evaluation_summary()
                st.info(f"Quality: {es.get('overall', 0)*100:.0f}%")

        with cols[3]:
            st.markdown("**Actions**")
            if st.button("Reset", use_container_width=True):
                reset_orchestrator()
                reset_memory()
                for k in ['initialized', 'orchestrator', 'data_layer', 'messages']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    # Pipeline Monitor
    with st.expander("Pipeline Monitor", expanded=False):
        orch = st.session_state.get("orchestrator")
        if orch:
            cols = st.columns(4)
            # Circuit breaker
            cb_status = "OPEN" if orch.circuit_breaker.is_open else "CLOSED"
            cb_color = "error" if orch.circuit_breaker.is_open else "success"
            with cols[0]:
                st.markdown("**Circuit Breaker**")
                getattr(st, cb_color)(cb_status)

            # Cache stats
            with cols[1]:
                st.markdown("**Cache**")
                if orch.memory and hasattr(orch.memory, 'cache'):
                    cache_stats = orch.memory.cache.stats
                    hit_rate = cache_stats.get('hit_rate', 0) * 100
                    st.info(f"Hit Rate: {hit_rate:.0f}%")
                    st.caption(f"Size: {cache_stats.get('size', 0)}")
                else:
                    st.info("N/A")

            # Memory
            with cols[2]:
                st.markdown("**Memory**")
                if orch.memory:
                    turns = len(orch.memory.short_term)
                    entities = len(orch.memory.entity_memory)
                    st.info(f"{turns} turns")
                    st.caption(f"{entities} entities tracked")
                else:
                    st.info("Disabled")

            # Average latency
            with cols[3]:
                st.markdown("**Latency**")
                if orch.memory and hasattr(orch.memory, 'analytics'):
                    analytics = orch.memory.analytics
                    avg_lat = analytics.get('avg_latency_ms', 0)
                    p95_lat = analytics.get('p95_latency_ms', 0)
                    st.info(f"Avg: {avg_lat:.0f}ms")
                    st.caption(f"P95: {p95_lat:.0f}ms")
                else:
                    timings = orch.get_node_timings()
                    if timings:
                        all_totals = []
                        for n, s in timings.items():
                            all_totals.append(s['avg_ms'])
                        st.info(f"Nodes: {sum(all_totals):.0f}ms")
                    else:
                        st.info("No data")
        else:
            st.info("Orchestrator not initialized")


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
        st.markdown("### Configuration")
        
        has_key, provider, _ = check_api_key()
        
        if has_key:
            st.success(f"Active: {provider.upper()} API")
            if provider == "groq":
                st.info(f"Model: {settings.groq_model}")
        else:
            st.warning("No API key configured")
            st.markdown("**Enter API Key (session only):**")
            
            # Provider selector
            selected_provider = st.selectbox("LLM Provider", ["Groq", "OpenAI", "Google"], index=0)
            
            api_key_input = st.text_input(f"{selected_provider} API Key", type="password", key="api_key_input")
            if api_key_input:
                import os
                provider_key = f"{selected_provider.upper()}_API_KEY"
                os.environ[provider_key] = api_key_input
                os.environ["LLM_PROVIDER"] = selected_provider.lower()
                st.success(f"{selected_provider} key set for this session")
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Get API Key:**")
        st.markdown("[Google AI Studio](https://aistudio.google.com/app/apikey)")
    
    # Auto-initialize
    if not st.session_state.initialized:
        with st.spinner("Starting Retail Insights AI..."):
            load_system()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        selected_page = st.radio(
            "Select Section",
            ["AI Assistant", "Analytics", "Data Upload", "Architecture", "Evaluation", "Reports"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        render_system_panel()
        st.markdown("---")

    # Render Header (always visible)
    render_hero()
    
    # Conditional Rendering based on Navigation
    if selected_page == "AI Assistant":
        render_kpis()
        render_ai_chat()
    
    elif selected_page == "Analytics":
        render_analytics()
        
    elif selected_page == "Data Upload":
        render_data_upload()
        
    elif selected_page == "Architecture":
        render_architecture()
        
    elif selected_page == "Evaluation":
        render_evaluation_dashboard()
        
    elif selected_page == "Reports":
        render_reports()
    
    render_footer()


if __name__ == "__main__":
    main()
