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


def render_ai_chat():
    """Render modern AI chat interface"""
    st.markdown("""
    <div class="section-header">
        <div>
            <h2 class="section-title">AI Insights Assistant</h2>
            <p class="section-desc">Interactive retail intelligence with advanced guardrails</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.initialized:
        st.warning("⏳ Initializing system...")
        return

    # Clear chat button in a reasonable spot
    col_e1, col_e2 = st.columns([5, 1])
    with col_e2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. How can I help you now?", "time": datetime.now().strftime("%H:%M")}]
            if st.session_state.orchestrator:
                st.session_state.orchestrator.clear_memory()
            st.rerun()
    
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Display chart if available
            if msg.get("chart") and msg["chart"].get("figure"):
                st.plotly_chart(msg["chart"]["figure"], use_container_width=True)
            
            st.markdown(msg["content"])
            if "conf" in msg:
                st.caption(f"Confidence: {msg['conf']:.0f}% | {msg['time']}")
            else:
                st.caption(msg["time"])

    # Quick action buttons (above chat input)
    st.markdown("**Suggestions:**")
    questions = {
        "Revenue": "What is the total revenue and how is it distributed across categories?",
        "Top States": "Which are the top 5 states by revenue?",
        "Cancellations": "Analyze the cancellation rate by category",
        "SQL Query": "Show the sql query for top 5 categories by revenue"
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
            "time": datetime.now().strftime("%H:%M")
        })
        # Display immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    report_content = st.session_state.get('report_content')
                    # Get full state result to access chart data
                    result_state = st.session_state.orchestrator.graph.invoke({
                        "question": prompt,
                        "query_intent": None,
                        "query_result": None,
                        "validation_passed": False,
                        "final_answer": "",
                        "error": None,
                        "confidence_scores": None,
                        "conversation_context": None,
                        "edge_case_handled": None,
                        "facts": None,
                        "report_content": report_content
                    })
                    
                    answer = result_state.get("final_answer", "I couldn't generate a response.")
                    chart_data = result_state.get("chart_data")
                    
                    # Display chart first if available
                    if chart_data and chart_data.get("figure"):
                        st.plotly_chart(chart_data["figure"], use_container_width=True, key=f"chart_{len(st.session_state.messages)}")
                    
                    confidence = 85
                    if st.session_state.orchestrator.evaluation:
                        es = st.session_state.orchestrator.get_evaluation_summary()
                        confidence = es.get('overall', 0.85) * 100
                    
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "time": datetime.now().strftime("%H:%M"),
                        "conf": confidence,
                        "chart": chart_data
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
    """Render scalability architecture section"""
    st.markdown("""
    <div class="section-header">
        <div>
            <h2 class="section-title">Scalability & Architecture</h2>
            <p class="section-desc">Design for 100GB+ enterprise retail data systems</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["System Design", "Data Engineering", "Storage & Query", "Cost Analysis"])
    
    with tabs[0]:
        st.markdown("### High-Level Enterprise Architecture")
        st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│  Streamlit / React / API Gateway                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   LangGraph  │  │  Query Cache │  │   Session    │         │
│  │ Orchestrator │  │    (Redis)   │  │  Management  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Query Agent  │  │Extract Agent │  │Validate Agent│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    DATA RETRIEVAL LAYER                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │         Vector Database (Semantic Search)             │      │
│  │  FAISS / Pinecone / Weaviate / ChromaDB              │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │         Metadata Store (Query Optimization)           │      │
│  │  PostgreSQL / DynamoDB                                │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    DATA WAREHOUSE LAYER                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │         Cloud Data Warehouse (OLAP)                   │      │
│  │  Snowflake / BigQuery / Redshift / Databricks        │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                      DATA LAKE LAYER                             │
│  ┌──────────────────────────────────────────────────────┐      │
│  │         Object Storage (Raw Data)                     │      │
│  │  AWS S3 / Google Cloud Storage / Azure Blob          │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
        """)
        
        st.info("💡 **Core Strategy**: Decouple storage from compute and use a multi-layered retrieval strategy to handle massive datasets while maintaining low latency.")

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Data Ingestion")
            st.markdown("""
            - **Batch ETL**: Apache Spark for processing massive CSV/Parquet files.
            - **Streaming**: Kafka + Spark Structured Streaming for real-time transactions.
            - **Data Format**: Delta Lake or Iceberg for ACID transactions on S3.
            """)
        with col2:
            st.markdown("#### Preprocessing")
            st.markdown("""
            - **Cleaning**: Automated PII masking and outlier removal.
            - **Enrichment**: Geographic mapping and product categorization.
            - **Partitioning**: Multi-level partitioning by `year/month/region`.
            """)
        
        st.markdown("#### Example PySpark Transformation")
        st.code("""
df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("year", "month", "region") \
    .save("s3://retail-data-lake/processed/sales")
        """, language="python")

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Storage Architecture")
            st.markdown("""
            - **Raw**: Parquet/ORC on Cloud Object Storage.
            - **Analytical**: Snowflake/BigQuery for complex JOINs.
            - **Semantic**: FAISS/Pinecone for RAG-based context retrieval.
            """)
        with col2:
            st.markdown("#### Optimization")
            st.markdown("""
            - **Intelligent Routing**: Route queries based on data size.
            - **Query Caching**: Redis layer for 80% faster repetitive queries.
            - **Materialized Views**: Pre-aggregated tables for dashboards.
            """)

    with tabs[3]:
        st.markdown("#### Estimated Monthly Cost (100GB Dataset)")
        cost_data = [
            {"Component": "Storage (S3)", "Cost": "$2.30", "Notes": "Standard tier"},
            {"Component": "Data Warehouse", "Cost": "$72.00", "Notes": "Snowflake X-Small"},
            {"Component": "Vector DB", "Cost": "$70.00", "Notes": "Pinecone Starter"},
            {"Component": "LLM API", "Cost": "$100-500", "Notes": "GPT-4 usage"},
            {"Component": "Compute", "Cost": "$30.00", "Notes": "t3.medium instance"}
        ]
        st.table(cost_data)
        st.success("**Projected Total**: ~$300 - $700 per month depending on traffic.")

def render_system_panel():
    """Render system status panel"""
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
                stats = st.session_state.data_layer.get_summary_stats()
                o = stats.get("overall", {})
                st.success(f"{o.get('total_orders', 0):,} records")
                st.info(f"{o.get('date_range', 'N/A')}")
        
        with cols[2]:
            st.markdown("**Session**")
            st.info(f"{len(st.session_state.messages)} queries")
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
