# Retail Insights AI

**Multi-Agent GenAI System for Retail Analytics**

A production-ready AI-powered analytics platform that enables natural language querying of retail sales data, automated insight generation, and scalable analytics.

---

## Features

- **AI-Powered Q&A** - Ask questions in natural language
- **Multi-Source Analysis** - Combines raw database results with uploaded business reports
- **Interactive Analytics** - Visual dashboards with Plotly charts
- **Flexible Data Upload** - Support for CSV, Excel (.xlsx), and JSON
- **Report Summarization** - Analyze text-based business reports (.txt)
- **Architecture Visualization** - In-app scalability architecture for 100GB+ scale
- **Evaluation Metrics** - Monitor AI quality and performance
- **Multi-Agent System** - 5 specialized AI agents with LangGraph

---

## Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/deepak2233/retaisl_insights_agent.git
cd retaisl_insights_agent

# Create .env file
echo "GOOGLE_API_KEY=your-api-key" > .env
echo "GEMINI_MODEL=gemini-2.5-flash" >> .env
echo "LLM_PROVIDER=google" >> .env

# Run
docker-compose up --build
```

### Option 2: Local Python
```bash
git clone https://github.com/deepak2233/retaisl_insights_agent.git
cd retaisl_insights_agent

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file (same as above)
streamlit run app.py
```

**Access the app at:** http://localhost:8501

---

## 🔑 API Key

Get a free Google AI API key at: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## 🏗️ Architecture

```
User Question → Query Agent → Data Extraction → Validation → Response Generation → Answer
```

| Agent | Function |
|-------|----------|
| Router Agent | Classifies intent and routes to appropriate pipeline |
| Query Agent | Converts natural language to SQL |
| Extraction Agent | Executes SQL on DuckDB |
| Validation Agent | Validates data quality |
| Response Agent | Generates insights from data and reports |

---

## 📁 Project Structure

```
├── app.py                 # Streamlit UI
├── agents/                # Multi-agent system
│   ├── orchestrator.py    # LangGraph workflow
│   ├── query_agent.py     # NL to SQL
│   ├── extraction_agent.py
│   ├── validation_agent.py
│   └── response_agent.py
├── utils/                 # Utilities
│   ├── data_layer.py      # DuckDB integration
│   ├── memory.py          # Conversation memory
│   ├── evaluation.py      # Quality metrics
│   └── edge_cases.py      # Error handling
├── data/                  # Data files
├── Dockerfile             # Docker config
└── requirements.txt       # Dependencies
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | SQL query correctness |
| Faithfulness | Response grounded in data |
| Relevance | Answer addresses question |
| Completeness | Full answer provided |

---

## 🛠️ Tech Stack

- **LangChain + LangGraph** - Agent orchestration
- **Google Gemini** - LLM provider
- **DuckDB** - Embedded analytics database
- **Streamlit** - Web UI
- **Plotly** - Interactive charts

---

## 👤 Author

**Deepak Yadav**  
📧 dk.yadav125566@gmail.com  
🔗 [GitHub](https://github.com/deepak2233)

---

## 📄 License

MIT License
