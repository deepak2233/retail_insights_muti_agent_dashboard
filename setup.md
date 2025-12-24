# 🚀 Retail Insights AI - Team Handover Guide

## Quick Start (3 Options)

### Option 1: Docker (Recommended for Production)
```bash
# 1. Clone the repository
git clone https://github.com/deepak2233/retaisl_insights_agent.git
cd retaisl_insights_agent

# 2. Create .env file with your API key
cat > .env << EOF
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_MODEL=gemini-2.5-flash
LLM_PROVIDER=google
EOF

# 3. Build and run with Docker Compose
docker-compose up --build

# App will be available at: http://localhost:8501
```

### Option 2: Local Python Environment
```bash
# 1. Clone the repository
git clone https://github.com/deepak2233/retaisl_insights_agent.git
cd retaisl_insights_agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cat > .env << EOF
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_MODEL=gemini-2.5-flash
LLM_PROVIDER=google
EOF

# 5. Run the app
streamlit run app.py

# App will be available at: http://localhost:8501
```

### Option 3: Streamlit Cloud (Free Hosting)
1. Fork/Push repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set secrets in Streamlit Cloud dashboard:
   ```toml
   GOOGLE_API_KEY = "your-google-api-key"
   GEMINI_MODEL = "gemini-2.5-flash"
   LLM_PROVIDER = "google"
   ```
5. Deploy!

---

## 📋 Prerequisites

### System Requirements
- Python 3.9+ (or Docker)
- 4GB RAM minimum
- 2GB disk space

### API Key Setup
Get a free Google AI API key:
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)

---

## 🐳 Docker Commands Reference

### Build Image
```bash
docker build -t retail-insights-ai .
```

### Run Container
```bash
docker run -d \
  --name retail-insights \
  -p 8501:8501 \
  -e GOOGLE_API_KEY=your-key-here \
  -e GEMINI_MODEL=gemini-2.5-flash \
  -e LLM_PROVIDER=google \
  retail-insights-ai
```

### Docker Compose
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up --build -d
```

### Check Container Status
```bash
docker ps
docker logs retail-insights-assistant
```

---

## 📁 Project Structure

```
retaisl_insights_agent/
├── app.py                 # Main Streamlit UI
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker orchestration
├── .env.example           # Environment template
│
├── agents/                # Multi-agent system
│   ├── orchestrator.py    # LangGraph workflow
│   ├── query_agent.py     # NL to SQL conversion
│   ├── extraction_agent.py # Data extraction
│   ├── validation_agent.py # Result validation
│   └── response_agent.py  # Response generation
│
├── utils/                 # Utilities
│   ├── data_layer.py      # DuckDB integration
│   ├── memory.py          # Conversation memory
│   ├── evaluation.py      # Quality metrics
│   ├── edge_cases.py      # Error handling
│   └── hallucination_prevention.py
│
├── data/                  # Data files
│   └── generate_data.py   # Data processing script
│
└── docs/                  # Documentation
```

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (google/openai) | google |
| `GOOGLE_API_KEY` | Google AI API key | Required |
| `GEMINI_MODEL` | Gemini model name | gemini-2.5-flash |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | Optional |
| `OPENAI_MODEL` | OpenAI model name | gpt-4o-mini |

### Using OpenAI Instead of Google
```bash
# .env file
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini
```

---

## 📊 Data Setup

The app uses Amazon India e-commerce sales data. To use your own data:

1. Place your CSV file in `data/` folder
2. Update `data_layer.py` to match your schema
3. Modify the SQL schema context in `data_layer.py`

### Expected Data Schema
```
order_id, date, status, category, state, city, amount, quantity, is_b2b
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific component
python test_data_only.py
python test_complete_system.py
```

---

## 🔒 Security Notes

1. **Never commit `.env` file** - It contains API keys
2. **Use secrets management** in production (AWS Secrets Manager, etc.)
3. **Rotate API keys** regularly
4. **Set usage limits** on your API keys in Google/OpenAI dashboard

---

## 🆘 Troubleshooting

### Common Issues

**1. "Module not found" error**
```bash
pip install -r requirements.txt
```

**2. DuckDB lock error**
```bash
rm -f data/*.duckdb*
```

**3. API rate limit**
- Wait a few minutes and try again
- Consider upgrading to paid API tier

**4. Docker permission denied**
```bash
sudo chmod 666 /var/run/docker.sock
# Or run with sudo
```

**5. Port 8501 already in use**
```bash
# Find and kill the process
lsof -i :8501
kill -9 <PID>
# Or use different port
streamlit run app.py --server.port 8502
```

---

## 📞 Support

- **Repository**: https://github.com/deepak2233/retaisl_insights_agent
- **Issues**: Create a GitHub issue for bugs/questions

---

## 📝 Quick Reference Card

```bash
# Clone
git clone https://github.com/deepak2233/retaisl_insights_agent.git

# Setup
cd retaisl_insights_agent
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your-key" > .env

# Run
streamlit run app.py

# Docker
docker-compose up --build
```

**App URL**: http://localhost:8501

---

*Last Updated: December 2024*
