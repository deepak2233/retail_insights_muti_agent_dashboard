#!/bin/bash

# Retail Insights Assistant - Setup Script
# This script automates the complete setup process

echo "=================================================="
echo "🛍️  RETAIL INSIGHTS ASSISTANT - SETUP"
echo "=================================================="
echo ""

# Check Python version
echo "📋 Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3,9)' 2>/dev/null; then
    echo "   ❌ Error: Python 3.9+ required"
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Create virtual environment
echo "📦 Step 2: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Step 3: Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Step 4: Upgrading pip..."
pip install --upgrade pip -q
echo "   ✅ Pip upgraded"
echo ""

# Install requirements
echo "📚 Step 5: Installing dependencies..."
echo "   This may take 2-3 minutes..."
pip install -r requirements.txt -q
if [ $? -eq 0 ]; then
    echo "   ✅ Dependencies installed successfully"
else
    echo "   ❌ Error installing dependencies"
    exit 1
fi
echo ""

# Check for .env file
echo "⚙️  Step 6: Checking configuration..."
if [ ! -f ".env" ]; then
    echo "   Creating .env from template..."
    cp .env.example .env
    echo "   ⚠️  IMPORTANT: Edit .env and add your API key!"
    echo "   For OpenAI: OPENAI_API_KEY=your-openai-api-key-here"
    echo "   For Gemini: GOOGLE_API_KEY=your-google-api-key-here"
else
    echo "   ✅ .env file exists"
fi
echo ""

# Generate sample data
echo "📊 Step 7: Generating sample data..."
if [ ! -f "data/sales_data.csv" ]; then
    echo "   Generating 50,000 sample records..."
    python data/generate_data.py
    if [ $? -eq 0 ]; then
        echo "   ✅ Sample data generated"
    else
        echo "   ❌ Error generating data"
        exit 1
    fi
else
    echo "   ✅ Sample data already exists"
fi
echo ""

# Run tests
echo "🧪 Step 8: Running tests (optional)..."
read -p "   Run tests now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/ -v
fi
echo ""

# Success message
echo "=================================================="
echo "✅ SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Configure API Key (if not done yet):"
echo "   Edit .env file and add your OpenAI or Gemini API key"
echo ""
echo "2. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "3. Or run quick demo:"
echo "   python demo.py"
echo ""
echo "4. Access the app at:"
echo "   http://localhost:8501"
echo ""
echo "📚 Documentation:"
echo "   • README.md - Full documentation"
echo "   • QUICKSTART.md - 5-minute guide"
echo "   • docs/SETUP_GUIDE.md - Detailed setup"
echo ""
echo "🆘 Having issues?"
echo "   Check docs/SETUP_GUIDE.md for troubleshooting"
echo ""
echo "=================================================="
