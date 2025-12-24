#!/bin/bash

# API Key Setup Script for Retail Insights Assistant
# This script helps you set up your API key for the Streamlit app

echo "============================================================"
echo "API KEY SETUP - Retail Insights Assistant"
echo "============================================================"
echo ""

# Check which API provider to use
echo "Which API provider do you want to use?"
echo "1) OpenAI (GPT-4)"
echo "2) Google (Gemini)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    read -p "Enter your OpenAI API key (starts with sk-): " api_key
    
    if [ -z "$api_key" ]; then
        echo "❌ Error: API key cannot be empty"
        exit 1
    fi
    
    # Set the environment variable
    export OPENAI_API_KEY="$api_key"
    
    # Add to .bashrc for persistence
    if ! grep -q "export OPENAI_API_KEY=" ~/.bashrc; then
        echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.bashrc
    else
        sed -i "s|export OPENAI_API_KEY=.*|export OPENAI_API_KEY=\"$api_key\"|" ~/.bashrc
    fi
    
    echo ""
    echo "✅ OpenAI API key set successfully!"
    echo "   Key: ${api_key:0:7}...${api_key: -4}"
    
elif [ "$choice" = "2" ]; then
    echo ""
    read -p "Enter your Google API key: " api_key
    
    if [ -z "$api_key" ]; then
        echo "❌ Error: API key cannot be empty"
        exit 1
    fi
    
    # Set the environment variable
    export GOOGLE_API_KEY="$api_key"
    
    # Add to .bashrc for persistence
    if ! grep -q "export GOOGLE_API_KEY=" ~/.bashrc; then
        echo "export GOOGLE_API_KEY=\"$api_key\"" >> ~/.bashrc
    else
        sed -i "s|export GOOGLE_API_KEY=.*|export GOOGLE_API_KEY=\"$api_key\"|" ~/.bashrc
    fi
    
    echo ""
    echo "✅ Google API key set successfully!"
    echo "   Key: ${api_key:0:7}...${api_key: -4}"
    
else
    echo "❌ Invalid choice. Please run again and select 1 or 2."
    exit 1
fi

echo ""
echo "============================================================"
echo "🚀 RESTARTING STREAMLIT APP"
echo "============================================================"

# Kill existing streamlit process
echo "Stopping existing Streamlit process..."
pkill -f streamlit
sleep 2

# Start Streamlit with the new API key
echo "Starting Streamlit with API key..."
cd /root/blend/retail-insights-assistant
nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &

sleep 5

# Check if it started
if ps aux | grep -v grep | grep streamlit > /dev/null; then
    echo "✅ Streamlit app started successfully!"
    echo ""
    echo "============================================================"
    echo "📊 ACCESS YOUR APP"
    echo "============================================================"
    echo ""
    echo "   Local URL:    http://localhost:8501"
    echo "   Network URL:  http://$(hostname -I | awk '{print $1}'):8501"
    echo ""
    echo "   Log file:     /root/blend/retail-insights-assistant/streamlit.log"
    echo ""
    echo "============================================================"
    echo "💡 TIP: You can now ask questions like:"
    echo "   - What were our total sales?"
    echo "   - Which state had the highest revenue?"
    echo "   - Show me top product categories"
    echo "============================================================"
else
    echo "❌ Failed to start Streamlit. Check the log file:"
    echo "   tail -f /root/blend/retail-insights-assistant/streamlit.log"
fi

echo ""
