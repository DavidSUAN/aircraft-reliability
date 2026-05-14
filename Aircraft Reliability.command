#!/bin/bash
cd "$(dirname "$0")"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start server
echo "Starting server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &

# Wait for server to be ready
sleep 2

# Open browser
open http://localhost:8000

# Keep terminal open
echo ""
echo "Server running at http://localhost:8000"
echo "Press Ctrl+C to stop"
read -p ""
