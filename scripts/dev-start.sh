#!/bin/bash
set -e

# Check if .dev-instance exists
if [ ! -f .dev-instance ]; then
    echo "❌ Error: .dev-instance file not found"
    echo "Run ./scripts/dev-setup.sh first to set up the development environment"
    exit 1
fi

# Load instance info
source .dev-instance

# Ensure SSH tunnel is running
echo "🚇 Checking SSH tunnel..."
if ! lsof -ti:23750 &>/dev/null; then
    echo "Starting SSH tunnel..."
    ssh -f -N -L 23750:localhost:23750 -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST"
    
    # Wait for tunnel to be ready
    for i in {1..10}; do
        if lsof -ti:23750 &>/dev/null; then
            echo "✅ SSH tunnel started"
            break
        fi
        [ $i -eq 10 ] && echo "❌ SSH tunnel failed to start" && exit 1
        sleep 0.5
    done
else
    echo "✅ SSH tunnel already running"
fi

# Switch to remote context
docker context use vast-orchestrator

# Verify Docker connection works
echo "🧪 Testing Docker connection..."
for i in {1..15}; do
    if docker info &>/dev/null; then
        echo "✅ Docker connection ready"
        break
    fi
    [ $i -eq 15 ] && echo "❌ Docker connection failed. Try: ./scripts/dev-tunnel.sh restart" && exit 1
    sleep 1
done

echo "🏗️  Building orchestrator..."
docker build -t cumulus-orchestrator ./orchestrator

echo "🚀 Starting orchestrator..."
# Remove old container if it exists
docker rm -f orchestrator 2>/dev/null || true

docker run -d \
  --name orchestrator \
  -p 50051:50051 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  cumulus-orchestrator

echo "✅ Orchestrator running on $PUBLIC_IP:50051"
echo "📋 View logs: ./scripts/dev-logs.sh"
echo "🛑 Stop orchestrator: ./scripts/dev-stop.sh"
