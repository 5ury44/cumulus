#!/bin/bash
set -e

# Check if .dev-instance exists
if [ ! -f .dev-instance ]; then
    echo "❌ Error: .dev-instance file not found"
    echo "No development environment to clean up"
    exit 0
fi

# Load instance info
source .dev-instance

echo "🗑️  Destroying Vast.ai instance..."
vastai destroy instance "$INSTANCE_ID"

# Kill SSH tunnel
echo "🚇 Closing SSH tunnel..."
lsof -ti:23750 | xargs kill -9 2>/dev/null || true

# Switch back to default context (keep vast-orchestrator context for reuse)
docker context use default

# Clean up files
rm -f .dev-instance

echo "✅ Development environment cleaned up"
echo "💡 Docker context 'vast-orchestrator' preserved for next setup"
