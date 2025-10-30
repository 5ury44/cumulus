#!/bin/bash
set -e

# Check if .dev-instance exists
if [ ! -f .dev-instance ]; then
    echo "❌ Error: .dev-instance file not found"
    echo "Run ./scripts/dev-setup.sh first"
    exit 1
fi

# Load instance info
source .dev-instance

ACTION=${1:-status}

case "$ACTION" in
    start)
        if lsof -ti:23750 &>/dev/null; then
            echo "✅ SSH tunnel already running"
        else
            echo "🚇 Starting SSH tunnel..."
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
        fi
        ;;
    
    stop)
        echo "🚇 Stopping SSH tunnel..."
        lsof -ti:23750 | xargs kill -9 2>/dev/null || true
        echo "✅ SSH tunnel stopped"
        ;;
    
    restart)
        echo "🔄 Restarting SSH tunnel..."
        lsof -ti:23750 | xargs kill -9 2>/dev/null || true
        
        # Wait for port to be released
        for i in {1..10}; do
            if ! lsof -ti:23750 &>/dev/null; then
                break
            fi
            sleep 0.2
        done
        
        ssh -f -N -L 23750:localhost:23750 -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST"
        
        # Wait for tunnel to be ready
        for i in {1..10}; do
            if lsof -ti:23750 &>/dev/null; then
                echo "✅ SSH tunnel restarted"
                break
            fi
            [ $i -eq 10 ] && echo "❌ SSH tunnel failed to start" && exit 1
            sleep 0.5
        done
        ;;
    
    status)
        if lsof -ti:23750 &>/dev/null; then
            PID=$(lsof -ti:23750)
            echo "✅ SSH tunnel is running (PID: $PID)"
            echo "   Forwarding localhost:23750 → $SSH_HOST:23750"
        else
            echo "❌ SSH tunnel is not running"
            echo "   Run: ./scripts/dev-tunnel.sh start"
        fi
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Manage SSH tunnel for Docker daemon port forwarding"
        echo ""
        echo "Commands:"
        echo "  start   - Start SSH tunnel"
        echo "  stop    - Stop SSH tunnel"
        echo "  restart - Restart SSH tunnel"
        echo "  status  - Check tunnel status (default)"
        exit 1
        ;;
esac
