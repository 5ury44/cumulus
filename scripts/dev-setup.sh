#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_INSTANCE_FILE="$PROJECT_ROOT/.dev-instance"
ENV_FILE="$SCRIPT_DIR/.env"

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 130 ] && [ -n "$INSTANCE_ID" ]; then
        echo "Instance $INSTANCE_ID may still be running"
    fi
}
trap cleanup EXIT

echo "🚀 Setting up Cumulus development environment"

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo "❌ jq required. Install with: brew install jq"
    exit 1
fi

# Helper function to ensure Docker is installed and configured on remote instance
ensure_docker_setup() {
    local ssh_host=$1
    local ssh_port=$2
    
    # Install Docker if needed
    if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$ssh_port" "root@$ssh_host" "command -v docker" &>/dev/null; then
        ssh -o StrictHostKeyChecking=no -p "$ssh_port" "root@$ssh_host" "curl -fsSL https://get.docker.com | sh && systemctl start docker && systemctl enable docker" >/dev/null
    fi
    
    # Configure Docker daemon
    ssh -o StrictHostKeyChecking=no -p "$ssh_port" "root@$ssh_host" << 'REMOTE_SCRIPT'
# Create Docker daemon configuration
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'DOCKER_EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
DOCKER_EOF

# Create systemd drop-in to expose Docker daemon on TCP port
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/tcp.conf << 'SYSTEMD_EOF'
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// -H tcp://0.0.0.0:23750
SYSTEMD_EOF

# Reload and restart Docker
systemctl daemon-reload
systemctl restart docker

# Wait for Docker to be ready
for i in {1..30}; do
    if docker info &>/dev/null; then
        break
    fi
    [ $i -eq 30 ] && echo "Docker startup timeout" && exit 1
    sleep 2
done
REMOTE_SCRIPT
    
    # Verify Docker is accessible
    for i in {1..15}; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$ssh_port" "root@$ssh_host" "docker info" &>/dev/null; then
            return 0
        fi
        [ $i -eq 15 ] && echo "❌ Docker verification timeout" && return 1
        sleep 2
    done
}

# Helper function to ensure SSH tunnel is running
ensure_ssh_tunnel() {
    local ssh_host=$1
    local ssh_port=$2
    
    # Kill existing tunnel and start new one
    if lsof -ti:23750 &>/dev/null; then
        pkill -f "ssh.*23750:localhost:23750" || true
        sleep 1
    fi
    
    ssh -f -N -L 23750:localhost:23750 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p "$ssh_port" "root@$ssh_host"
    
    # Verify tunnel
    for i in {1..10}; do
        if lsof -ti:23750 &>/dev/null; then
            return 0
        fi
        [ $i -eq 10 ] && echo "❌ Failed to establish SSH tunnel" && return 1
        sleep 1
    done
}

# Install vastai if needed
if ! command -v vastai &> /dev/null; then
    pip install vastai
fi

# Load API key from .env
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

if [ -z "$VASTAI_API_KEY" ]; then
    echo "❌ VASTAI_API_KEY not found in scripts/.env"
    exit 1
fi

vastai set api-key "$VASTAI_API_KEY"

# Verify API key works
echo "Verifying API key..."
USER_INFO=$(vastai show user 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ Invalid API key or connection failed:"
    echo "$USER_INFO"
    exit 1
fi

# Check if we have an existing instance
if [ -f "$DEV_INSTANCE_FILE" ]; then
    source "$DEV_INSTANCE_FILE"
    
    if [ -n "$INSTANCE_ID" ]; then
        echo "Found existing instance: $INSTANCE_ID"
        
        INSTANCE_INFO_RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>&1)
        if [ $? -ne 0 ]; then
            echo "❌ Failed to query instance $INSTANCE_ID:"
            echo "$INSTANCE_INFO_RAW"
            rm -f "$DEV_INSTANCE_FILE"
            exit 1
        fi
        
        if ! echo "$INSTANCE_INFO_RAW" | jq empty 2>/dev/null; then
            echo "❌ Invalid JSON response for instance $INSTANCE_ID:"
            echo "$INSTANCE_INFO_RAW"
            rm -f "$DEV_INSTANCE_FILE"
            exit 1
        fi
        
        INSTANCE_STATUS=$(echo "$INSTANCE_INFO_RAW" | jq -r '.actual_status // "not_found"')
        
        if [ "$INSTANCE_STATUS" = "not_found" ]; then
            echo "Instance not found, creating new one"
            rm -f "$DEV_INSTANCE_FILE"
        fi
        
        if [ "$INSTANCE_STATUS" = "running" ]; then
            
            # Get connection info if missing
            if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] || [ -z "$PUBLIC_IP" ]; then
                PUBLIC_IP=$(echo "$INSTANCE_INFO_RAW" | jq -r '.public_ipaddr // empty')
                
                # Get the correct SSH connection details using vastai ssh-url
                SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
                if [ $? -eq 0 ] && [ -n "$SSH_URL" ]; then
                    # Parse ssh://root@host:port format
                    SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f1)
                    SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f2)
                else
                    # Fallback to JSON parsing (may not work for direct connections)
                    SSH_HOST=$(echo "$INSTANCE_INFO_RAW" | jq -r '.public_ipaddr // empty')
                    SSH_PORT=$(echo "$INSTANCE_INFO_RAW" | jq -r '.ports."22/tcp"[0].HostPort // empty')
                fi
                
                SSH_CONNECTION="ssh://root@$SSH_HOST:$SSH_PORT"
                
                cat > "$DEV_INSTANCE_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
PUBLIC_IP=$PUBLIC_IP
SSH_CONNECTION=$SSH_CONNECTION
EOF
            fi
            
            # Test SSH and continue setup
            if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "echo ready" &> /dev/null 2>&1; then
                
                ensure_docker_setup "$SSH_HOST" "$SSH_PORT"
                ensure_ssh_tunnel "$SSH_HOST" "$SSH_PORT"
                
                if ! docker context inspect vast-orchestrator &>/dev/null; then
                    docker context create vast-orchestrator --docker "host=tcp://localhost:23750"
                fi
                
                docker context use vast-orchestrator
                
                # Test Docker connection
                for i in {1..20}; do
                    if docker info &>/dev/null; then
                        break
                    fi
                    [ $i -eq 20 ] && echo "❌ Docker connection failed" && exit 1
                    sleep 2
                done
                
                # Initialize Swarm
                if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
                    docker swarm init --advertise-addr "$PUBLIC_IP" 2>/dev/null || true
                fi
                
                # Get Swarm token
                if [ -z "$SWARM_TOKEN" ]; then
                    SWARM_TOKEN=$(docker swarm join-token worker -q 2>/dev/null || echo "")
                    if [ -n "$SWARM_TOKEN" ]; then
                        cat >> "$DEV_INSTANCE_FILE" << EOF
SWARM_TOKEN=$SWARM_TOKEN
SWARM_MANAGER=$PUBLIC_IP:2377
EOF
                    fi
                fi
                
                echo "✅ Environment ready!"
                echo "Instance: $INSTANCE_ID ($PUBLIC_IP)"
                echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"
                echo "Swarm: docker swarm join --token $SWARM_TOKEN $PUBLIC_IP:2377"
                exit 0
            else
                echo "SSH connection failed, will recreate instance"
            fi
        elif [ "$INSTANCE_STATUS" = "loading" ] || [ "$INSTANCE_STATUS" = "starting" ]; then
            echo "Instance still starting. Wait 3-5 minutes and run script again."
            exit 0
        else
            echo "Instance not running (status: $INSTANCE_STATUS)"
        fi
    fi
fi

# Search for new instance with VM support
echo "Searching for VM-capable instance..."

SEARCH_RESULT=$(vastai search offers "cpu_cores>=4 cpu_ram>=16 disk_space>=31 vms_enabled=true compute_cap<=900" --order "dph+" --raw 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ Failed to search for offers:"
    echo "$SEARCH_RESULT"
    exit 1
fi

# Debug: Check if result is valid JSON
if ! echo "$SEARCH_RESULT" | jq empty 2>/dev/null; then
    echo "❌ Invalid JSON response from vastai search:"
    echo "$SEARCH_RESULT"
    exit 1
fi

OFFER_ID=$(echo "$SEARCH_RESULT" | jq -r '.[0].id // empty' 2>/dev/null)
if [ -z "$OFFER_ID" ] || [ "$OFFER_ID" = "null" ]; then
    echo "❌ No suitable offers found"
    echo "Search result: $SEARCH_RESULT"
    exit 1
fi

OFFER_DETAILS=$(echo "$SEARCH_RESULT" | jq -r '.[0] | "CPU: \(.cpu_cores) cores, RAM: \(.cpu_ram)GB, Disk: \(.disk_space)GB, Price: $\(.dph_total)/hr"' 2>/dev/null)
echo "Selected: $OFFER_DETAILS"

CREATE_RESULT=$(vastai create instance "$OFFER_ID" \
  --image docker.io/vastai/kvm:ubuntu_cli_22.04-2025-05-16 \
  --disk 31 \
  --ssh \
  --direct \
  --env '-p 2377:2377 -p 50051:50051' \
  --raw 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Failed to create instance:"
    echo "$CREATE_RESULT"
    exit 1
fi

# Handle both JSON and Python dict formats
if echo "$CREATE_RESULT" | jq empty 2>/dev/null; then
    # Valid JSON format
    INSTANCE_ID=$(echo "$CREATE_RESULT" | jq -r '.new_contract // empty' 2>/dev/null)
elif echo "$CREATE_RESULT" | grep -q "new_contract"; then
    # Python dict format - extract the number
    INSTANCE_ID=$(echo "$CREATE_RESULT" | grep -o "'new_contract': [0-9]*" | grep -o "[0-9]*")
    if [ -z "$INSTANCE_ID" ]; then
        # Try alternative extraction
        INSTANCE_ID=$(echo "$CREATE_RESULT" | sed -n "s/.*'new_contract': \([0-9]*\).*/\1/p")
    fi
else
    echo "❌ Unexpected response format from vastai create:"
    echo "$CREATE_RESULT"
    exit 1
fi

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
    echo "❌ Failed to extract instance ID"
    echo "Create result: $CREATE_RESULT"
    exit 1
fi

cat > "$DEV_INSTANCE_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
EOF

echo "✅ Instance rented: $INSTANCE_ID"
echo "⏳ Waiting for instance to start (KVM takes 3-5 minutes)..."
for i in {1..30}; do
    INSTANCE_INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>&1)
    if [ $? -ne 0 ]; then
        echo "❌ Failed to query instance status:"
        echo "$INSTANCE_INFO"
        exit 1
    fi
    
    if ! echo "$INSTANCE_INFO" | jq empty 2>/dev/null; then
        echo "❌ Invalid JSON response for instance status:"
        echo "$INSTANCE_INFO"
        exit 1
    fi
    
    INSTANCE_STATUS=$(echo "$INSTANCE_INFO" | jq -r '.actual_status // "unknown"')
    
    case "$INSTANCE_STATUS" in
        "running")
            
            # Get connection info
            PUBLIC_IP=$(echo "$INSTANCE_INFO" | jq -r '.public_ipaddr // empty')
            
            # Get the correct SSH connection details using vastai ssh-url
            SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
            if [ $? -eq 0 ] && [ -n "$SSH_URL" ]; then
                # Parse ssh://root@host:port format
                SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f1)
                SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://root@||' | cut -d: -f2)
            else
                # Fallback to JSON parsing
                SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.public_ipaddr // empty')
                SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ports."22/tcp"[0].HostPort // empty')
            fi
            
            if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] || [ -z "$PUBLIC_IP" ]; then
                echo "❌ Missing connection information from instance"
                exit 1
            fi
            
            SSH_CONNECTION="ssh://root@$SSH_HOST:$SSH_PORT"
            
            # Update instance file with connection info
            cat > "$DEV_INSTANCE_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
PUBLIC_IP=$PUBLIC_IP
SSH_CONNECTION=$SSH_CONNECTION
EOF
            break
            ;;
        "loading"|"starting")
            echo "⏳ Status: $INSTANCE_STATUS (attempt $i/30)"
            ;;
        "exited"|"stopped")
            echo "❌ Instance failed to start (status: $INSTANCE_STATUS)"
            exit 1
            ;;
        *)
            echo "⏳ Status: $INSTANCE_STATUS (attempt $i/30)"
            ;;
    esac
    
    if [ $i -eq 30 ]; then
        echo ""
        echo "⏰ Instance still starting up (status: $INSTANCE_STATUS)"
        echo "   KVM instances take 3-5 minutes to boot"
        echo "   Wait a few minutes and run this script again"
        echo ""
        echo "💡 Check status: vastai show instance $INSTANCE_ID"
        exit 0
    fi
    sleep 2
done

# Wait for SSH
for i in {1..60}; do
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "echo ready" &> /dev/null 2>&1; then
        break
    fi
    [ $i -eq 60 ] && echo "❌ SSH timeout" && exit 1
    sleep 5
done

# Update packages
ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" "apt-get update && apt-get install -y curl jq netstat-nat" || true

# Install and configure Docker
ensure_docker_setup "$SSH_HOST" "$SSH_PORT"

# Setup SSH tunnel
ensure_ssh_tunnel "$SSH_HOST" "$SSH_PORT"

# Create Docker context
if ! docker context inspect vast-orchestrator &>/dev/null; then
    docker context create vast-orchestrator --docker "host=tcp://localhost:23750"
fi

docker context use vast-orchestrator

# Test Docker connection
for i in {1..30}; do
    if docker info &>/dev/null; then
        break
    fi
    [ $i -eq 30 ] && echo "❌ Docker connection failed" && exit 1
    sleep 2
done

# Initialize Swarm
docker swarm init --advertise-addr "$PUBLIC_IP" 2>/dev/null || true

# Get join token
SWARM_TOKEN=$(docker swarm join-token worker -q)

# Update instance file with swarm info
cat >> "$DEV_INSTANCE_FILE" << EOF
SWARM_TOKEN=$SWARM_TOKEN
SWARM_MANAGER=$PUBLIC_IP:2377
EOF

echo "✅ Environment ready!"
echo "Instance: $INSTANCE_ID ($PUBLIC_IP)"
echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo "Swarm: docker swarm join --token $SWARM_TOKEN $PUBLIC_IP:2377"
