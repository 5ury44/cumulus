# Cumulus Development Scripts

Scripts for managing a remote Vast.ai development environment with Docker Swarm.

## Architecture

The development setup uses **SSH port forwarding** to tunnel the Docker daemon from the remote Vast.ai instance to your local machine. This provides a stable, reusable Docker context that doesn't need to be recreated.

```
Local Machine                    Vast.ai Instance
┌─────────────────┐             ┌──────────────────┐
│ Docker CLI      │             │ Docker Daemon    │
│ (localhost)     │             │ (port 23750)     │
└────────┬────────┘             └────────▲─────────┘
         │                               │
         │  SSH Tunnel (port 23750)      │
         └───────────────────────────────┘
         
Docker Context: tcp://localhost:23750
```

## Quick Start

```bash
# 1. Set up environment (one-time)
./scripts/dev-setup.sh

# 2. Build and start orchestrator
./scripts/dev-start.sh

# 3. View logs
./scripts/dev-logs.sh

# 4. Destroy instance when done
./scripts/dev-stop.sh
```

## Scripts

### `dev-setup.sh`
One-time setup that:
- Rents a Vast.ai CPU instance
- Installs Docker and configures daemon
- Creates stable Docker context (`vast-orchestrator`)
- Starts SSH tunnel for Docker daemon
- Initializes Docker Swarm

**Docker Context**: Created once, reused across sessions
**SSH Tunnel**: Automatically started, can be managed with `dev-tunnel.sh`

### `dev-start.sh`
Start the orchestrator:
- Checks/starts SSH tunnel if needed
- Builds orchestrator image
- Runs orchestrator container
- Shows connection info

### `dev-stop.sh`
Clean up:
- Destroys Vast.ai instance
- Kills SSH tunnel
- Keeps Docker context for reuse

### `dev-tunnel.sh`
Manage SSH tunnel independently:
```bash
./scripts/dev-tunnel.sh status   # Check tunnel status
./scripts/dev-tunnel.sh start    # Start tunnel
./scripts/dev-tunnel.sh stop     # Stop tunnel
./scripts/dev-tunnel.sh restart  # Restart tunnel
```

### `dev-logs.sh`
View orchestrator logs:
```bash
./scripts/dev-logs.sh
```

## Configuration

### Environment Variables
Create `scripts/.env`:
```bash
VASTAI_API_KEY=your_api_key_here
```

### Ports

**Local (tunneled via SSH):**
- `23750` - Docker daemon (localhost only)

**Remote (exposed on Vast.ai):**
- `2377` - Docker Swarm manager
- `50051` - Orchestrator gRPC endpoint

## Benefits

✅ **Stable Docker Context**: Created once, reused forever
✅ **Fast Iteration**: No context recreation on each setup
✅ **Secure**: Docker daemon not exposed publicly
✅ **Simple**: Just manage SSH tunnel
✅ **Cheap**: ~$0.05-0.10/hour for CPU instance

## Troubleshooting

### SSH tunnel died
```bash
./scripts/dev-tunnel.sh restart
```

### Docker context not working
```bash
# Check tunnel
./scripts/dev-tunnel.sh status

# Test connection
docker context use vast-orchestrator
docker info
```

### Port 23750 already in use
```bash
# Kill existing tunnel
lsof -ti:23750 | xargs kill -9

# Restart
./scripts/dev-tunnel.sh start
```

### Instance info lost
If you lose `.dev-instance` file, you'll need to:
1. Find instance ID: `vastai show instances`
2. Destroy it: `vastai destroy instance <ID>`
3. Run setup again: `./scripts/dev-setup.sh`

## Cost Estimation

- **CPU Instance**: ~$0.05-0.10/hour
- **4-hour dev session**: ~$0.20-0.40
- **Full day**: ~$1.20-2.40

Remember to run `./scripts/dev-stop.sh` when done to avoid charges!
