package chronos

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

// ChronosManager manages GPU partitions on worker nodes via SSH
type ChronosManager struct {
	sshClients map[string]*ssh.Client // Worker address -> SSH connection
	sshConfig  *ssh.ClientConfig
	mu         sync.RWMutex
	partitions map[string]*PartitionInfo // Partition ID -> partition info
}

// PartitionInfo tracks information about a Chronos partition
type PartitionInfo struct {
	ID        string
	WorkerID  string
	Address   string
	Device    int
	Memory    float64
	Duration  int
	JobID     string
	CreatedAt time.Time
	ExpiresAt time.Time
}

// SSHConfig contains SSH connection configuration
type SSHConfig struct {
	User       string
	KeyPath    string
	Timeout    time.Duration
	MaxRetries int
}

// NewChronosManager creates a new ChronosManager instance
func NewChronosManager(config *SSHConfig) (*ChronosManager, error) {
	// Load SSH private key
	key, err := loadPrivateKey(config.KeyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load SSH private key: %w", err)
	}

	sshConfig := &ssh.ClientConfig{
		User: config.User,
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(key),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(), // TODO: Use proper host key verification in production
		Timeout:         config.Timeout,
	}

	return &ChronosManager{
		sshClients: make(map[string]*ssh.Client),
		sshConfig:  sshConfig,
		partitions: make(map[string]*PartitionInfo),
	}, nil
}

// loadPrivateKey loads an SSH private key from file
func loadPrivateKey(keyPath string) (ssh.Signer, error) {
	// For now, we'll implement a basic key loader
	// In a real implementation, this would read from the file system
	// and handle different key formats (RSA, ECDSA, Ed25519)

	// This is a placeholder - in the actual implementation we would:
	// 1. Read the key file from keyPath
	// 2. Parse it using ssh.ParsePrivateKey()
	// 3. Return the signer

	return nil, fmt.Errorf("SSH key loading not implemented - placeholder for keyPath: %s", keyPath)
}

// Close closes all SSH connections
func (m *ChronosManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errors []string
	for addr, client := range m.sshClients {
		if err := client.Close(); err != nil {
			errors = append(errors, fmt.Sprintf("failed to close SSH connection to %s: %v", addr, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors closing SSH connections: %s", strings.Join(errors, "; "))
	}

	return nil
}

// getSSHClient gets or creates an SSH client for the given worker address
func (m *ChronosManager) getSSHClient(ctx context.Context, address string) (*ssh.Client, error) {
	m.mu.RLock()
	client, exists := m.sshClients[address]
	m.mu.RUnlock()

	if exists && client != nil {
		// Test the connection
		session, err := client.NewSession()
		if err == nil {
			session.Close()
			return client, nil
		}
		// Connection is dead, remove it
		m.mu.Lock()
		delete(m.sshClients, address)
		m.mu.Unlock()
	}

	// Create new connection
	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check after acquiring write lock
	if client, exists := m.sshClients[address]; exists && client != nil {
		return client, nil
	}

	// Create new SSH connection
	client, err := ssh.Dial("tcp", address+":22", m.sshConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to %s: %w", address, err)
	}

	m.sshClients[address] = client
	return client, nil
}

// executeCommand executes a command on the worker via SSH
func (m *ChronosManager) executeCommand(ctx context.Context, address, command string) (string, error) {
	client, err := m.getSSHClient(ctx, address)
	if err != nil {
		return "", err
	}

	session, err := client.NewSession()
	if err != nil {
		return "", fmt.Errorf("failed to create SSH session: %w", err)
	}
	defer session.Close()

	// Set up context cancellation
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			session.Signal(ssh.SIGTERM)
		case <-done:
		}
	}()

	output, err := session.CombinedOutput(command)
	close(done)

	if err != nil {
		return "", fmt.Errorf("command failed: %w, output: %s", err, string(output))
	}

	return string(output), nil
}

// CreatePartition creates a Chronos GPU partition on a worker
func (m *ChronosManager) CreatePartition(ctx context.Context, workerID, address string, device int, memory float64, duration int, jobID string) (string, error) {
	// Build chronos_cli command
	command := fmt.Sprintf("chronos_cli create %d %.2f %d", device, memory, duration)

	// Execute command on worker
	output, err := m.executeCommand(ctx, address, command)
	if err != nil {
		return "", fmt.Errorf("failed to create partition on worker %s: %w", workerID, err)
	}

	// Parse partition ID from output
	partitionID, err := parsePartitionID(output)
	if err != nil {
		return "", fmt.Errorf("failed to parse partition ID from output: %w", err)
	}

	// Store partition info
	partition := &PartitionInfo{
		ID:        partitionID,
		WorkerID:  workerID,
		Address:   address,
		Device:    device,
		Memory:    memory,
		Duration:  duration,
		JobID:     jobID,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(time.Duration(duration) * time.Second),
	}

	m.mu.Lock()
	m.partitions[partitionID] = partition
	m.mu.Unlock()

	return partitionID, nil
}

// ReleasePartition releases a Chronos partition
func (m *ChronosManager) ReleasePartition(ctx context.Context, partitionID string) error {
	m.mu.RLock()
	partition, exists := m.partitions[partitionID]
	m.mu.RUnlock()

	if !exists {
		return fmt.Errorf("partition %s not found", partitionID)
	}

	// Build chronos_cli command
	command := fmt.Sprintf("chronos_cli release %s", partitionID)

	// Execute command on worker
	_, err := m.executeCommand(ctx, partition.Address, command)
	if err != nil {
		return fmt.Errorf("failed to release partition %s on worker %s: %w", partitionID, partition.WorkerID, err)
	}

	// Remove partition from tracking
	m.mu.Lock()
	delete(m.partitions, partitionID)
	m.mu.Unlock()

	return nil
}

// GetPartition gets information about a partition
func (m *ChronosManager) GetPartition(partitionID string) (*PartitionInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	partition, exists := m.partitions[partitionID]
	if !exists {
		return nil, false
	}

	// Return a copy to avoid race conditions
	partitionCopy := *partition
	return &partitionCopy, true
}

// ListPartitions returns all active partitions
func (m *ChronosManager) ListPartitions() []*PartitionInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	partitions := make([]*PartitionInfo, 0, len(m.partitions))
	for _, partition := range m.partitions {
		// Return copies to avoid race conditions
		partitionCopy := *partition
		partitions = append(partitions, &partitionCopy)
	}

	return partitions
}

// CleanupExpiredPartitions removes expired partitions from tracking
func (m *ChronosManager) CleanupExpiredPartitions() {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	for id, partition := range m.partitions {
		if now.After(partition.ExpiresAt) {
			delete(m.partitions, id)
		}
	}
}

// GetWorkerPartitions returns all partitions for a specific worker
func (m *ChronosManager) GetWorkerPartitions(workerID string) []*PartitionInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var workerPartitions []*PartitionInfo
	for _, partition := range m.partitions {
		if partition.WorkerID == workerID {
			// Return a copy to avoid race conditions
			partitionCopy := *partition
			workerPartitions = append(workerPartitions, &partitionCopy)
		}
	}

	return workerPartitions
}

// parsePartitionID extracts the partition ID from chronos_cli output
func parsePartitionID(output string) (string, error) {
	// Expected output format: "Created partition: partition_12345"
	// or similar patterns

	// Try different regex patterns for partition ID extraction
	patterns := []string{
		`partition[_-]([a-zA-Z0-9]+)`,                       // partition_12345 or partition-12345
		`Created partition:?\s*([a-zA-Z0-9_-]+)`,            // Created partition: abc123
		`Partition ID:?\s*([a-zA-Z0-9_-]+)`,                 // Partition ID: abc123
		`Successfully created partition\s+([a-zA-Z0-9_-]+)`, // Successfully created partition abc123
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(output)
		if len(matches) > 1 {
			return matches[1], nil
		}
	}

	// If no pattern matches, try to extract the last word that looks like an ID
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) > 0 {
		lastLine := strings.TrimSpace(lines[len(lines)-1])
		words := strings.Fields(lastLine)
		if len(words) > 0 {
			lastWord := words[len(words)-1]
			// Check if it looks like a partition ID (alphanumeric, 6+ chars)
			if matched, _ := regexp.MatchString(`^[a-zA-Z0-9_-]{6,}$`, lastWord); matched {
				return lastWord, nil
			}
		}
	}

	return "", fmt.Errorf("could not parse partition ID from output: %s", output)
}

// TestConnection tests SSH connectivity to a worker
func (m *ChronosManager) TestConnection(ctx context.Context, address string) error {
	_, err := m.executeCommand(ctx, address, "echo 'connection test'")
	return err
}

// GetChronosStats gets Chronos statistics from a worker
func (m *ChronosManager) GetChronosStats(ctx context.Context, address string) (string, error) {
	return m.executeCommand(ctx, address, "chronos_cli stats")
}
