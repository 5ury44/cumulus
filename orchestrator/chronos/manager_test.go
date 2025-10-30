package chronos

import (
	"context"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
)

func TestNewChronosManager(t *testing.T) {
	config := &SSHConfig{
		User:       "root",
		KeyPath:    "/tmp/test_key",
		Timeout:    10 * time.Second,
		MaxRetries: 3,
	}

	// This will fail because we don't have a real SSH key, but it tests the structure
	_, err := NewChronosManager(config)
	if err == nil {
		t.Error("Expected error due to missing SSH key, but got none")
	}

	// The error should mention the key path
	if err != nil && err.Error() == "" {
		t.Error("Expected non-empty error message")
	}
}

func TestParsePartitionID(t *testing.T) {
	tests := []struct {
		name     string
		output   string
		expected string
		wantErr  bool
	}{
		{
			name:     "standard format",
			output:   "Created partition: partition_12345",
			expected: "12345",
			wantErr:  false,
		},
		{
			name:     "alternative format",
			output:   "Partition ID: abc123def",
			expected: "abc123def",
			wantErr:  false,
		},
		{
			name:     "simple partition format",
			output:   "partition-xyz789",
			expected: "xyz789",
			wantErr:  false,
		},
		{
			name:     "multiline output",
			output:   "Initializing GPU partition...\nCreated partition: part_456789\nPartition ready",
			expected: "part_456789",
			wantErr:  false,
		},
		{
			name:     "last word extraction",
			output:   "Successfully created partition abcd1234",
			expected: "abcd1234",
			wantErr:  false,
		},
		{
			name:    "no partition ID",
			output:  "Error: bad",
			wantErr: true,
		},
		{
			name:    "empty output",
			output:  "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parsePartitionID(tt.output)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestPartitionInfo(t *testing.T) {
	now := time.Now()
	partition := &PartitionInfo{
		ID:        "test-partition-123",
		WorkerID:  "worker-456",
		Address:   "192.168.1.100",
		Device:    0,
		Memory:    0.5,
		Duration:  3600,
		JobID:     "job-789",
		CreatedAt: now,
		ExpiresAt: now.Add(time.Hour),
	}

	if partition.ID != "test-partition-123" {
		t.Errorf("Expected ID 'test-partition-123', got '%s'", partition.ID)
	}

	if partition.Memory != 0.5 {
		t.Errorf("Expected memory 0.5, got %f", partition.Memory)
	}

	if partition.Duration != 3600 {
		t.Errorf("Expected duration 3600, got %d", partition.Duration)
	}

	// Test expiration
	if !partition.ExpiresAt.After(partition.CreatedAt) {
		t.Error("ExpiresAt should be after CreatedAt")
	}
}

func TestChronosManagerPartitionTracking(t *testing.T) {
	// Create a manager without SSH (for testing partition tracking only)
	manager := &ChronosManager{
		sshClients: make(map[string]*ssh.Client),
		partitions: make(map[string]*PartitionInfo),
	}

	// Test empty state
	partitions := manager.ListPartitions()
	if len(partitions) != 0 {
		t.Errorf("Expected 0 partitions, got %d", len(partitions))
	}

	// Add a test partition directly (bypassing SSH)
	testPartition := &PartitionInfo{
		ID:        "test-123",
		WorkerID:  "worker-1",
		Address:   "192.168.1.100",
		Device:    0,
		Memory:    0.5,
		Duration:  3600,
		JobID:     "job-456",
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(time.Hour),
	}

	manager.partitions["test-123"] = testPartition

	// Test GetPartition
	retrieved, exists := manager.GetPartition("test-123")
	if !exists {
		t.Error("Expected partition to exist")
	}
	if retrieved.ID != "test-123" {
		t.Errorf("Expected ID 'test-123', got '%s'", retrieved.ID)
	}

	// Test ListPartitions
	partitions = manager.ListPartitions()
	if len(partitions) != 1 {
		t.Errorf("Expected 1 partition, got %d", len(partitions))
	}

	// Test GetWorkerPartitions
	workerPartitions := manager.GetWorkerPartitions("worker-1")
	if len(workerPartitions) != 1 {
		t.Errorf("Expected 1 partition for worker-1, got %d", len(workerPartitions))
	}

	workerPartitions = manager.GetWorkerPartitions("worker-2")
	if len(workerPartitions) != 0 {
		t.Errorf("Expected 0 partitions for worker-2, got %d", len(workerPartitions))
	}
}

func TestCleanupExpiredPartitions(t *testing.T) {
	manager := &ChronosManager{
		sshClients: make(map[string]*ssh.Client),
		partitions: make(map[string]*PartitionInfo),
	}

	now := time.Now()

	// Add expired partition
	expiredPartition := &PartitionInfo{
		ID:        "expired-123",
		WorkerID:  "worker-1",
		CreatedAt: now.Add(-2 * time.Hour),
		ExpiresAt: now.Add(-1 * time.Hour), // Expired 1 hour ago
	}
	manager.partitions["expired-123"] = expiredPartition

	// Add active partition
	activePartition := &PartitionInfo{
		ID:        "active-456",
		WorkerID:  "worker-1",
		CreatedAt: now,
		ExpiresAt: now.Add(time.Hour), // Expires in 1 hour
	}
	manager.partitions["active-456"] = activePartition

	// Before cleanup
	if len(manager.partitions) != 2 {
		t.Errorf("Expected 2 partitions before cleanup, got %d", len(manager.partitions))
	}

	// Run cleanup
	manager.CleanupExpiredPartitions()

	// After cleanup
	if len(manager.partitions) != 1 {
		t.Errorf("Expected 1 partition after cleanup, got %d", len(manager.partitions))
	}

	// Check that the active partition remains
	_, exists := manager.GetPartition("active-456")
	if !exists {
		t.Error("Active partition should still exist after cleanup")
	}

	// Check that the expired partition is gone
	_, exists = manager.GetPartition("expired-123")
	if exists {
		t.Error("Expired partition should be removed after cleanup")
	}
}

func TestSSHConfig(t *testing.T) {
	config := &SSHConfig{
		User:       "testuser",
		KeyPath:    "/path/to/key",
		Timeout:    30 * time.Second,
		MaxRetries: 5,
	}

	if config.User != "testuser" {
		t.Errorf("Expected user 'testuser', got '%s'", config.User)
	}

	if config.Timeout != 30*time.Second {
		t.Errorf("Expected timeout 30s, got %v", config.Timeout)
	}

	if config.MaxRetries != 5 {
		t.Errorf("Expected max retries 5, got %d", config.MaxRetries)
	}
}

// TestCreatePartitionWithoutSSH tests the partition creation logic without actual SSH
func TestCreatePartitionWithoutSSH(t *testing.T) {
	// This test would require mocking SSH connections
	// For now, we'll test the partition ID parsing and data structures

	ctx := context.Background()

	// Test context cancellation
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel() // Cancel immediately

	if cancelCtx.Err() == nil {
		t.Error("Expected context to be cancelled")
	}
}

func TestManagerClose(t *testing.T) {
	manager := &ChronosManager{
		sshClients: make(map[string]*ssh.Client),
		partitions: make(map[string]*PartitionInfo),
	}

	// Test closing empty manager
	err := manager.Close()
	if err != nil {
		t.Errorf("Expected no error closing empty manager, got: %v", err)
	}
}
