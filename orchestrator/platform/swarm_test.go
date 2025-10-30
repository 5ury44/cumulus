package platform

import (
	"context"
	"testing"
	"time"
)

func TestNewSwarmClient(t *testing.T) {
	client, err := NewSwarmClient()
	if err != nil {
		t.Fatalf("Failed to create SwarmClient: %v", err)
	}
	defer client.Close()

	// Test basic connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Check if we can connect to Docker
	isManager, err := client.IsSwarmManager(ctx)
	if err != nil {
		t.Logf("Docker connection test failed (this is expected if Docker is not running): %v", err)
		return
	}

	t.Logf("Docker connection successful. Is Swarm manager: %v", isManager)
}

func TestListWorkers(t *testing.T) {
	client, err := NewSwarmClient()
	if err != nil {
		t.Fatalf("Failed to create SwarmClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	workers, err := client.ListWorkers(ctx)
	if err != nil {
		t.Logf("ListWorkers failed (this is expected if Swarm is not initialized): %v", err)
		return
	}

	t.Logf("Found %d workers", len(workers))
	for _, worker := range workers {
		t.Logf("Worker: ID=%s, Hostname=%s, Status=%s, GPUs=%d",
			worker.ID, worker.Hostname, worker.Status, len(worker.GPUDevices))
	}
}

func TestParseGPUDevices(t *testing.T) {
	tests := []struct {
		name     string
		labels   map[string]string
		expected int
	}{
		{
			name: "single GPU",
			labels: map[string]string{
				"gpu.count":    "1",
				"gpu.0.name":   "NVIDIA A100",
				"gpu.0.memory": "40960",
			},
			expected: 1,
		},
		{
			name: "dual GPU",
			labels: map[string]string{
				"gpu.count":    "2",
				"gpu.0.name":   "NVIDIA A100",
				"gpu.0.memory": "40960",
				"gpu.1.name":   "NVIDIA A100",
				"gpu.1.memory": "40960",
			},
			expected: 2,
		},
		{
			name:     "no GPU labels",
			labels:   map[string]string{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			devices, err := parseGPUDevices(tt.labels)
			if err != nil {
				t.Fatalf("parseGPUDevices failed: %v", err)
			}

			if len(devices) != tt.expected {
				t.Errorf("Expected %d devices, got %d", tt.expected, len(devices))
			}

			for i, device := range devices {
				if device.Index != i {
					t.Errorf("Device %d has wrong index: %d", i, device.Index)
				}
				if device.TotalCapacity != 1.0 {
					t.Errorf("Device %d has wrong total capacity: %f", i, device.TotalCapacity)
				}
			}
		})
	}
}

func TestHasGPULabels(t *testing.T) {
	tests := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "has gpu.count",
			labels: map[string]string{
				"gpu.count": "1",
			},
			expected: true,
		},
		{
			name: "has chronos.installed",
			labels: map[string]string{
				"chronos.installed": "true",
			},
			expected: true,
		},
		{
			name: "no GPU labels",
			labels: map[string]string{
				"some.other.label": "value",
			},
			expected: false,
		},
		{
			name:     "nil labels",
			labels:   nil,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasGPULabels(tt.labels)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestCreateService(t *testing.T) {
	client, err := NewSwarmClient()
	if err != nil {
		t.Fatalf("Failed to create SwarmClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test service spec creation (won't actually create since Swarm isn't running)
	spec := &ServiceSpec{
		Image:   "cumulus-job:latest",
		Command: []string{"python", "main.py"},
		Environment: []string{
			"JOB_ID=test-job-123",
			"CHRONOS_PARTITION_ID=partition-456",
		},
		NodeID: "test-node-id",
		Mounts: []Mount{
			{
				Source: "/tmp/job-code",
				Target: "/job",
				Type:   "bind",
			},
		},
	}

	_, err = client.CreateService(ctx, spec)
	if err != nil {
		t.Logf("CreateService failed (expected if Swarm not running): %v", err)
		return
	}

	t.Log("Service creation succeeded (unexpected in test environment)")
}

func TestServiceSpec(t *testing.T) {
	// Test ServiceSpec struct creation
	spec := &ServiceSpec{
		Image:   "cumulus-job:latest",
		Command: []string{"python", "main.py"},
		Environment: []string{
			"JOB_ID=test-job-123",
			"GPU_DEVICE=0",
		},
		NodeID: "node-123",
		Mounts: []Mount{
			{
				Source: "/host/path",
				Target: "/container/path",
				Type:   "bind",
			},
		},
	}

	if spec.Image != "cumulus-job:latest" {
		t.Errorf("Expected image 'cumulus-job:latest', got '%s'", spec.Image)
	}

	if len(spec.Command) != 2 {
		t.Errorf("Expected 2 command args, got %d", len(spec.Command))
	}

	if len(spec.Environment) != 2 {
		t.Errorf("Expected 2 environment variables, got %d", len(spec.Environment))
	}

	if len(spec.Mounts) != 1 {
		t.Errorf("Expected 1 mount, got %d", len(spec.Mounts))
	}
}

func TestGetServiceStatus(t *testing.T) {
	client, err := NewSwarmClient()
	if err != nil {
		t.Fatalf("Failed to create SwarmClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test getting status of non-existent service
	_, err = client.GetServiceStatus(ctx, "non-existent-service")
	if err != nil {
		t.Logf("GetServiceStatus failed (expected if Swarm not running): %v", err)
		return
	}

	t.Log("GetServiceStatus succeeded (unexpected in test environment)")
}

func TestGetServiceLogs(t *testing.T) {
	client, err := NewSwarmClient()
	if err != nil {
		t.Fatalf("Failed to create SwarmClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test getting logs of non-existent service
	_, err = client.GetServiceLogs(ctx, "non-existent-service")
	if err != nil {
		t.Logf("GetServiceLogs failed (expected if Swarm not running): %v", err)
		return
	}

	t.Log("GetServiceLogs succeeded (unexpected in test environment)")
}

func TestRemoveService(t *testing.T) {
	client, err := NewSwarmClient()
	if err != nil {
		t.Fatalf("Failed to create SwarmClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test removing non-existent service
	err = client.RemoveService(ctx, "non-existent-service")
	if err != nil {
		t.Logf("RemoveService failed (expected if Swarm not running): %v", err)
		return
	}

	t.Log("RemoveService succeeded (unexpected in test environment)")
}

func TestServiceStatus(t *testing.T) {
	// Test ServiceStatus struct
	status := &ServiceStatus{
		ID:          "service-123",
		Name:        "cumulus-job-test",
		State:       "running",
		NodeID:      "node-456",
		ContainerID: "container-789",
		CreatedAt:   time.Now().Unix(),
		UpdatedAt:   time.Now().Unix(),
		Error:       "",
	}

	if status.ID != "service-123" {
		t.Errorf("Expected ID 'service-123', got '%s'", status.ID)
	}

	if status.State != "running" {
		t.Errorf("Expected state 'running', got '%s'", status.State)
	}

	if status.Error != "" {
		t.Errorf("Expected no error, got '%s'", status.Error)
	}
}
