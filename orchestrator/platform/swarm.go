package platform

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/client"
)

// SwarmClient wraps the Docker API client for Swarm operations
type SwarmClient struct {
	cli *client.Client
}

// WorkerInfo represents information about a GPU worker node
type WorkerInfo struct {
	ID         string
	Hostname   string
	Address    string
	GPUDevices []GPUDevice
	Status     string // "ready", "down", "draining"
}

// GPUDevice represents a single GPU on a worker
type GPUDevice struct {
	Index            int
	Name             string
	MemoryMB         int
	TotalCapacity    float64 // Always 1.0 per physical GPU
	AllocatedAmount  float64 // Sum of all active partitions
	ActivePartitions []PartitionInfo
}

// PartitionInfo represents an active Chronos partition
type PartitionInfo struct {
	PartitionID string
	JobID       string
	Amount      float64
	CreatedAt   int64
	ExpiresAt   int64
}

// ServiceSpec defines a container service to be created
type ServiceSpec struct {
	Image       string
	Command     []string
	Environment []string
	NodeID      string // Target Swarm node
	Mounts      []Mount
}

// Mount represents a volume mount
type Mount struct {
	Source string
	Target string
	Type   string // "bind", "volume", "tmpfs"
}

// NewSwarmClient creates a new SwarmClient instance
func NewSwarmClient() (*SwarmClient, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	return &SwarmClient{
		cli: cli,
	}, nil
}

// Close closes the Docker client connection
func (s *SwarmClient) Close() error {
	return s.cli.Close()
}

// ListWorkers lists all Swarm nodes with GPU resources
func (s *SwarmClient) ListWorkers(ctx context.Context) ([]*WorkerInfo, error) {
	nodes, err := s.cli.NodeList(ctx, types.NodeListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list Swarm nodes: %w", err)
	}

	var workers []*WorkerInfo
	for _, node := range nodes {
		// Skip manager-only nodes
		if node.Spec.Role == swarm.NodeRoleManager && node.Spec.Availability != swarm.NodeAvailabilityActive {
			continue
		}

		// Check if node has GPU labels
		if !hasGPULabels(node.Spec.Labels) {
			continue
		}

		worker := &WorkerInfo{
			ID:       node.ID,
			Hostname: node.Description.Hostname,
			Status:   string(node.Status.State),
		}

		// Extract SSH address from labels
		if addr, ok := node.Spec.Labels["ssh.address"]; ok {
			worker.Address = addr
		}

		// Parse GPU devices from labels
		gpuDevices, err := parseGPUDevices(node.Spec.Labels)
		if err != nil {
			// Log warning but continue
			fmt.Printf("Warning: failed to parse GPU devices for node %s: %v\n", node.ID, err)
			continue
		}
		worker.GPUDevices = gpuDevices

		workers = append(workers, worker)
	}

	return workers, nil
}

// GetNodeInfo gets detailed information about a specific node
func (s *SwarmClient) GetNodeInfo(ctx context.Context, nodeID string) (*WorkerInfo, error) {
	node, _, err := s.cli.NodeInspectWithRaw(ctx, nodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect node %s: %w", nodeID, err)
	}

	worker := &WorkerInfo{
		ID:       node.ID,
		Hostname: node.Description.Hostname,
		Status:   string(node.Status.State),
	}

	// Extract SSH address from labels
	if addr, ok := node.Spec.Labels["ssh.address"]; ok {
		worker.Address = addr
	}

	// Parse GPU devices from labels
	gpuDevices, err := parseGPUDevices(node.Spec.Labels)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GPU devices for node %s: %w", nodeID, err)
	}
	worker.GPUDevices = gpuDevices

	return worker, nil
}

// hasGPULabels checks if a node has GPU-related labels
func hasGPULabels(labels map[string]string) bool {
	if labels == nil {
		return false
	}

	// Check for gpu.count label
	if _, ok := labels["gpu.count"]; ok {
		return true
	}

	// Check for chronos.installed label
	if installed, ok := labels["chronos.installed"]; ok && installed == "true" {
		return true
	}

	return false
}

// parseGPUDevices parses GPU device information from node labels
func parseGPUDevices(labels map[string]string) ([]GPUDevice, error) {
	if labels == nil {
		return nil, nil
	}

	// Get GPU count
	gpuCountStr, ok := labels["gpu.count"]
	if !ok {
		return nil, nil
	}

	gpuCount, err := strconv.Atoi(gpuCountStr)
	if err != nil {
		return nil, fmt.Errorf("invalid gpu.count value: %s", gpuCountStr)
	}

	devices := make([]GPUDevice, gpuCount)
	for i := 0; i < gpuCount; i++ {
		device := GPUDevice{
			Index:           i,
			TotalCapacity:   1.0, // Always 1.0 per physical GPU
			AllocatedAmount: 0.0, // Will be updated by partition tracking
		}

		// Get GPU name
		nameKey := fmt.Sprintf("gpu.%d.name", i)
		if name, ok := labels[nameKey]; ok {
			device.Name = name
		}

		// Get GPU memory
		memoryKey := fmt.Sprintf("gpu.%d.memory", i)
		if memoryStr, ok := labels[memoryKey]; ok {
			if memory, err := strconv.Atoi(memoryStr); err == nil {
				device.MemoryMB = memory
			}
		}

		devices[i] = device
	}

	return devices, nil
}

// IsSwarmManager checks if the current Docker daemon is a Swarm manager
func (s *SwarmClient) IsSwarmManager(ctx context.Context) (bool, error) {
	info, err := s.cli.Info(ctx)
	if err != nil {
		return false, fmt.Errorf("failed to get Docker info: %w", err)
	}

	return info.Swarm.ControlAvailable, nil
}

// GetSwarmInfo returns information about the Swarm cluster
func (s *SwarmClient) GetSwarmInfo(ctx context.Context) (*swarm.Swarm, error) {
	info, err := s.cli.Info(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get Docker info: %w", err)
	}

	if info.Swarm.NodeID == "" {
		return nil, fmt.Errorf("Docker daemon is not part of a Swarm cluster")
	}

	swarmInfo, err := s.cli.SwarmInspect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect Swarm: %w", err)
	}

	return &swarmInfo, nil
}

// CreateService creates a Docker service (container) on Swarm
func (s *SwarmClient) CreateService(ctx context.Context, spec *ServiceSpec) (string, error) {
	// Convert our ServiceSpec to Docker's ServiceSpec
	serviceSpec := swarm.ServiceSpec{
		Annotations: swarm.Annotations{
			Name: fmt.Sprintf("cumulus-job-%d", time.Now().Unix()),
		},
		TaskTemplate: swarm.TaskSpec{
			ContainerSpec: &swarm.ContainerSpec{
				Image:   spec.Image,
				Command: spec.Command,
				Env:     spec.Environment,
			},
			RestartPolicy: &swarm.RestartPolicy{
				Condition: swarm.RestartPolicyConditionNone,
			},
			Placement: &swarm.Placement{
				Constraints: []string{
					fmt.Sprintf("node.id == %s", spec.NodeID),
				},
			},
		},
	}

	// Add mounts if specified
	if len(spec.Mounts) > 0 {
		mounts := make([]mount.Mount, len(spec.Mounts))
		for i, m := range spec.Mounts {
			mounts[i] = mount.Mount{
				Source: m.Source,
				Target: m.Target,
				Type:   mount.Type(m.Type),
			}
		}
		serviceSpec.TaskTemplate.ContainerSpec.Mounts = mounts
	}

	// Create the service
	response, err := s.cli.ServiceCreate(ctx, serviceSpec, types.ServiceCreateOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to create service: %w", err)
	}

	return response.ID, nil
}

// GetServiceStatus checks the status of a service
func (s *SwarmClient) GetServiceStatus(ctx context.Context, serviceID string) (*ServiceStatus, error) {
	service, _, err := s.cli.ServiceInspectWithRaw(ctx, serviceID, types.ServiceInspectOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to inspect service %s: %w", serviceID, err)
	}

	// Get tasks for this service
	taskFilters := filters.NewArgs()
	taskFilters.Add("service", serviceID)

	tasks, err := s.cli.TaskList(ctx, types.TaskListOptions{
		Filters: taskFilters,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list tasks for service %s: %w", serviceID, err)
	}

	status := &ServiceStatus{
		ID:    service.ID,
		Name:  service.Spec.Name,
		State: "unknown",
	}

	// Determine overall service state from tasks
	if len(tasks) > 0 {
		// Use the most recent task
		task := tasks[0]
		for _, t := range tasks {
			if t.CreatedAt.After(task.CreatedAt) {
				task = t
			}
		}

		status.State = string(task.Status.State)
		status.NodeID = task.NodeID
		status.ContainerID = task.Status.ContainerStatus.ContainerID
		status.CreatedAt = task.CreatedAt.Unix()
		status.UpdatedAt = task.UpdatedAt.Unix()

		if task.Status.Err != "" {
			status.Error = task.Status.Err
		}
	}

	return status, nil
}

// GetServiceLogs streams logs from a service
func (s *SwarmClient) GetServiceLogs(ctx context.Context, serviceID string) (io.ReadCloser, error) {
	return s.cli.ServiceLogs(ctx, serviceID, types.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     true,
		Timestamps: true,
	})
}

// RemoveService stops and removes a service
func (s *SwarmClient) RemoveService(ctx context.Context, serviceID string) error {
	err := s.cli.ServiceRemove(ctx, serviceID)
	if err != nil {
		return fmt.Errorf("failed to remove service %s: %w", serviceID, err)
	}
	return nil
}

// ServiceStatus represents the status of a Docker service
type ServiceStatus struct {
	ID          string
	Name        string
	State       string // "new", "pending", "running", "complete", "failed", "shutdown", "rejected"
	NodeID      string
	ContainerID string
	CreatedAt   int64
	UpdatedAt   int64
	Error       string
}
