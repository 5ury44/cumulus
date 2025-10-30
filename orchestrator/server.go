package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/cumulus/orchestrator/chronos"
	"github.com/cumulus/orchestrator/jobs"
	"github.com/cumulus/orchestrator/platform"
	pb "github.com/cumulus/orchestrator/proto"
)

// OrchestratorServer implements the gRPC OrchestratorService
type OrchestratorServer struct {
	pb.UnimplementedOrchestratorServiceServer

	swarmClient    *platform.SwarmClient
	chronosManager *chronos.ChronosManager
	jobHandler     *jobs.JobHandler
	config         *Config
	startTime      time.Time
	mu             sync.RWMutex
}

// Config holds orchestrator configuration
type Config struct {
	GRPCPort     int    `yaml:"grpc_port"`
	SwarmManager string `yaml:"swarm_manager"`

	SSH struct {
		KeyPath string        `yaml:"key_path"`
		User    string        `yaml:"user"`
		Timeout time.Duration `yaml:"timeout"`
	} `yaml:"ssh"`

	Container struct {
		Image    string `yaml:"image"`
		Registry string `yaml:"registry"`
	} `yaml:"container"`

	MaxJobDuration     int           `yaml:"max_job_duration"`
	DefaultDuration    int           `yaml:"default_duration"`
	JobCleanupInterval time.Duration `yaml:"job_cleanup_interval"`
	JobRetentionPeriod time.Duration `yaml:"job_retention_period"`
}

// NewOrchestratorServer creates a new OrchestratorServer instance
func NewOrchestratorServer(config *Config) (*OrchestratorServer, error) {
	// Initialize Swarm client
	swarmClient, err := platform.NewSwarmClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Swarm client: %w", err)
	}

	// Initialize Chronos manager
	chronosConfig := &chronos.SSHConfig{
		User:       config.SSH.User,
		KeyPath:    config.SSH.KeyPath,
		Timeout:    config.SSH.Timeout,
		MaxRetries: 3,
	}

	chronosManager, err := chronos.NewChronosManager(chronosConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Chronos manager: %w", err)
	}

	// Initialize job handler
	jobHandler := jobs.NewJobHandler(swarmClient, chronosManager)

	server := &OrchestratorServer{
		swarmClient:    swarmClient,
		chronosManager: chronosManager,
		jobHandler:     jobHandler,
		config:         config,
		startTime:      time.Now(),
	}

	// Start background cleanup routine
	go server.backgroundCleanup()

	return server, nil
}

// Close closes all connections and cleans up resources
func (s *OrchestratorServer) Close() error {
	var errors []error

	if err := s.swarmClient.Close(); err != nil {
		errors = append(errors, fmt.Errorf("failed to close Swarm client: %w", err))
	}

	if err := s.chronosManager.Close(); err != nil {
		errors = append(errors, fmt.Errorf("failed to close Chronos manager: %w", err))
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors during shutdown: %v", errors)
	}

	return nil
}

// backgroundCleanup runs periodic cleanup tasks
func (s *OrchestratorServer) backgroundCleanup() {
	ticker := time.NewTicker(s.config.JobCleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		// Clean up old completed jobs
		s.jobHandler.CleanupCompletedJobs(s.config.JobRetentionPeriod)

		// Clean up expired Chronos partitions
		s.chronosManager.CleanupExpiredPartitions()
	}
}

// HealthCheck implements the HealthCheck RPC
func (s *OrchestratorServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthStatus, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check Swarm connectivity
	swarmConnected := true
	_, err := s.swarmClient.IsSwarmManager(ctx)
	if err != nil {
		swarmConnected = false
	}

	// Get worker count
	workers, err := s.swarmClient.ListWorkers(ctx)
	availableWorkers := int32(0)
	if err == nil {
		for _, worker := range workers {
			if worker.Status == "ready" {
				availableWorkers++
			}
		}
	}

	// Determine overall status
	status := "healthy"
	if !swarmConnected {
		status = "degraded"
	}
	if availableWorkers == 0 {
		status = "unhealthy"
	}

	return &pb.HealthStatus{
		Status:           status,
		Version:          "1.0.0", // TODO: Get from build info
		ActiveJobs:       int32(s.jobHandler.GetActiveJobCount()),
		AvailableWorkers: availableWorkers,
		SwarmConnected:   swarmConnected,
		UptimeSeconds:    int64(time.Since(s.startTime).Seconds()),
	}, nil
}

// ListWorkers implements the ListWorkers RPC
func (s *OrchestratorServer) ListWorkers(ctx context.Context, req *pb.ListWorkersRequest) (*pb.WorkerList, error) {
	workers, err := s.swarmClient.ListWorkers(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list workers: %w", err)
	}

	// Convert to proto format
	protoWorkers := make([]*pb.WorkerInfo, len(workers))
	for i, worker := range workers {
		protoWorkers[i] = &pb.WorkerInfo{
			Id:       worker.ID,
			Hostname: worker.Hostname,
			Address:  worker.Address,
			Status:   worker.Status,
		}

		// Convert GPU devices
		gpuDevices := make([]*pb.GPUDevice, len(worker.GPUDevices))
		for j, gpu := range worker.GPUDevices {
			gpuDevices[j] = &pb.GPUDevice{
				Index:           int32(gpu.Index),
				Name:            gpu.Name,
				MemoryMb:        int32(gpu.MemoryMB),
				TotalCapacity:   float32(gpu.TotalCapacity),
				AllocatedAmount: float32(gpu.AllocatedAmount),
			}

			// Convert active partitions
			partitions := make([]*pb.PartitionInfo, len(gpu.ActivePartitions))
			for k, partition := range gpu.ActivePartitions {
				partitions[k] = &pb.PartitionInfo{
					PartitionId: partition.PartitionID,
					JobId:       partition.JobID,
					Amount:      float32(partition.Amount),
					CreatedAt:   partition.CreatedAt,
					ExpiresAt:   partition.ExpiresAt,
				}
			}
			gpuDevices[j].ActivePartitions = partitions
		}
		protoWorkers[i].GpuDevices = gpuDevices
	}

	return &pb.WorkerList{
		Workers: protoWorkers,
	}, nil
}

// GetJobStatus implements the GetJobStatus RPC
func (s *OrchestratorServer) GetJobStatus(ctx context.Context, req *pb.JobStatusRequest) (*pb.JobStatus, error) {
	job, exists := s.jobHandler.GetJob(req.JobId)
	if !exists {
		return nil, fmt.Errorf("job %s not found", req.JobId)
	}

	return &pb.JobStatus{
		JobId:       job.ID,
		State:       job.Status,
		Message:     fmt.Sprintf("Job is %s", job.Status.String()),
		WorkerId:    job.WorkerID,
		PartitionId: job.PartitionID,
		ContainerId: job.ServiceID,
		CreatedAt:   job.CreatedAt.Unix(),
		StartedAt:   job.StartedAt.Unix(),
		CompletedAt: job.CompletedAt.Unix(),
		Error:       job.Error,
	}, nil
}

// CancelJob implements the CancelJob RPC
func (s *OrchestratorServer) CancelJob(ctx context.Context, req *pb.JobCancelRequest) (*pb.CancelJobResponse, error) {
	job, exists := s.jobHandler.GetJob(req.JobId)
	if !exists {
		return &pb.CancelJobResponse{
			Cancelled: false,
			Message:   fmt.Sprintf("Job %s not found", req.JobId),
		}, nil
	}

	// Only cancel if job is not already completed
	if job.Status == pb.JobState_COMPLETED ||
		job.Status == pb.JobState_FAILED ||
		job.Status == pb.JobState_CANCELLED {
		return &pb.CancelJobResponse{
			Cancelled: false,
			Message:   fmt.Sprintf("Job %s is already %s", req.JobId, job.Status.String()),
		}, nil
	}

	// Remove Docker service if it exists
	if job.ServiceID != "" {
		if err := s.swarmClient.RemoveService(ctx, job.ServiceID); err != nil {
			// Log error but continue with cancellation
			fmt.Printf("Warning: failed to remove service %s: %v\n", job.ServiceID, err)
		}
	}

	// Release Chronos partition if it exists
	if job.PartitionID != "" {
		if err := s.chronosManager.ReleasePartition(ctx, job.PartitionID); err != nil {
			// Log error but continue with cancellation
			fmt.Printf("Warning: failed to release partition %s: %v\n", job.PartitionID, err)
		}
	}

	// Update job status
	s.jobHandler.UpdateJobStatus(req.JobId, pb.JobState_CANCELLED, "Job cancelled by user")

	return &pb.CancelJobResponse{
		Cancelled: true,
		Message:   fmt.Sprintf("Job %s cancelled successfully", req.JobId),
	}, nil
}

// SubmitJob implements the SubmitJob RPC - streams job execution progress
func (s *OrchestratorServer) SubmitJob(submission *pb.JobSubmission, stream pb.OrchestratorService_SubmitJobServer) error {
	// Generate unique job ID
	jobID := fmt.Sprintf("job-%d", time.Now().UnixNano())

	// Validate job submission
	if err := s.validateJobSubmission(submission); err != nil {
		return fmt.Errorf("invalid job submission: %w", err)
	}

	// Create job entry
	_ = s.jobHandler.CreateJob(jobID, submission)

	// Send initial event
	if err := stream.Send(&pb.JobEvent{
		JobId:     jobID,
		State:     pb.JobState_SCHEDULING,
		Message:   "Job received, selecting worker...",
		Timestamp: time.Now().Unix(),
	}); err != nil {
		return fmt.Errorf("failed to send initial event: %w", err)
	}

	// Execute job asynchronously and stream results
	go s.executeJob(jobID, submission, stream)

	return nil
}

// executeJob handles the actual job execution workflow
func (s *OrchestratorServer) executeJob(jobID string, submission *pb.JobSubmission, stream pb.OrchestratorService_SubmitJobServer) {
	ctx := context.Background()

	// Helper function to send events
	sendEvent := func(state pb.JobState, message string, result []byte) {
		event := &pb.JobEvent{
			JobId:     jobID,
			State:     state,
			Message:   message,
			Timestamp: time.Now().Unix(),
		}
		if result != nil {
			event.Result = result
		}

		if err := stream.Send(event); err != nil {
			fmt.Printf("Failed to send event for job %s: %v\n", jobID, err)
		}
	}

	// Helper function to handle errors
	handleError := func(err error, message string) {
		s.jobHandler.SetJobError(jobID, err)
		sendEvent(pb.JobState_FAILED, fmt.Sprintf("%s: %v", message, err), nil)
	}

	// Step 1: Select worker
	worker, deviceIndex, err := s.jobHandler.SelectWorker(ctx, float64(submission.GpuMemory))
	if err != nil {
		handleError(err, "Failed to select worker")
		return
	}

	// Update job with worker info
	if job, exists := s.jobHandler.GetJob(jobID); exists {
		job.WorkerID = worker.ID
		job.GPUDevice = deviceIndex
		job.GPUAmount = float64(submission.GpuMemory)
	}

	sendEvent(pb.JobState_CREATING_PARTITION,
		fmt.Sprintf("Selected worker %s (GPU %d), creating partition...", worker.Hostname, deviceIndex), nil)

	// Step 2: Create Chronos partition
	partitionID, err := s.chronosManager.CreatePartition(
		ctx,
		worker.ID,
		worker.Address,
		deviceIndex,
		float64(submission.GpuMemory),
		int(submission.Duration),
		jobID,
	)
	if err != nil {
		handleError(err, "Failed to create GPU partition")
		return
	}

	// Update job with partition info
	if job, exists := s.jobHandler.GetJob(jobID); exists {
		job.PartitionID = partitionID
	}
	s.jobHandler.UpdateJobStatus(jobID, pb.JobState_CREATING_PARTITION, "Partition created")

	sendEvent(pb.JobState_RUNNING,
		fmt.Sprintf("GPU partition %s created, starting container...", partitionID), nil)

	// Step 3: Create container service spec
	serviceSpec := &platform.ServiceSpec{
		Image:   s.config.Container.Image,
		Command: []string{"/app/entrypoint.sh"},
		Environment: []string{
			fmt.Sprintf("JOB_ID=%s", jobID),
			fmt.Sprintf("CHRONOS_PARTITION_ID=%s", partitionID),
			fmt.Sprintf("GPU_DEVICE=%d", deviceIndex),
		},
		NodeID: worker.ID,
		Mounts: []platform.Mount{
			{
				Source: fmt.Sprintf("/tmp/cumulus-jobs/%s", jobID),
				Target: "/job",
				Type:   "bind",
			},
		},
	}

	// Step 4: Write job code to worker (simplified - in production this would be more sophisticated)
	if err := s.writeJobCode(jobID, submission.CodePackage, submission.Requirements); err != nil {
		handleError(err, "Failed to write job code")
		s.cleanupPartition(ctx, partitionID)
		return
	}

	// Step 5: Create Docker service
	serviceID, err := s.swarmClient.CreateService(ctx, serviceSpec)
	if err != nil {
		handleError(err, "Failed to create container service")
		s.cleanupPartition(ctx, partitionID)
		return
	}

	// Update job with service info
	if job, exists := s.jobHandler.GetJob(jobID); exists {
		job.ServiceID = serviceID
	}
	s.jobHandler.UpdateJobStatus(jobID, pb.JobState_RUNNING, "Container started")

	sendEvent(pb.JobState_RUNNING,
		fmt.Sprintf("Container %s started, executing job...", serviceID[:12]), nil)

	// Step 6: Monitor job execution
	s.monitorJobExecution(ctx, jobID, serviceID, partitionID, stream)
}

// writeJobCode writes the job code package to the worker (simplified implementation)
func (s *OrchestratorServer) writeJobCode(jobID string, codePackage []byte, requirements []string) error {
	// In a real implementation, this would:
	// 1. Create a temporary directory on the worker
	// 2. Write the code package (ZIP file) to the directory
	// 3. Create a requirements.txt file
	// 4. Set proper permissions

	// For now, we'll simulate this step
	fmt.Printf("Writing job code for %s (%d bytes)\n", jobID, len(codePackage))

	// Simulate some work
	time.Sleep(100 * time.Millisecond)

	return nil
}

// monitorJobExecution monitors the container execution and streams results
func (s *OrchestratorServer) monitorJobExecution(ctx context.Context, jobID, serviceID, partitionID string, stream pb.OrchestratorService_SubmitJobServer) {
	// Helper function to send events
	sendEvent := func(state pb.JobState, message string, result []byte) {
		event := &pb.JobEvent{
			JobId:     jobID,
			State:     state,
			Message:   message,
			Timestamp: time.Now().Unix(),
		}
		if result != nil {
			event.Result = result
		}

		if err := stream.Send(event); err != nil {
			fmt.Printf("Failed to send event for job %s: %v\n", jobID, err)
		}
	}

	// Monitor container status
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	timeout := time.After(time.Duration(s.config.MaxJobDuration) * time.Second)

	for {
		select {
		case <-timeout:
			// Job timed out
			s.jobHandler.SetJobError(jobID, fmt.Errorf("job exceeded maximum duration"))
			sendEvent(pb.JobState_FAILED, "Job timed out", nil)
			s.cleanupJob(ctx, serviceID, partitionID)
			return

		case <-ticker.C:
			// Check container status
			status, err := s.swarmClient.GetServiceStatus(ctx, serviceID)
			if err != nil {
				fmt.Printf("Failed to get service status for %s: %v\n", serviceID, err)
				continue
			}

			switch status.State {
			case "complete":
				// Job completed successfully
				result := s.getJobResult(ctx, serviceID)
				s.jobHandler.UpdateJobStatus(jobID, pb.JobState_COMPLETED, "Job completed successfully")
				sendEvent(pb.JobState_COMPLETED, "Job completed successfully", result)
				s.cleanupJob(ctx, serviceID, partitionID)
				return

			case "failed":
				// Job failed
				errorMsg := status.Error
				if errorMsg == "" {
					errorMsg = "Container execution failed"
				}
				s.jobHandler.SetJobError(jobID, fmt.Errorf("%s", errorMsg))
				sendEvent(pb.JobState_FAILED, errorMsg, nil)
				s.cleanupJob(ctx, serviceID, partitionID)
				return

			case "running":
				// Job is still running - could send progress updates here
				// For now, we'll just continue monitoring
				continue

			default:
				// Other states (pending, etc.) - continue monitoring
				continue
			}
		}
	}
}

// getJobResult retrieves the job result from the container
func (s *OrchestratorServer) getJobResult(ctx context.Context, serviceID string) []byte {
	// In a real implementation, this would:
	// 1. Get the container logs
	// 2. Parse the JSON result from stdout
	// 3. Return the result data

	// For now, return a mock result
	result := map[string]interface{}{
		"status":    "success",
		"result":    "Job completed successfully",
		"timestamp": time.Now().Unix(),
	}

	resultBytes, _ := json.Marshal(result)
	return resultBytes
}

// cleanupJob cleans up job resources (container and partition)
func (s *OrchestratorServer) cleanupJob(ctx context.Context, serviceID, partitionID string) {
	// Remove Docker service
	if serviceID != "" {
		if err := s.swarmClient.RemoveService(ctx, serviceID); err != nil {
			fmt.Printf("Warning: failed to remove service %s: %v\n", serviceID, err)
		}
	}

	// Release Chronos partition
	s.cleanupPartition(ctx, partitionID)
}

// cleanupPartition releases a Chronos partition
func (s *OrchestratorServer) cleanupPartition(ctx context.Context, partitionID string) {
	if partitionID != "" {
		if err := s.chronosManager.ReleasePartition(ctx, partitionID); err != nil {
			fmt.Printf("Warning: failed to release partition %s: %v\n", partitionID, err)
		}
	}
}

// validateJobSubmission validates a job submission
func (s *OrchestratorServer) validateJobSubmission(submission *pb.JobSubmission) error {
	if len(submission.CodePackage) == 0 {
		return fmt.Errorf("code package is required")
	}

	if submission.GpuMemory <= 0 {
		return fmt.Errorf("GPU memory must be positive")
	}

	// Validate GPU amount (fractional 0.1-0.9 or whole 1-8)
	if submission.GpuMemory < 0.1 || submission.GpuMemory > 8.0 {
		return fmt.Errorf("GPU memory must be between 0.1 and 8.0")
	}

	// Check if it's a valid fractional or whole number
	if submission.GpuMemory < 1.0 {
		// Fractional GPU - check if it's a valid increment (0.1, 0.2, etc.)
		scaled := int(submission.GpuMemory * 10)
		if float32(scaled)/10.0 != submission.GpuMemory {
			return fmt.Errorf("fractional GPU must be in 0.1 increments (0.1, 0.2, ..., 0.9)")
		}
	} else {
		// Whole GPU - check if it's an integer
		if float32(int(submission.GpuMemory)) != submission.GpuMemory {
			return fmt.Errorf("whole GPU allocation must be an integer (1, 2, 3, ...)")
		}
	}

	if submission.Duration < 60 || submission.Duration > int32(s.config.MaxJobDuration) {
		return fmt.Errorf("duration must be between 60 and %d seconds", s.config.MaxJobDuration)
	}

	return nil
}
