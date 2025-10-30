package jobs

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/cumulus/orchestrator/chronos"
	"github.com/cumulus/orchestrator/platform"
	pb "github.com/cumulus/orchestrator/proto"
)

// JobHandler manages job lifecycle and state
type JobHandler struct {
	swarmClient    *platform.SwarmClient
	chronosManager *chronos.ChronosManager
	jobs           map[string]*JobInfo
	mu             sync.RWMutex
}

// JobInfo tracks information about a running job
type JobInfo struct {
	ID            string
	ServiceID     string  // Docker service ID
	WorkerID      string  // Swarm node ID
	PartitionID   string  // Chronos partition ID
	GPUDevice     int     // GPU device index
	GPUAmount     float64 // Allocated GPU amount
	Status        pb.JobState
	CreatedAt     time.Time
	StartedAt     time.Time
	CompletedAt   time.Time
	Error         string
	JobSubmission *pb.JobSubmission
}

// NewJobHandler creates a new JobHandler instance
func NewJobHandler(swarmClient *platform.SwarmClient, chronosManager *chronos.ChronosManager) *JobHandler {
	return &JobHandler{
		swarmClient:    swarmClient,
		chronosManager: chronosManager,
		jobs:           make(map[string]*JobInfo),
	}
}

// GetJob gets information about a job
func (h *JobHandler) GetJob(jobID string) (*JobInfo, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	job, exists := h.jobs[jobID]
	if !exists {
		return nil, false
	}

	// Return a copy to avoid race conditions
	jobCopy := *job
	return &jobCopy, true
}

// ListJobs returns all jobs
func (h *JobHandler) ListJobs() []*JobInfo {
	h.mu.RLock()
	defer h.mu.RUnlock()

	jobs := make([]*JobInfo, 0, len(h.jobs))
	for _, job := range h.jobs {
		// Return copies to avoid race conditions
		jobCopy := *job
		jobs = append(jobs, &jobCopy)
	}

	return jobs
}

// UpdateJobStatus updates the status of a job
func (h *JobHandler) UpdateJobStatus(jobID string, status pb.JobState, message string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	job, exists := h.jobs[jobID]
	if !exists {
		return
	}

	job.Status = status

	switch status {
	case pb.JobState_RUNNING:
		if job.StartedAt.IsZero() {
			job.StartedAt = time.Now()
		}
	case pb.JobState_COMPLETED, pb.JobState_FAILED, pb.JobState_CANCELLED:
		if job.CompletedAt.IsZero() {
			job.CompletedAt = time.Now()
		}
	}
}

// SetJobError sets an error message for a job
func (h *JobHandler) SetJobError(jobID string, err error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	job, exists := h.jobs[jobID]
	if !exists {
		return
	}

	job.Error = err.Error()
	job.Status = pb.JobState_FAILED
	if job.CompletedAt.IsZero() {
		job.CompletedAt = time.Now()
	}
}

// CreateJob creates a new job entry
func (h *JobHandler) CreateJob(jobID string, submission *pb.JobSubmission) *JobInfo {
	h.mu.Lock()
	defer h.mu.Unlock()

	job := &JobInfo{
		ID:            jobID,
		Status:        pb.JobState_SCHEDULING,
		CreatedAt:     time.Now(),
		JobSubmission: submission,
	}

	h.jobs[jobID] = job
	return job
}

// RemoveJob removes a job from tracking
func (h *JobHandler) RemoveJob(jobID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	delete(h.jobs, jobID)
}

// SelectWorker selects an appropriate worker for a job
func (h *JobHandler) SelectWorker(ctx context.Context, gpuAmount float64) (*platform.WorkerInfo, int, error) {
	workers, err := h.swarmClient.ListWorkers(ctx)
	if err != nil {
		return nil, -1, fmt.Errorf("failed to list workers: %w", err)
	}

	if len(workers) == 0 {
		return nil, -1, fmt.Errorf("no GPU workers available")
	}

	// Simple first-fit algorithm for GPU allocation
	for _, worker := range workers {
		if worker.Status != "ready" {
			continue
		}

		deviceIndex, canAllocate := h.canAllocateGPU(worker, gpuAmount)
		if canAllocate {
			return worker, deviceIndex, nil
		}
	}

	return nil, -1, fmt.Errorf("no workers have sufficient GPU capacity for %.2f GPU units", gpuAmount)
}

// canAllocateGPU checks if a worker can allocate the requested GPU amount
func (h *JobHandler) canAllocateGPU(worker *platform.WorkerInfo, gpuAmount float64) (int, bool) {
	if gpuAmount < 1.0 {
		// Fractional GPU: find device with enough free capacity
		for i, gpu := range worker.GPUDevices {
			available := gpu.TotalCapacity - gpu.AllocatedAmount
			if available >= gpuAmount {
				return i, true
			}
		}
	} else {
		// Whole GPU(s): find contiguous free GPUs
		needed := int(gpuAmount)
		freeCount := 0
		startIdx := -1

		for i, gpu := range worker.GPUDevices {
			if gpu.AllocatedAmount == 0 {
				if startIdx == -1 {
					startIdx = i
				}
				freeCount++
				if freeCount == needed {
					return startIdx, true
				}
			} else {
				freeCount = 0
				startIdx = -1
			}
		}
	}

	return -1, false
}

// UpdateWorkerGPUAllocation updates the GPU allocation tracking for a worker
func (h *JobHandler) UpdateWorkerGPUAllocation(workerID string, deviceIndex int, amount float64, allocate bool) error {
	// This would typically update the worker's GPU allocation tracking
	// For now, we'll rely on the Chronos partition tracking
	// In a production system, this might update a database or cache
	return nil
}

// CleanupCompletedJobs removes old completed jobs from memory
func (h *JobHandler) CleanupCompletedJobs(maxAge time.Duration) {
	h.mu.Lock()
	defer h.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)

	for jobID, job := range h.jobs {
		if !job.CompletedAt.IsZero() && job.CompletedAt.Before(cutoff) {
			delete(h.jobs, jobID)
		}
	}
}

// GetActiveJobCount returns the number of active (non-completed) jobs
func (h *JobHandler) GetActiveJobCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()

	count := 0
	for _, job := range h.jobs {
		if job.Status != pb.JobState_COMPLETED &&
			job.Status != pb.JobState_FAILED &&
			job.Status != pb.JobState_CANCELLED {
			count++
		}
	}

	return count
}
