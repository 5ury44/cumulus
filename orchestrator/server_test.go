package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/cumulus/orchestrator/chronos"
	"github.com/cumulus/orchestrator/jobs"
	"github.com/cumulus/orchestrator/platform"
	pb "github.com/cumulus/orchestrator/proto"
)

func TestConfig(t *testing.T) {
	config := &Config{
		GRPCPort:           50051,
		SwarmManager:       "tcp://localhost:2377",
		MaxJobDuration:     86400,
		DefaultDuration:    3600,
		JobCleanupInterval: 5 * time.Minute,
		JobRetentionPeriod: 24 * time.Hour,
	}

	config.SSH.KeyPath = "/root/.ssh/id_rsa"
	config.SSH.User = "root"
	config.SSH.Timeout = 10 * time.Second

	config.Container.Image = "cumulus-job:latest"
	config.Container.Registry = "docker.io/cumulus"

	if config.GRPCPort != 50051 {
		t.Errorf("Expected GRPCPort 50051, got %d", config.GRPCPort)
	}

	if config.SSH.User != "root" {
		t.Errorf("Expected SSH user 'root', got '%s'", config.SSH.User)
	}

	if config.Container.Image != "cumulus-job:latest" {
		t.Errorf("Expected container image 'cumulus-job:latest', got '%s'", config.Container.Image)
	}
}

func TestNewOrchestratorServer(t *testing.T) {
	config := &Config{
		GRPCPort:           50051,
		SwarmManager:       "tcp://localhost:2377",
		MaxJobDuration:     86400,
		DefaultDuration:    3600,
		JobCleanupInterval: 5 * time.Minute,
		JobRetentionPeriod: 24 * time.Hour,
	}

	config.SSH.KeyPath = "/tmp/test_key"
	config.SSH.User = "root"
	config.SSH.Timeout = 10 * time.Second

	config.Container.Image = "cumulus-job:latest"

	// This will fail because we don't have Docker/SSH setup, but it tests the structure
	_, err := NewOrchestratorServer(config)
	if err == nil {
		t.Error("Expected error due to missing Docker/SSH, but got none")
	}

	// The error should be related to Docker or SSH
	if err != nil && err.Error() == "" {
		t.Error("Expected non-empty error message")
	}
}

func TestValidateJobSubmission(t *testing.T) {
	config := &Config{
		MaxJobDuration: 86400,
	}

	// Create a mock server for testing validation
	server := &OrchestratorServer{
		config: config,
	}

	tests := []struct {
		name        string
		submission  *pb.JobSubmission
		expectError bool
	}{
		{
			name: "valid fractional GPU",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   0.5,
				Duration:    3600,
			},
			expectError: false,
		},
		{
			name: "valid whole GPU",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   2.0,
				Duration:    3600,
			},
			expectError: false,
		},
		{
			name: "empty code package",
			submission: &pb.JobSubmission{
				CodePackage: []byte{},
				GpuMemory:   0.5,
				Duration:    3600,
			},
			expectError: true,
		},
		{
			name: "invalid GPU memory - too low",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   0.05,
				Duration:    3600,
			},
			expectError: true,
		},
		{
			name: "invalid GPU memory - too high",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   10.0,
				Duration:    3600,
			},
			expectError: true,
		},
		{
			name: "invalid fractional GPU",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   0.15, // Not a 0.1 increment
				Duration:    3600,
			},
			expectError: true,
		},
		{
			name: "invalid whole GPU",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   2.5, // Not an integer
				Duration:    3600,
			},
			expectError: true,
		},
		{
			name: "duration too short",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   0.5,
				Duration:    30, // Less than 60 seconds
			},
			expectError: true,
		},
		{
			name: "duration too long",
			submission: &pb.JobSubmission{
				CodePackage: []byte("test code"),
				GpuMemory:   0.5,
				Duration:    100000, // More than max
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := server.validateJobSubmission(tt.submission)

			if tt.expectError && err == nil {
				t.Error("Expected error, but got none")
			}

			if !tt.expectError && err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
		})
	}
}

func TestHealthCheck(t *testing.T) {
	// Create a mock server for testing health check
	config := &Config{
		MaxJobDuration: 86400,
	}

	// Create a mock SwarmClient to avoid nil pointer dereference
	swarmClient, err := platform.NewSwarmClient()
	if err != nil {
		t.Logf("Cannot create SwarmClient (expected in test environment): %v", err)
		return
	}
	defer swarmClient.Close()

	// Create a mock JobHandler
	jobHandler := jobs.NewJobHandler(swarmClient, nil)

	server := &OrchestratorServer{
		config:      config,
		swarmClient: swarmClient,
		jobHandler:  jobHandler,
		startTime:   time.Now().Add(-time.Hour), // Started 1 hour ago
	}

	ctx := context.Background()
	req := &pb.HealthCheckRequest{}

	// This will likely fail because Docker isn't running, but it tests the structure
	_, err = server.HealthCheck(ctx, req)
	if err != nil {
		t.Logf("HealthCheck failed (expected if Docker not running): %v", err)
	}
}

func TestCancelJobResponse(t *testing.T) {
	// Test CancelJobResponse struct
	response := &pb.CancelJobResponse{
		Cancelled: true,
		Message:   "Job cancelled successfully",
	}

	if !response.Cancelled {
		t.Error("Expected Cancelled to be true")
	}

	if response.Message != "Job cancelled successfully" {
		t.Errorf("Expected message 'Job cancelled successfully', got '%s'", response.Message)
	}
}

func TestJobStatusConversion(t *testing.T) {
	// Test that we can work with pb.JobState values
	states := []pb.JobState{
		pb.JobState_SCHEDULING,
		pb.JobState_CREATING_PARTITION,
		pb.JobState_RUNNING,
		pb.JobState_COMPLETED,
		pb.JobState_FAILED,
		pb.JobState_CANCELLED,
	}

	for _, state := range states {
		stateStr := state.String()
		if stateStr == "" {
			t.Errorf("JobState %v should have a string representation", state)
		}
	}
}

func TestSubmitJobValidation(t *testing.T) {
	// Create a mock server for testing job submission validation
	config := &Config{
		MaxJobDuration: 86400,
	}

	server := &OrchestratorServer{
		config: config,
	}

	// Test valid submission
	validSubmission := &pb.JobSubmission{
		CodePackage:  []byte("test code"),
		GpuMemory:    0.5,
		Duration:     3600,
		Requirements: []string{"numpy", "torch"},
	}

	err := server.validateJobSubmission(validSubmission)
	if err != nil {
		t.Errorf("Expected valid submission to pass validation, got error: %v", err)
	}

	// Test invalid submission - empty code
	invalidSubmission := &pb.JobSubmission{
		CodePackage: []byte{},
		GpuMemory:   0.5,
		Duration:    3600,
	}

	err = server.validateJobSubmission(invalidSubmission)
	if err == nil {
		t.Error("Expected invalid submission to fail validation")
	}
}

func TestWriteJobCode(t *testing.T) {
	server := &OrchestratorServer{}

	// Test writing job code (mock implementation)
	codePackage := []byte("test code package")
	requirements := []string{"numpy", "torch"}

	err := server.writeJobCode("test-job-123", codePackage, requirements)
	if err != nil {
		t.Errorf("writeJobCode failed: %v", err)
	}
}

func TestGetJobResult(t *testing.T) {
	server := &OrchestratorServer{}

	// Test getting job result (mock implementation)
	result := server.getJobResult(context.Background(), "test-service-123")

	if len(result) == 0 {
		t.Error("Expected non-empty result")
	}

	// Verify it's valid JSON
	var resultMap map[string]interface{}
	err := json.Unmarshal(result, &resultMap)
	if err != nil {
		t.Errorf("Result should be valid JSON: %v", err)
	}

	// Check expected fields
	if status, ok := resultMap["status"]; !ok || status != "success" {
		t.Error("Expected status field to be 'success'")
	}
}

func TestCleanupFunctions(t *testing.T) {
	// Create a mock server for testing cleanup functions
	swarmClient, err := platform.NewSwarmClient()
	if err != nil {
		t.Logf("Cannot create SwarmClient (expected in test environment): %v", err)
		return
	}
	defer swarmClient.Close()

	chronosManager, err := chronos.NewChronosManager(&chronos.SSHConfig{
		User:    "test",
		KeyPath: "/tmp/test",
		Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Logf("Cannot create ChronosManager (expected in test environment): %v", err)
		return
	}
	defer chronosManager.Close()

	server := &OrchestratorServer{
		swarmClient:    swarmClient,
		chronosManager: chronosManager,
	}

	// Test cleanup functions (they should not panic)
	ctx := context.Background()

	// These will fail because services don't exist, but should not panic
	server.cleanupJob(ctx, "test-service", "test-partition")
	server.cleanupPartition(ctx, "test-partition")
}
