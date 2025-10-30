# Requirements Document

## Introduction

This document specifies the requirements for adding a central orchestrator to the Cumulus distributed execution system. Currently, clients connect directly to individual GPU worker nodes. The new architecture introduces a central relay server that receives job submissions (code + requirements), schedules containers on GPU workers, and streams results back to clients.

**Key Architectural Decisions:**
- **Orchestrator as Relay**: All client communication goes through the orchestrator (no direct client-worker connections)
- **Container-Based Execution**: Jobs run in Docker containers with GPU access via Chronos
- **Platform-Managed Workers**: Use existing orchestration platforms (Nomad or Docker Swarm) for worker pool management, health checks, and scheduling
- **One-Script Onboarding**: Vast.ai GPU instances join the pool by running a single setup script

This design eliminates authentication complexity, leverages battle-tested orchestration platforms, and makes worker onboarding trivial.

## Glossary

- **Central Orchestrator**: The relay server that receives job submissions from clients, schedules containers on GPU workers, and streams results back
- **GPU Worker**: A Docker host with GPU resources that runs job containers (managed by Nomad/Swarm)
- **Client SDK**: The Python SDK used by end users to submit jobs to the orchestrator
- **Job Submission**: A complete job package including code, requirements, and GPU resource needs sent to the orchestrator
- **Worker Pool**: The collection of GPU-enabled Docker hosts managed by the orchestration platform (Nomad or Docker Swarm)
- **Container Orchestration Platform**: External system (Nomad or Docker Swarm) that handles worker registration, health checks, scheduling, and service discovery
- **Job Container**: A Docker container that runs user code with GPU access via Chronos
- **Chronos**: The C++ GPU partitioning library that provides time-based GPU slicing
- **Relay Architecture**: Design pattern where all communication flows through the orchestrator (no direct client-worker connections)
- **One-Script Setup**: A single bash script that configures a Vast.ai instance to join the worker pool automatically

## Requirements

### Requirement 1: Central Orchestrator as Relay Server

**User Story:** As a system administrator, I want a central orchestrator that acts as a relay between clients and GPU workers, so that I can control all job execution centrally.

#### Acceptance Criteria

1. THE Central Orchestrator SHALL expose an HTTP or gRPC API endpoint for receiving complete job submissions from clients
2. THE Central Orchestrator SHALL integrate with a container orchestration platform for worker pool management
3. THE Central Orchestrator SHALL relay all job communication between clients and workers without requiring direct client-worker connections
4. WHEN the Central Orchestrator receives a job submission, THE Central Orchestrator SHALL include code, requirements, and GPU resource specifications
5. THE Central Orchestrator SHALL provide the ability to terminate any running job by killing its container

### Requirement 2: Platform-Managed Worker Pool

**User Story:** As a system administrator, I want the orchestration platform to handle worker registration and health checks automatically, so that I don't need to build custom worker management.

#### Acceptance Criteria

1. THE Central Orchestrator SHALL delegate worker pool management to an external orchestration platform
2. THE orchestration platform SHALL handle worker registration, health checks, and failure detection automatically
3. THE Central Orchestrator SHALL query the orchestration platform for available workers when scheduling jobs
4. WHEN a worker becomes unhealthy, THE orchestration platform SHALL remove it from the available pool without orchestrator intervention
5. THE Central Orchestrator SHALL support either HashiCorp Nomad or Docker Swarm as the orchestration platform

### Requirement 3: Complete Job Submission Handling

**User Story:** As a developer, I want to submit my complete job (code + requirements) to the orchestrator and receive results, so that I only need to interact with one endpoint.

#### Acceptance Criteria

1. WHEN the Client SDK submits a job, THE Central Orchestrator SHALL receive the complete job package including code, Python requirements, GPU memory fraction, and duration
2. THE Central Orchestrator SHALL validate that the job submission contains all required fields
3. WHEN a job submission specifies resource requirements that exceed all available workers, THEN THE Central Orchestrator SHALL return an error response with status code 503
4. THE Central Orchestrator SHALL assign each valid job submission a unique job identifier
5. THE Central Orchestrator SHALL stream job progress and results back to the client through the same connection

### Requirement 4: GPU Partition and Container Scheduling

**User Story:** As a system operator, I want the orchestrator to create GPU partitions and schedule job containers within them, so that jobs execute with proper GPU resource isolation.

#### Acceptance Criteria

1. WHEN scheduling a job, THE Central Orchestrator SHALL first create a Chronos GPU partition on the selected worker with the requested memory fraction and duration
2. THE Central Orchestrator SHALL create a container specification that launches within the Chronos partition context
3. THE Central Orchestrator SHALL submit the container specification to the orchestration platform for scheduling on the worker with the active partition
4. THE orchestration platform SHALL start the job container with GPU access inherited from the Chronos partition
5. THE Central Orchestrator SHALL monitor container status and capture output for streaming to the client

### Requirement 5: Result Streaming to Client

**User Story:** As a developer, I want to receive job progress and results through the orchestrator, so that I don't need to connect to workers directly.

#### Acceptance Criteria

1. WHEN a job container starts executing, THE Central Orchestrator SHALL stream status updates to the client
2. THE Central Orchestrator SHALL capture container stdout and stderr for streaming to the client
3. WHEN the job completes successfully, THE Central Orchestrator SHALL return the final result to the client
4. WHEN the job fails, THE Central Orchestrator SHALL return error details including logs to the client
5. THE client connection SHALL remain open throughout job execution for continuous progress updates

### Requirement 6: Simplified Client SDK Integration

**User Story:** As a developer, I want to use the Python SDK to submit jobs to the orchestrator and receive results, so that I have a simple single-endpoint API.

#### Acceptance Criteria

1. THE Client SDK SHALL send complete job submissions to the Central Orchestrator including code, requirements, and GPU specifications
2. THE Client SDK SHALL maintain a single connection to the orchestrator for the duration of job execution
3. THE Client SDK SHALL receive streaming updates on job progress through the orchestrator connection
4. THE Client SDK SHALL receive final results or errors through the same orchestrator connection
5. THE Client SDK SHALL NOT establish direct connections to GPU workers

### Requirement 7: Container Execution with GPU Partitioning

**User Story:** As a security administrator, I want jobs to run in isolated containers with time-sliced GPU access, so that jobs cannot interfere with each other or monopolize GPU resources.

#### Acceptance Criteria

1. WHEN scheduling a job, THE system SHALL create a Chronos GPU partition on the target worker with specified memory fraction and duration
2. THE job container SHALL be launched within the context of the Chronos partition to inherit GPU access limits
3. THE job container SHALL extract the user's code package and install requirements in an isolated environment
4. WHEN the job completes or fails, THE system SHALL release the Chronos partition to free GPU resources
5. THE orchestration platform SHALL enforce additional resource limits on containers including CPU and system memory

### Requirement 8: Platform API Integration

**User Story:** As a system architect, I want the orchestrator to use the orchestration platform's native API, so that we leverage existing reliability features.

#### Acceptance Criteria

1. THE Central Orchestrator SHALL use the orchestration platform's native API for all worker and container operations
2. THE orchestration platform SHALL handle network failures, retries, and worker reconnections automatically
3. THE Central Orchestrator SHALL query the platform API for worker status and availability
4. THE Central Orchestrator SHALL submit container specifications to the platform API for scheduling
5. THE Central Orchestrator SHALL monitor container status through the platform API

### Requirement 9: Container Lifecycle Management

**User Story:** As a system operator, I want automatic cleanup of completed job containers, so that resources are released and available for new jobs.

#### Acceptance Criteria

1. WHEN a job container completes successfully, THE orchestration platform SHALL remove the container automatically
2. WHEN a job container fails, THE orchestration platform SHALL remove the container and make resources available
3. WHEN a job exceeds its allocated duration, THE Central Orchestrator SHALL terminate the container via the platform API
4. THE Central Orchestrator SHALL maintain job history including status, duration, and results for at least 24 hours
5. THE orchestration platform SHALL clean up container filesystem and network resources automatically

### Requirement 10: Monitoring and Observability

**User Story:** As a system administrator, I want to monitor the orchestrator and worker pool status, so that I can identify and resolve issues quickly.

#### Acceptance Criteria

1. THE Central Orchestrator SHALL expose a health check endpoint that returns status code 200 when operational
2. THE Central Orchestrator SHALL log all job submissions, container scheduling events, and errors with timestamps
3. THE Central Orchestrator SHALL provide an API endpoint listing all active jobs with their status and assigned workers
4. THE orchestration platform SHALL provide native monitoring and metrics for worker health and resource usage
5. THE Central Orchestrator SHALL expose basic metrics including active jobs count, job success rate, and average job duration

### Requirement 11: Simplified Configuration Management

**User Story:** As a system administrator, I want minimal configuration for the orchestrator, so that deployment is simple and straightforward.

#### Acceptance Criteria

1. THE Central Orchestrator SHALL load configuration from a simple YAML file specifying the orchestration platform endpoint
2. THE Central Orchestrator SHALL support environment variables for sensitive configuration
3. THE configuration file SHALL specify the orchestration platform type, endpoint, and container image for jobs
4. THE Central Orchestrator SHALL validate configuration on startup and report errors for invalid entries
5. THE orchestration platform SHALL handle worker configuration independently of the orchestrator

### Requirement 12: Error Handling and Resilience

**User Story:** As a developer, I want clear error messages when job submission fails, so that I can understand and resolve issues quickly.

#### Acceptance Criteria

1. WHEN no workers are available in the pool, THE Central Orchestrator SHALL return an error response with message "No GPU workers available" and status code 503
2. WHEN the orchestration platform cannot schedule the container, THE Central Orchestrator SHALL return an error response with scheduling failure details
3. WHEN a job container fails during execution, THE Central Orchestrator SHALL return error details including container logs
4. THE Central Orchestrator SHALL include a job identifier in all error responses for troubleshooting
5. THE Central Orchestrator SHALL continue operating normally when individual containers or workers fail

### Requirement 13: Technology Stack and Implementation

**User Story:** As a system architect, I want to use proven technologies and platforms, so that the system is reliable and maintainable.

#### Acceptance Criteria

1. THE Central Orchestrator SHALL be implemented in Go version 1.23 or higher
2. THE orchestration platform SHALL be Docker Swarm
3. THE job containers SHALL use a standard Docker image with Python, OpenCL, and common ML dependencies
4. THE Client SDK SHALL remain in Python with minimal changes to the existing API
5. THE Central Orchestrator SHALL communicate with clients via gRPC and with Docker Swarm via the Docker Engine API

### Requirement 14: One-Script Worker Onboarding

**User Story:** As a system operator, I want to add GPU workers to the pool by running a single setup script, so that onboarding Vast.ai instances is trivial.

#### Acceptance Criteria

1. THE system SHALL provide a worker setup script that installs all dependencies including Docker, Chronos, and platform agents
2. WHEN the setup script runs on a fresh Vast.ai GPU instance, THE instance SHALL join the worker pool within 5 minutes
3. THE setup script SHALL configure GPU access, install the Chronos library, and register with the orchestration platform
4. THE orchestration platform SHALL automatically detect the new worker and make it available for job scheduling
5. THE setup script SHALL accept the orchestrator endpoint as a parameter and configure the worker accordingly
6. THE setup script SHALL configure the Docker daemon to accept remote connections for Swarm communication
7. WHEN provisioning a Vast.ai instance as Swarm manager, THE system SHALL expose port 2377 for Swarm communication and port 50051 for orchestrator gRPC

### Requirement 15: Standard Job Container Image

**User Story:** As a system architect, I want a single container image for all jobs, so that job execution is consistent and predictable.

#### Acceptance Criteria

1. THE system SHALL provide a Docker image containing Python, OpenCL libraries, and common ML frameworks
2. THE job container SHALL accept the code package as input and extract it to a working directory
3. THE job container SHALL install Python requirements specified in the job submission
4. THE job container SHALL inherit GPU access from the Chronos partition created before container launch
5. THE job container SHALL write results to stdout in JSON format for the orchestrator to capture
