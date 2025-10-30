---
inclusion: always
---

# Cumulus Product Overview

Cumulus is a distributed execution system for sending code to remote GPU servers with GPU partitioning. It combines GPU time-sharing with container orchestration for scalable, isolated job execution.

## Core Components

**Central Orchestrator** - A Go-based relay server that receives job submissions from clients, schedules containers on GPU workers via Docker Swarm, and streams results back. Acts as the single control point for all job execution.

**Cumulus SDK** - A Python-based client SDK that packages code and submits jobs to the orchestrator via gRPC. Provides a simple API for remote ML training and compute tasks.

**Chronos (cumulus_core)** - A C++ GPU time-sharing library that provides fair, time-based GPU partitioning with automatic expiration. Works with any GPU vendor (NVIDIA, AMD, Intel, Apple Silicon) via OpenCL.

**Job Containers** - Docker containers that execute user code with GPU access inherited from Chronos partitions. Provides isolation and consistent execution environment.

## Architecture

**Relay Pattern**: All client communication flows through the orchestrator (no direct client-worker connections)
- Client → Orchestrator (gRPC)
- Orchestrator → Docker Swarm → GPU Workers
- Results stream back through orchestrator to client

**Platform-Managed Workers**: Docker Swarm handles worker pool management, health checks, scheduling, and service discovery automatically.

## Key Features

- **Container-based execution** - Jobs run in isolated Docker containers with GPU access
- **Time-based GPU allocation** - Chronos provides automatic cleanup and fair sharing
- **Central control** - Orchestrator can monitor and kill any job
- **One-script worker onboarding** - Vast.ai GPU instances join pool with single command
- **Framework-agnostic checkpointing** - PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM
- **Two-tier checkpoint storage** - L1 (local disk) + L2 (S3 cloud storage)
- **Cross-machine job resumption** - Via S3
- **Sub-1% GPU performance overhead**
- **Simplified security** - No authentication complexity, orchestrator controls all access

## Use Cases

- ML model training on shared GPU infrastructure
- Research labs with limited GPU resources
- Cross-machine training job resumption
- Scalable GPU compute with automatic worker management
- Multi-tenant GPU sharing with time-based isolation
