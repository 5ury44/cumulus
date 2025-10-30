package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"gopkg.in/yaml.v3"

	pb "github.com/cumulus/orchestrator/proto"
)

func main() {
	// Load configuration
	config, err := loadConfig("config.yaml")
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Create orchestrator server
	server, err := NewOrchestratorServer(config)
	if err != nil {
		log.Fatalf("Failed to create orchestrator server: %v", err)
	}
	defer server.Close()

	// Create gRPC server
	grpcServer := grpc.NewServer()
	pb.RegisterOrchestratorServiceServer(grpcServer, server)

	// Start listening
	listen, err := net.Listen("tcp", fmt.Sprintf(":%d", config.GRPCPort))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", config.GRPCPort, err)
	}

	// Start server in a goroutine
	go func() {
		log.Printf("🚀 Cumulus Orchestrator starting on port %d", config.GRPCPort)
		if err := grpcServer.Serve(listen); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("🛑 Shutting down orchestrator...")

	// Graceful shutdown
	grpcServer.GracefulStop()
	log.Println("✅ Orchestrator stopped")
}

// loadConfig loads configuration from a YAML file
func loadConfig(configPath string) (*Config, error) {
	// Set default configuration
	config := &Config{
		GRPCPort:           50051,
		SwarmManager:       "tcp://localhost:2377",
		MaxJobDuration:     86400, // 24 hours
		DefaultDuration:    3600,  // 1 hour
		JobCleanupInterval: 5 * time.Minute,
		JobRetentionPeriod: 24 * time.Hour,
	}

	// Set default SSH config
	config.SSH.KeyPath = "/root/.ssh/id_rsa"
	config.SSH.User = "root"
	config.SSH.Timeout = 10 * time.Second

	// Set default container config
	config.Container.Image = "cumulus-job:latest"
	config.Container.Registry = "docker.io/cumulus"

	// Try to load from file
	if _, err := os.Stat(configPath); err == nil {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}

		if err := yaml.Unmarshal(data, config); err != nil {
			return nil, fmt.Errorf("failed to parse config file: %w", err)
		}

		log.Printf("📋 Loaded configuration from %s", configPath)
	} else {
		log.Printf("📋 Using default configuration (config file %s not found)", configPath)
	}

	return config, nil
}
