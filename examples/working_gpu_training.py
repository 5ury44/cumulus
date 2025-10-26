#!/usr/bin/env python3
"""
Working GPU training example with proper tensor handling
"""

import sys
import os
sys.path.insert(0, '/tmp/cumulus')

from sdk.client import CumulusClient
import time


def train_model():
    """Train a neural network model on GPU."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)
    
    # Generate sample data
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(3):  # Reduced epochs for demo
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} completed, Average Loss: {total_loss/len(dataloader):.4f}")
    
    # Convert model state to CPU and convert to lists for JSON serialization
    model_state_cpu = {}
    for k, v in model.state_dict().items():
        model_state_cpu[k] = v.cpu().tolist()
    
    # Return model state
    return {
        "model_state": model_state_cpu,
        "final_loss": total_loss / len(dataloader),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters())
    }


def evaluate_model(model_state):
    """Evaluate a trained model."""
    import torch
    import torch.nn as nn
    
    # Recreate model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Convert lists back to tensors
    model_state_tensors = {}
    for k, v in model_state.items():
        model_state_tensors[k] = torch.tensor(v)
    
    # Load state
    model.load_state_dict(model_state_tensors)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate test data
    X_test = torch.randn(100, 784).to(device)
    y_test = torch.randint(0, 10, (100,)).to(device)
    
    # Evaluate
    with torch.no_grad():
        predictions = model(X_test)
        accuracy = (predictions.argmax(dim=1) == y_test).float().mean()
    
    return {
        "accuracy": float(accuracy),
        "test_samples": len(X_test),
        "device": str(device)
    }


def main():
    print("🚀 Starting working GPU training example...")
    
    # Create client
    client = CumulusClient("http://localhost:8081")
    
    # Train model
    print("Training model on remote GPU...")
    training_result = client.run(
        func=train_model,
        gpu_memory=0.8,
        duration=1800,
        requirements=["torch"]
    )
    print(f"Training completed: {training_result}")
    
    # Evaluate model
    print("Evaluating model on remote GPU...")
    evaluation_result = client.run(
        func=evaluate_model,
        args=[training_result["model_state"]],
        gpu_memory=0.6,
        duration=1800,
        requirements=["torch"]
    )
    print(f"Evaluation completed: {evaluation_result}")
    
    print("✅ Example completed successfully!")


if __name__ == "__main__":
    main()
