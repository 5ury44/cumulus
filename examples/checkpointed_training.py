#!/usr/bin/env python3
"""
Checkpointed GPU training example using Cumulus SDK with pause/resume support
"""

import sys
import os
sys.path.insert(0, '/tmp/cumulus')

from sdk.client import CumulusClient
import time
from typing import Dict, List


def train_model_with_checkpoints(resume_from: str = None, ckpt_every_steps: int = 200, epochs: int = 5):
    """Train a neural network model on GPU with checkpointing support."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from runtime import Checkpointer, should_pause
    
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
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Generate sample data
    X = torch.randn(10000, 784).to(device)
    y = torch.randint(0, 10, (10000,)).to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Checkpointing
    ckpt = Checkpointer()
    start_epoch, start_step = 0, 0
    
    if resume_from:
        ckpt.path = resume_from
    
    if ckpt.exists():
        print(f"Loading checkpoint from {ckpt.path}")
        state = ckpt.load(model, optimizer)
        start_epoch, start_step = int(state['epoch']), int(state['step'])
        print(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        step = 0 if epoch > start_epoch else start_step
        
        for i, (data, target) in enumerate(dataloader):
            if i < step:  # Skip to resume point
                continue
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            step = i + 1
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}")
            
            # Check for pause signal
            if should_pause():
                print("Pause signal detected, saving checkpoint...")
                ckpt.save(model, optimizer, epoch, step)
                return {
                    'status': 'paused',
                    'checkpoint': ckpt.path,
                    'epoch': epoch,
                    'step': step,
                    'device': str(device)
                }
            
            # Periodic checkpointing
            if ckpt.time_to_checkpoint(step, every_steps=ckpt_every_steps):
                print(f"Checkpointing at step {step}")
                ckpt.save(model, optimizer, epoch, step)
    
    # Final checkpoint
    final_path = ckpt.save(model, optimizer, epochs, 0)
    print(f"Training completed. Final checkpoint: {final_path}")
    
    return {
        'status': 'completed',
        'checkpoint': final_path,
        'device': str(device),
        'final_loss': loss.item()
    }


def main():
    print("🚀 Starting checkpointed GPU training example...")
    
    # Create client
    client = CumulusClient("http://localhost:8081")
    
    # Start training
    print("Starting training...")
    result = client.run(
        func=train_model_with_checkpoints,
        gpu_memory=0.8,
        duration=300,
        requirements=["torch", "torchvision"]
    )
    
    print(f"Training result: {result}")
    
    if result.get('status') == 'paused':
        print("Training was paused. You can resume it later.")
        print(f"Checkpoint saved at: {result['checkpoint']}")
        
        # Example of resuming (uncomment to test)
        # print("Resuming training...")
        # resumed_result = client.run(
        #     func=train_model_with_checkpoints,
        #     args=[result['checkpoint']],  # Pass checkpoint path
        #     gpu_memory=0.8,
        #     duration=300,
        #     requirements=["torch", "torchvision"]
        # )
        # print(f"Resumed training result: {resumed_result}")
    
    print("✅ Example completed!")


if __name__ == "__main__":
    main()
