#!/usr/bin/env python3
"""
Demonstration of pause/resume functionality with Cumulus SDK
"""

import sys
import os
import time
sys.path.insert(0, '/tmp/cumulus')

from sdk.client import CumulusClient


def long_running_training(epochs: int = 10, steps_per_epoch: int = 100):
    """A long-running training function that can be paused and resumed."""
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
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Checkpointing
    ckpt = Checkpointer()
    
    # Training loop
    for epoch in range(epochs):
        for step, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss.item():.4f}")
            
            # Check for pause signal every few steps
            if step % 20 == 0 and should_pause():
                print("Pause signal detected, saving checkpoint...")
                ckpt.save(model, optimizer, epoch, step)
                return {
                    'status': 'paused',
                    'checkpoint': ckpt.path,
                    'epoch': epoch,
                    'step': step,
                    'device': str(device),
                    'message': 'Training paused by user'
                }
            
            # Periodic checkpointing
            if ckpt.time_to_checkpoint(step, every_steps=50):
                print(f"Checkpointing at epoch {epoch+1}, step {step+1}")
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
    print("🚀 Starting pause/resume demonstration...")
    
    # Create client
    client = CumulusClient("http://localhost:8081")
    
    # Start training
    print("Starting long-running training...")
    job = client.run_async(
        func=long_running_training,
        gpu_memory=0.8,
        duration=600,  # 10 minutes
        requirements=["torch", "torchvision"]
    )
    
    print(f"Job started with ID: {job.job_id}")
    
    # Wait a bit, then pause
    print("Waiting 30 seconds before pausing...")
    time.sleep(30)
    
    # Check job status
    status = job.status()
    print(f"Current job status: {status}")
    
    if status == "running":
        print("Pausing job...")
        pause_result = job.pause()
        print(f"Pause result: {pause_result}")
        
        # Wait a bit more
        print("Waiting 10 seconds...")
        time.sleep(10)
        
        # Check checkpoints
        checkpoints = job.get_checkpoints()
        print(f"Available checkpoints: {len(checkpoints)}")
        for i, ckpt in enumerate(checkpoints[:3]):  # Show first 3
            print(f"  {i+1}. {ckpt['filename']} - Epoch {ckpt['epoch']}, Step {ckpt['step']}")
        
        # Resume the job
        print("Resuming job...")
        resume_result = job.resume()
        print(f"Resume result: {resume_result}")
        
        # Wait for completion
        print("Waiting for job to complete...")
        result = job.result(timeout=300)  # 5 minute timeout
        print(f"Final result: {result}")
    else:
        print(f"Job status is {status}, cannot pause")
    
    print("✅ Demonstration completed!")


if __name__ == "__main__":
    main()
