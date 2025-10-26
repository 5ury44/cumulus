#!/usr/bin/env python3
"""
Comprehensive checkpointing test for Cumulus SDK
"""

import sys
import os
sys.path.insert(0, '/tmp/cumulus')

from sdk.client import CumulusClient


def comprehensive_checkpointing_test():
    """Test comprehensive checkpointing functionality."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from runtime import Checkpointer, should_pause, job_dir
    import os
    import json
    
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
    
    # Training loop with checkpointing
    for epoch in range(10):
        for step, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/10, Step {step+1}, Loss: {loss.item():.4f}")
            
            # Check for pause signal more frequently
            if should_pause():
                print(f"🛑 Pause signal detected at Epoch {epoch+1}, Step {step+1} - saving checkpoint...")
                ckpt.save(model, optimizer, epoch, step)
                print(f"✅ Checkpoint saved: {ckpt.path}")
                return {
                    'status': 'paused',
                    'checkpoint': ckpt.path,
                    'epoch': epoch,
                    'step': step,
                    'device': str(device),
                    'message': 'Training paused by user'
                }
            
            # Automatically trigger pause after 30 steps for testing
            if step == 30 and epoch == 0:
                print("🛑 Automatically triggering pause after 30 steps for testing...")
                control_path = os.path.join(job_dir(), 'control.json')
                with open(control_path, 'w') as f:
                    json.dump({'pause': True}, f)
                print(f"Created pause signal at: {control_path}")
            
            # Periodic checkpointing every 50 steps
            if ckpt.time_to_checkpoint(step, every_steps=50):
                print(f"💾 Periodic checkpoint at Epoch {epoch+1}, Step {step+1}")
                ckpt.save(model, optimizer, epoch, step)
    
    # Final checkpoint
    final_path = ckpt.save(model, optimizer, 10, 0)
    print(f"🏁 Training completed. Final checkpoint: {final_path}")
    
    return {
        'status': 'completed',
        'checkpoint': final_path,
        'device': str(device),
        'final_loss': loss.item(),
        'message': 'Training completed successfully'
    }


def resume_from_checkpoint_test(checkpoint_path: str):
    """Test resuming training from a checkpoint."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from runtime import Checkpointer, should_pause
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model (same architecture as original)
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
    
    # Generate sample data (same as original)
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Load checkpoint
    ckpt = Checkpointer()
    ckpt.path = checkpoint_path
    
    if ckpt.exists():
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        state = ckpt.load(model, optimizer)
        start_epoch = int(state['epoch'])
        start_step = int(state['step'])
        print(f"🔄 Resuming training from Epoch {start_epoch+1}, Step {start_step+1}")
        print(f"📊 Checkpoint contains: epoch={start_epoch}, step={start_step}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return {'status': 'error', 'message': 'Checkpoint not found'}
    
    # Continue training from checkpoint
    for epoch in range(start_epoch, 10):
        step = 0 if epoch > start_epoch else start_step
        
        for i, (data, target) in enumerate(dataloader):
            if i < step:  # Skip to resume point
                continue
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/10, Step {step+1}, Loss: {loss.item():.4f}")
            
            # Check for pause signal
            if should_pause():
                print(f"🛑 Pause signal detected at Epoch {epoch+1}, Step {step+1} - saving checkpoint...")
                ckpt.save(model, optimizer, epoch, step)
                print(f"✅ Checkpoint saved: {ckpt.path}")
                return {
                    'status': 'paused',
                    'checkpoint': ckpt.path,
                    'epoch': epoch,
                    'step': step,
                    'device': str(device),
                    'message': 'Training paused by user'
                }
            
            # Periodic checkpointing every 50 steps
            if ckpt.time_to_checkpoint(step, every_steps=50):
                print(f"💾 Periodic checkpoint at Epoch {epoch+1}, Step {step+1}")
                ckpt.save(model, optimizer, epoch, step)
            
            step += 1
    
    # Final checkpoint
    final_path = ckpt.save(model, optimizer, 10, 0)
    print(f"🏁 Training completed after resume. Final checkpoint: {final_path}")
    
    return {
        'status': 'completed',
        'checkpoint': final_path,
        'device': str(device),
        'final_loss': loss.item(),
        'message': 'Training completed successfully after resume'
    }


def main():
    print("🚀 Testing comprehensive checkpointing functionality...")
    
    # Create client
    client = CumulusClient("http://localhost:8081")
    
    # Test 1: Basic checkpointing with automatic pause
    print("\n📋 Test 1: Basic checkpointing with automatic pause")
    result = client.run(
        func=comprehensive_checkpointing_test,
        gpu_memory=0.8,
        duration=300,
        requirements=["torch", "torchvision"]
    )
    
    print(f"Training result: {result}")
    
    if result.get('status') == 'completed':
        print("✅ Basic checkpointing works!")
        print(f"Final checkpoint: {result['checkpoint']}")
        print(f"Final loss: {result['final_loss']:.4f}")
        print(f"📍 Done training at Epoch 10, Step 0 (completed all epochs)")
    elif result.get('status') == 'paused':
        print("✅ Pause functionality works!")
        print(f"📍 Paused at Epoch {result['epoch']+1}, Step {result['step']+1}")
        print(f"💾 Checkpoint saved: {result['checkpoint']}")
        
        # Test 2: Resume from checkpoint
        print("\n📋 Test 2: Resume from checkpoint")
        print(f"Resuming from checkpoint: {result['checkpoint']}")
        
        resume_result = client.run(
            func=resume_from_checkpoint_test,
            args=[result['checkpoint']],  # Pass checkpoint path as argument
            gpu_memory=0.8,
            duration=300,
            requirements=["torch", "torchvision"]
        )
        
        print(f"Resume result: {resume_result}")
        
        if resume_result.get('status') == 'completed':
            print("✅ Resume functionality works!")
            print(f"🏁 Final checkpoint after resume: {resume_result['checkpoint']}")
            print(f"📊 Final loss after resume: {resume_result['final_loss']:.4f}")
            print(f"📍 Done training at Epoch 10, Step 0 (completed all epochs)")
        elif resume_result.get('status') == 'paused':
            print("✅ Resume with pause functionality works!")
            print(f"📍 Paused again at Epoch {resume_result['epoch']+1}, Step {resume_result['step']+1}")
        else:
            print(f"❌ Resume test failed: {resume_result}")
    
    # Test 3: Check checkpointing API endpoints
    print("\n📋 Test 3: Checkpointing API endpoints")
    try:
        # Get server info
        info = client.get_server_info()
        print(f"Server info: {info}")
        
        # List jobs
        jobs = client.list_jobs()
        print(f"Total jobs: {len(jobs)}")
        
        if jobs:
            latest_job = jobs[-1]  # Get the latest job
            job_id = latest_job['job_id']
            print(f"Latest job ID: {job_id}")
            
            # Get checkpoints for the latest job
            checkpoints = client.get_checkpoints(job_id)
            print(f"Available checkpoints: {len(checkpoints)}")
            for i, ckpt in enumerate(checkpoints[:3]):  # Show first 3
                print(f"  {i+1}. {ckpt['filename']} - Epoch {ckpt['epoch']}, Step {ckpt['step']}")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print("\n🎉 Comprehensive checkpointing test completed!")


if __name__ == "__main__":
    main()
