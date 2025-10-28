#!/usr/bin/env python3
"""
Comprehensive checkpointing test for Cumulus SDK (moved under tests/).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sdk.client import CumulusClient


def comprehensive_checkpointing_test():
    """Test comprehensive checkpointing functionality."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from runtime import Checkpointer, should_pause, job_dir, save_on_interrupt
    import os
    import json
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    ckpt = Checkpointer()
    
    epoch = 0
    step = 0
    with save_on_interrupt(ckpt, model, optimizer, lambda: epoch, lambda: step):
        for epoch in range(10):
            for step, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/10, Step {step+1}, Loss: {loss.item():.4f}")
                
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
                
                if step == 30 and epoch == 0:
                    print("🛑 Automatically triggering pause after 30 steps for testing...")
                    control_path = os.path.join(job_dir(), 'control.json')
                    with open(control_path, 'w') as f:
                        json.dump({'pause': True}, f)
                    print(f"Created pause signal at: {control_path}")
                
                if ckpt.time_to_checkpoint(step, every_steps=50):
                    print(f"💾 Periodic checkpoint at Epoch {epoch+1}, Step {step+1}")
                    ckpt.save(model, optimizer, epoch, step)
    
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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from runtime import Checkpointer, should_pause, save_on_interrupt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    ckpt = Checkpointer()
    ckpt.path = checkpoint_path
    
    if ckpt.exists():
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        state = ckpt.load(model, optimizer)
        start_epoch = int(state['epoch'])
        start_step = int(state['step'])
        print(f"🔄 Resuming training from Epoch {start_epoch+1}, Step {start_step+1}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return {'status': 'error', 'message': 'Checkpoint not found'}
    
    epoch = start_epoch
    step = start_step
    with save_on_interrupt(ckpt, model, optimizer, lambda: epoch, lambda: step):
        for epoch in range(start_epoch, 10):
            step = 0 if epoch > start_epoch else start_step
            for i, (data, target) in enumerate(dataloader):
                if i < step:
                    continue
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/10, Step {step+1}, Loss: {loss.item():.4f}")
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
                if ckpt.time_to_checkpoint(step, every_steps=50):
                    print(f"💾 Periodic checkpoint at Epoch {epoch+1}, Step {step+1}")
                    ckpt.save(model, optimizer, epoch, step)
                step += 1
    
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
    client = CumulusClient("http://localhost:8080")
    print("\n📋 Test 1: Basic checkpointing with automatic pause")
    result = client.run(
        func=comprehensive_checkpointing_test,
        gpu_memory=0.8,
        duration=300,
        requirements=["torch", "torchvision"]
    )
    print(f"Training result: {result}")
    if result.get('status') == 'paused':
        print("\n📋 Test 2: Resume from checkpoint")
        resume_result = client.run(
            func=resume_from_checkpoint_test,
            args=[result['checkpoint']],
            gpu_memory=0.8,
            duration=300,
            requirements=["torch", "torchvision"]
        )
        print(f"Resume result: {resume_result}")


if __name__ == "__main__":
    main()


