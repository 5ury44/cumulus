#!/usr/bin/env python3
"""
Complete neural network training and resume test in a single job
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sdk.client import CumulusClient
import time


def complete_nn_training_and_resume():
    """Complete neural network training with checkpointing and resume in one job."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from runtime import get_checkpointer, save_checkpoint, load_checkpoint
    
    print("🚀 Complete Neural Network Training and Resume Test")
    print("=" * 60)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create synthetic data
    X = torch.randn(1000, 784).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)
    
    # Get checkpointer (for diagnostics only)
    checkpointer = get_checkpointer()
    print(f"📦 Checkpointer type: {type(checkpointer).__name__}")
    if hasattr(checkpointer, 's3_client') and checkpointer.s3_client:
        print("✅ S3 client is available!")
        print(f"📦 S3 Bucket: {checkpointer.s3_bucket}")
    
    # Training loop - first 3 epochs
    epochs = 5
    batch_size = 64
    
    print(f"\n📋 Phase 1: Training first 3 epochs")
    epoch = 0
    batch_idx = 0
    num_batches = len(X) // batch_size
    for epoch in range(3):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            data = X[start_idx:end_idx]
            target = y[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = epoch_loss / num_batches
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/5: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save checkpoint after 3 epochs
    print(f"\n💾 Saving checkpoint after epoch 3...")
    checkpoint_info = save_checkpoint(model, optimizer, epoch=2, step=num_batches-1, framework="pytorch")
    print(f"✅ Checkpoint saved: {checkpoint_info}")
    
    # Resume training from checkpoint
    print(f"\n📋 Phase 2: Resuming from checkpoint")
    try:
        print("📥 Loading checkpoint...")
        checkpoint_path = checkpoint_info.get('s3_key') or checkpoint_info['local_path']
        load_res = load_checkpoint(model, optimizer, checkpoint_path=checkpoint_path, framework="pytorch")
        state = load_res['state'] if isinstance(load_res, dict) and 'state' in load_res else load_res
        start_epoch = state['epoch']
        start_batch = state['step']
        print(f"✅ Checkpoint loaded successfully!")
        print(f"📊 Resuming from epoch {start_epoch+1}, batch {start_batch+1}")
        
        # Continue training for remaining epochs
        print(f"\n📋 Phase 3: Continuing training for remaining epochs")
        epoch = start_epoch
        batch_idx = 0
        for epoch in range(start_epoch + 1, epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            num_batches = len(X) // batch_size
            
            batch_start = 0 if epoch > start_epoch else start_batch + 1
            
            for batch_idx in range(batch_start, num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                data = X[start_idx:end_idx]
                target = y[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / max(1, num_batches - batch_start)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/5: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Final checkpoint using unified API
        final_checkpoint = save_checkpoint(model, optimizer, epoch=epochs-1, step=num_batches-1, framework="pytorch")
        print(f"\n✅ Final checkpoint saved: {final_checkpoint}")
        
        return {
            'status': 'completed',
            'final_loss': avg_loss,
            'final_accuracy': accuracy,
            'checkpoint': final_checkpoint,
            'device': str(device)
        }
        
    except Exception as e:
        print(f"❌ Failed to resume training: {e}")
        return {'status': 'error', 'message': str(e)}


def main():
    print("🧪 Complete Neural Network Training and Resume Test")
    print("=" * 60)
    
    # Create client
    client = CumulusClient("http://localhost:8080")
    
    # Run complete test
    result = client.run(
        func=complete_nn_training_and_resume,
        gpu_memory=0.4,
        duration=600,
        requirements=["torch", "boto3"]
    )
    
    print(f"\n📊 Test Result:")
    print(f"Status: {result['status']}")
    if result['status'] == 'completed':
        print(f"Final Loss: {result['final_loss']:.4f}")
        print(f"Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"Device: {result['device']}")
        print(f"Final Checkpoint: {result['checkpoint']}")
        print("✅ Complete neural network training and resume test successful!")
    else:
        print(f"❌ Test failed: {result}")


if __name__ == "__main__":
    main()
