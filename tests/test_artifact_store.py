#!/usr/bin/env python3
"""
Test generic artifact store functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sdk.client import CumulusClient


def test_artifact_store():
    """Test that users can save/load their own artifacts."""
    from runtime import get_artifact_store
    import json
    
    # Get artifact store (will use LocalArtifactStore since no S3 configured)
    store = get_artifact_store(use_distributed=False)
    
    # Test 1: Save bytes as artifact
    print("📦 Test 1: Saving bytes as artifact")
    data = b"Hello from artifact store!"
    info = store.save_artifact_bytes("hello.txt", data)
    print(f"✅ Saved to: {info['local_path']}")
    
    # Test 2: Save JSON as bytes
    print("\n📦 Test 2: Saving JSON data")
    json_data = json.dumps({"test": "data", "value": 42}).encode('utf-8')
    info2 = store.save_artifact_bytes("test.json", json_data)
    print(f"✅ Saved to: {info2['local_path']}")
    
    # Test 3: Load artifact as bytes
    print("\n📦 Test 3: Loading artifact as bytes")
    bytes_data = store.load_artifact_bytes("hello.txt")
    print(f"✅ Loaded: {bytes_data.decode()}")
    
    # Test 4: List artifacts
    print("\n📦 Test 4: Listing artifacts")
    artifacts = store.list_artifacts()
    print(f"✅ Found {len(artifacts)} artifacts:")
    for art in artifacts:
        print(f"  - {art['name']} ({art['source']})")
    
    return {
        'status': 'completed',
        'num_artifacts': len(artifacts),
        'message': 'Artifact store tests passed!'
    }


if __name__ == "__main__":
    print("🧪 Testing Generic Artifact Store")
    print("=" * 60)
    
    client = CumulusClient("http://localhost:8080")
    
    result = client.run(
        func=test_artifact_store,
        gpu_memory=0.2,
        duration=60,
        requirements=[]
    )
    
    print(f"\n📊 Test Result: {result}")

