#!/usr/bin/env python3
"""
Comprehensive unified checkpointing test from local machine to remote GPU server.
Tests all frameworks with S3 integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import CumulusClient
import json

def test_unified_with_s3():
    """Test unified checkpointing for all frameworks with S3 integration."""
    results = []
    
    # PyTorch test with S3
    def test_pytorch_s3():
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sdk.distributed_checkpointer import DistributedCheckpointer
        
        model = nn.Linear(10, 1).cuda() if torch.cuda.is_available() else nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters())
        
        ckpt = DistributedCheckpointer(job_id="remote-pytorch-s3", s3_bucket="cumulus-jobs", local_dir="/tmp/remote-s3")
        save_info = ckpt.save_checkpoint(model, optimizer, epoch=1, step=10, framework="pytorch")
        
        # Verify S3 upload
        assert save_info["s3_key"] is not None, "S3 key should be present"
        
        # Remove local file to force S3 download
        import os
        os.remove(save_info["local_path"])
        
        loaded_model = nn.Linear(10, 1).cuda() if torch.cuda.is_available() else nn.Linear(10, 1)
        loaded_optimizer = optim.Adam(loaded_model.parameters())
        load_result = ckpt.load_checkpoint(loaded_model, loaded_optimizer, 
                                         checkpoint_path=f"s3://{ckpt.s3_bucket}/{save_info['s3_key']}", 
                                         framework="pytorch")
        
        return {"framework": "pytorch", "status": "ok", "s3": True, "save_path": save_info["local_path"]}
    
    # TensorFlow test with S3
    def test_tensorflow_s3():
        import tensorflow as tf
        from sdk.distributed_checkpointer import DistributedCheckpointer
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        ckpt = DistributedCheckpointer(job_id="remote-tensorflow-s3", s3_bucket="cumulus-jobs", local_dir="/tmp/remote-s3")
        save_info = ckpt.save_checkpoint(model, epoch=1, step=5, framework="tensorflow")
        
        # Verify S3 upload
        assert save_info["s3_key"] is not None, "S3 key should be present"
        
        # Remove local file to force S3 download
        import os
        os.remove(save_info["local_path"])
        
        loaded_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1)
        ])
        load_result = ckpt.load_checkpoint(loaded_model, 
                                         checkpoint_path=f"s3://{ckpt.s3_bucket}/{save_info['s3_key']}", 
                                         framework="tensorflow")
        
        return {"framework": "tensorflow", "status": "ok", "s3": True, "save_path": save_info["local_path"]}
    
    # scikit-learn test with S3
    def test_sklearn_s3():
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        from sdk.distributed_checkpointer import DistributedCheckpointer
        
        X = np.random.randn(50, 10)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(max_iter=100).fit(X, y)
        
        ckpt = DistributedCheckpointer(job_id="remote-sklearn-s3", s3_bucket="cumulus-jobs", local_dir="/tmp/remote-s3")
        save_info = ckpt.save_checkpoint(model, epoch=0, step=1, framework="sklearn")
        
        # Verify S3 upload
        assert save_info["s3_key"] is not None, "S3 key should be present"
        
        # Remove local file to force S3 download
        import os
        os.remove(save_info["local_path"])
        
        load_result = ckpt.load_checkpoint(checkpoint_path=f"s3://{ckpt.s3_bucket}/{save_info['s3_key']}", framework="sklearn")
        loaded_model = load_result["state"]
        
        # Verify predictions match
        pred1 = model.predict(X[:5])
        pred2 = loaded_model.predict(X[:5])
        assert (pred1 == pred2).all(), "Predictions should match"
        
        return {"framework": "sklearn", "status": "ok", "s3": True, "save_path": save_info["local_path"]}
    
    # XGBoost test with S3
    def test_xgboost_s3():
        import xgboost as xgb
        import numpy as np
        from sdk.distributed_checkpointer import DistributedCheckpointer
        
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {"objective": "binary:logistic", "tree_method": "hist"}
        booster = xgb.train(params, dtrain, num_boost_round=10)
        
        ckpt = DistributedCheckpointer(job_id="remote-xgboost-s3", s3_bucket="cumulus-jobs", local_dir="/tmp/remote-s3")
        save_info = ckpt.save_checkpoint(booster, epoch=0, step=1, framework="xgboost")
        
        # Verify S3 upload
        assert save_info["s3_key"] is not None, "S3 key should be present"
        
        # Remove local file to force S3 download
        import os
        os.remove(save_info["local_path"])
        
        load_result = ckpt.load_checkpoint(checkpoint_path=f"s3://{ckpt.s3_bucket}/{save_info['s3_key']}", framework="xgboost")
        loaded_booster = load_result["state"]
        
        return {"framework": "xgboost", "status": "ok", "s3": True, "save_path": save_info["local_path"]}
    
    # LightGBM test with S3
    def test_lightgbm_s3():
        import lightgbm as lgb
        import numpy as np
        from sdk.distributed_checkpointer import DistributedCheckpointer
        
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        dataset = lgb.Dataset(X, label=y, free_raw_data=True)
        
        params = {"objective": "binary", "verbosity": -1}
        booster = lgb.train(params, dataset, num_boost_round=10)
        
        ckpt = DistributedCheckpointer(job_id="remote-lightgbm-s3", s3_bucket="cumulus-jobs", local_dir="/tmp/remote-s3")
        save_info = ckpt.save_checkpoint(booster, epoch=0, step=1, framework="lightgbm")
        
        # Verify S3 upload
        assert save_info["s3_key"] is not None, "S3 key should be present"
        
        # Remove local file to force S3 download
        import os
        os.remove(save_info["local_path"])
        
        load_result = ckpt.load_checkpoint(checkpoint_path=f"s3://{ckpt.s3_bucket}/{save_info['s3_key']}", framework="lightgbm")
        loaded_booster = load_result["state"]
        
        return {"framework": "lightgbm", "status": "ok", "s3": True, "save_path": save_info["local_path"]}
    
    # Run all S3 tests
    tests = [test_pytorch_s3, test_tensorflow_s3, test_sklearn_s3, test_xgboost_s3, test_lightgbm_s3]
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            results.append({"framework": test_func.__name__, "status": "error", "s3": True, "error": str(e)})
    
    return results

if __name__ == "__main__":
    # Connect to remote server
    client = CumulusClient("http://localhost:8083")
    
    print("🚀 Testing unified checkpointing with S3 integration for all frameworks on remote GPU server...")
    
    # Run the test remotely
    results = client.run(
        func=test_unified_with_s3,
        gpu_memory=0.8,
        duration=600,
        requirements=["torch", "tensorflow", "scikit-learn", "xgboost", "lightgbm"]
    )
    
    print("📊 Results:")
    print(json.dumps(results, indent=2))
    
    # Summary
    passed = sum(1 for r in results if r.get("status") == "ok")
    total = len(results)
    s3_tests = sum(1 for r in results if r.get("s3") == True and r.get("status") == "ok")
    
    print(f"\n✅ {passed}/{total} frameworks passed unified checkpointing test")
    print(f"🌐 {s3_tests}/{total} frameworks passed S3 integration test")
    
    if passed == total:
        print("\n🎉 All frameworks working with unified checkpointing and S3!")
    else:
        print(f"\n⚠️  {total - passed} frameworks had issues")
