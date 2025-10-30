#!/usr/bin/env python3
"""
Test job for Cumulus container
"""

import time
import os


def main():
    """Main function that will be executed by the job runner."""
    print("🧪 Test job starting...")
    
    # Get environment info
    job_id = os.environ.get("JOB_ID", "test-job")
    partition_id = os.environ.get("CHRONOS_PARTITION_ID", "test-partition")
    
    print(f"Job ID: {job_id}")
    print(f"Partition ID: {partition_id}")
    
    # Do some simple computation
    result = 0
    for i in range(1000000):
        result += i * 2
    
    # Simulate some work
    time.sleep(1)
    
    print("🧪 Test job completed!")
    
    return {
        "message": "Test job completed successfully",
        "computation_result": result,
        "job_id": job_id,
        "partition_id": partition_id
    }


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    result = main()
    print(f"Result: {result}")