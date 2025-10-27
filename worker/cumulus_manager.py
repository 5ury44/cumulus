"""
CumulusManager - Manages Cumulus GPU partitions for remote execution
"""

import subprocess
import json
import time
from typing import List, Dict, Optional, Any
import os


class CumulusManager:
    """
    Manages Cumulus GPU partitions for remote code execution.
    """
    
    def __init__(self, cumulus_path: str = "/usr/local/bin/chronos_cli"):
        self.cumulus_path = cumulus_path
        self.available = self._check_cumulus_availability()
        self.active_partitions: Dict[str, Dict] = {}
    
    def _check_cumulus_availability(self) -> bool:
        """Check if Cumulus is available and working."""
        try:
            # Set library path for Chronos
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')
            
            result = subprocess.run(
                [self.cumulus_path, "stats"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if Chronos is available."""
        return self.available
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available GPU devices."""
        if not self.available:
            return []
        
        try:
            result = subprocess.run(
                [self.cumulus_path, "stats"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            # Parse output to extract device information
            devices = []
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'Device' in line and ':' in line:
                    # Extract device info
                    parts = line.split(':')
                    if len(parts) >= 2:
                        device_id = parts[0].split()[-1]
                        device_name = parts[1].strip()
                        
                        devices.append({
                            'id': device_id,
                            'name': device_name,
                            'type': 'GPU' if 'GPU' in line else 'CPU'
                        })
            
            return devices
            
        except Exception as e:
            print(f"Error getting devices: {e}")
            return []
    
    def create_partition(self, 
                        device: int = 0, 
                        memory_fraction: float = 0.5, 
                        duration: int = 3600) -> str:
        """
        Create a Chronos GPU partition.
        
        Args:
            device: GPU device index
            memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
            duration: Partition duration in seconds
            
        Returns:
            Partition ID
        """
        if not self.available:
            raise RuntimeError("Chronos is not available")
        
        try:
            # Set library path for Chronos
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')
            
            # Create partition
            result = subprocess.run(
                [self.cumulus_path, "create", str(device), str(memory_fraction), str(duration)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create partition: {result.stderr}")
            
            # Extract partition ID from output
            partition_id = self._extract_partition_id(result.stdout)
            
            if not partition_id:
                raise RuntimeError("Could not extract partition ID from Chronos output")
            
            # Store partition info
            self.active_partitions[partition_id] = {
                'device': device,
                'memory_fraction': memory_fraction,
                'duration': duration,
                'created_at': time.time()
            }
            
            return partition_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Chronos partition: {str(e)}")
    
    def release_partition(self, partition_id: str) -> bool:
        """
        Release a Chronos partition.
        
        Args:
            partition_id: ID of the partition to release
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            return False
        
        try:
            # Set library path for Chronos
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')
            
            # Release partition
            result = subprocess.run(
                [self.cumulus_path, "release", partition_id],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            # Remove from active partitions
            if partition_id in self.active_partitions:
                del self.active_partitions[partition_id]
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error releasing partition {partition_id}: {e}")
            return False
    
    def list_partitions(self) -> List[Dict]:
        """List all active partitions."""
        if not self.available:
            return []
        
        try:
            result = subprocess.run(
                [self.cumulus_path, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            # Parse partition list
            partitions = []
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'partition_' in line:
                    # Extract partition info
                    parts = line.split()
                    if len(parts) >= 2:
                        partition_id = parts[0]
                        device_info = parts[1] if len(parts) > 1 else "Unknown"
                        
                        partitions.append({
                            'id': partition_id,
                            'device': device_info,
                            'status': 'active'
                        })
            
            return partitions
            
        except Exception as e:
            print(f"Error listing partitions: {e}")
            return []
    
    def get_available_memory(self, device: int = 0) -> float:
        """
        Get available memory percentage for a device.
        
        Args:
            device: GPU device index
            
        Returns:
            Available memory percentage (0.0-1.0)
        """
        if not self.available:
            return 0.0
        
        try:
            result = subprocess.run(
                [self.cumulus_path, "available", str(device)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return 0.0
            
            # Parse percentage from output
            try:
                percentage = float(result.stdout.strip())
                return percentage / 100.0  # Convert to 0.0-1.0 range
            except ValueError:
                return 0.0
                
        except Exception as e:
            print(f"Error getting available memory: {e}")
            return 0.0
    
    def _extract_partition_id(self, output: str) -> Optional[str]:
        """Extract partition ID from Chronos output."""
        lines = output.split('\n')
        
        for line in lines:
            if 'partition_' in line:
                # Look for pattern like "partition_0001"
                words = line.split()
                for word in words:
                    if word.startswith('partition_'):
                        return word
        
        return None
    
    def cleanup_expired_partitions(self):
        """Clean up expired partitions from tracking."""
        current_time = time.time()
        expired = []
        
        for partition_id, info in self.active_partitions.items():
            created_at = info['created_at']
            duration = info['duration']
            
            if current_time - created_at > duration:
                expired.append(partition_id)
        
        for partition_id in expired:
            self.release_partition(partition_id)
    
    def get_partition_info(self, partition_id: str) -> Optional[Dict]:
        """Get information about a specific partition."""
        return self.active_partitions.get(partition_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Chronos statistics."""
        if not self.available:
            return {
                'available': False,
                'active_partitions': 0,
                'devices': []
            }
        
        try:
            # Get device stats
            devices = self.get_available_devices()
            
            # Get partition list
            partitions = self.list_partitions()
            
            return {
                'available': True,
                'active_partitions': len(partitions),
                'devices': devices,
                'partitions': partitions
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'active_partitions': 0,
                'devices': []
            }
