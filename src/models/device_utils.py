"""
src/models/device_utils.py

Detect available hardware (GPU/CPU)
Crucial for offline inference!
"""

import torch
import psutil
from typing import Dict, Tuple
from pathlib import Path


class DeviceUtils:
    """Utilities for hardware detection and management"""
    
    @staticmethod
    def get_device_info() -> Dict:
        """
        Get complete hardware information
        
        Returns:
            Dict with device capabilities
        """
        info = {
            'device': 'cpu',
            'gpu_available': False,
            'gpu_name': None,
            'gpu_memory_total': 0,
            'gpu_memory_free': 0,
            'cpu_cores': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
            'ram_available': psutil.virtual_memory().available / (1024**3),
            'cuda_version': None,
            'pytorch_version': torch.__version__
        }
        
        # Check for GPU
        if torch.cuda.is_available():
            info['device'] = 'cuda'
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_memory_free'] = torch.cuda.mem_get_info()[0] / (1024**3)
            info['cuda_version'] = torch.version.cuda
        
        return info
    
    @staticmethod
    def print_device_info():
        """Print device info in readable format"""
        info = DeviceUtils.get_device_info()
        
        print("=" * 70)
        print("💻 HARDWARE CONFIGURATION")
        print("=" * 70)
        
        print(f"\n🖥️  CPU:")
        print(f"   Cores: {info['cpu_cores']}")
        print(f"   Usage: {info['cpu_percent']}%")
        print(f"   RAM: {info['ram_available']:.1f} GB / {info['ram_total']:.1f} GB")
        
        if info['gpu_available']:
            print(f"\n🎮 GPU:")
            print(f"   Device: {info['gpu_name']}")
            print(f"   Memory: {info['gpu_memory_free']:.1f} GB / {info['gpu_memory_total']:.1f} GB")
            print(f"   CUDA: {info['cuda_version']}")
        else:
            print(f"\n⚠️  GPU: NOT AVAILABLE (CPU mode)")
        
        print(f"\n📦 PyTorch: {info['pytorch_version']}")
        print("=" * 70 + "\n")
        
        return info
    
    @staticmethod
    def get_optimal_device() -> str:
        """
        Determine optimal device for inference
        
        Returns:
            'cuda' if GPU available, else 'cpu'
        """
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    @staticmethod
    def get_memory_usage() -> Dict:
        """Get current memory usage"""
        memory = {
            'ram_used_gb': (psutil.virtual_memory().used) / (1024**3),
            'ram_available_gb': (psutil.virtual_memory().available) / (1024**3),
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            memory['gpu_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            memory['gpu_percent'] = (memory['gpu_used_gb'] / 
                                    (torch.cuda.get_device_properties(0).total_memory / (1024**3))) * 100
        
        return memory
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    # Display hardware info
    DeviceUtils.print_device_info()
    
    # Get memory usage
    memory = DeviceUtils.get_memory_usage()
    print("Memory Usage:")
    for key, value in memory.items():
        print(f"  {key}: {value}")