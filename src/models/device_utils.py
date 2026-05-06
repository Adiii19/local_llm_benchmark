import torch
import psutil
from typing import Dict


class DeviceUtils:
  
    
    @staticmethod
    def get_device_info() -> Dict:
        
        
        info = {
            'device': 'cpu',
            'gpu_available': False,
            'gpu_name': None,
            'gpu_memory_total': 0,
            'gpu_memory_free': 0,
            'gpu_count': 0,
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3),
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_percent': psutil.virtual_memory().percent,
            'pytorch_version': torch.__version__,
            'cuda_available': False,
            'cuda_version': None
        }
        
        # Check for GPU availability
        if torch.cuda.is_available():
            info['device'] = 'cuda'
            info['gpu_available'] = True
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['gpu_memory_free'] = torch.cuda.mem_get_info()[0] / (1024**3)
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
        
        return info
    
    @staticmethod
    def print_device_info() -> Dict:
        
        info = DeviceUtils.get_device_info()
        
        print("=" * 80)
        print("💻 SYSTEM INFORMATION & DEVICE DETECTION")
        print("=" * 80)
        
        print(f"\n🎯 DEVICE SELECTION:")
        if info['gpu_available']:
            print(f"   ✅ GPU DETECTED - Will use: {info['gpu_name']}")
        else:
            print(f"   ℹ️  No GPU found - Will use: CPU")
        
        print(f"\n🖥️  CPU:")
        print(f"   Physical Cores: {info['cpu_cores']}")
        print(f"   Logical Cores (Threads): {info['cpu_threads']}")
        print(f"   Frequency: {info['cpu_freq']:.1f} MHz")
        print(f"   Current Usage: {info['cpu_percent']:.1f}%")
        

        print(f"\n💾 RAM (System Memory):")
        print(f"   Total: {info['ram_total_gb']:.1f} GB")
        print(f"   Used: {info['ram_used_gb']:.1f} GB")
        print(f"   Available: {info['ram_available_gb']:.1f} GB")
        print(f"   Usage: {info['ram_percent']:.1f}%")
        
        if info['gpu_available']:
            print(f"\n🎮 GPU (NVIDIA CUDA):")
            print(f"   Device Name: {info['gpu_name']}")
            print(f"   GPU Count: {info['gpu_count']}")
            print(f"   Total Memory: {info['gpu_memory_total']:.1f} GB")
            print(f"   Free Memory: {info['gpu_memory_free']:.1f} GB")
            print(f"   CUDA Version: {info['cuda_version']}")
        else:
            print(f"\n🎮 GPU:")
            print(f"   Status: NOT AVAILABLE")
            print(f"   Note: 1B models work great on CPU!")
        
        
        print(f"\n📦 Framework:")
        print(f"   PyTorch: {info['pytorch_version']}")
        
        print(f"\n✅ RECOMMENDATIONS FOR 1B MODELS:")
        if info['ram_available_gb'] >= 2:
            print(f"   ✓ You have {info['ram_available_gb']:.1f}GB free RAM")
            print(f"   ✓ All 1B models will work perfectly!")
        else:
            print(f"   ⚠️  Only {info['ram_available_gb']:.1f}GB free RAM")
            print(f"   💡 Close other programs before running")
        
        print("=" * 80 + "\n")
        
        return info
    
    @staticmethod
    def get_optimal_device() -> str:
       
        if torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                return 'cuda'
            except Exception as e:
                print(f"⚠️  CUDA available but not working: {e}")
                print(f"   Falling back to CPU")
                return 'cpu'
        
        return 'cpu'
    
    @staticmethod
    def get_memory_usage() -> Dict:
        
        memory = {
            'ram_used_gb': (psutil.virtual_memory().used) / (1024**3),
            'ram_available_gb': (psutil.virtual_memory().available) / (1024**3),
            'ram_percent': psutil.virtual_memory().percent,
            'device': 'unknown'
        }
        
        if torch.cuda.is_available():
            memory['device'] = 'cuda'
            memory['gpu_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            memory['gpu_percent'] = (
                (memory['gpu_used_gb'] / 
                 (torch.cuda.get_device_properties(0).total_memory / (1024**3))) * 100
            )
        else:
            memory['device'] = 'cpu'
        
        return memory
    
    @staticmethod
    def clear_gpu_memory():
       
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("✓ GPU memory cleared")
        else:
            print("ℹ️  No GPU to clear (CPU only mode)")
    
    @staticmethod
    def print_memory_usage():
        
        memory = DeviceUtils.get_memory_usage()
        
        print("\n" + "="*70)
        print("📊 MEMORY USAGE")
        print("="*70)
        
        print(f"\n🖥️  CPU RAM:")
        print(f"   Used: {memory['ram_used_gb']:.1f} GB")
        print(f"   Available: {memory['ram_available_gb']:.1f} GB")
        print(f"   Usage: {memory['ram_percent']:.1f}%")
        
        if 'gpu_used_gb' in memory:
            print(f"\n🎮 GPU Memory:")
            print(f"   Used: {memory['gpu_used_gb']:.1f} GB")
            print(f"   Reserved: {memory['gpu_reserved_gb']:.1f} GB")
            print(f"   Usage: {memory['gpu_percent']:.1f}%")
        else:
            print(f"\n🎮 GPU Memory:")
            print(f"   Not available (CPU mode)")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING DEVICE DETECTION")
    print("="*70)
    
    info = DeviceUtils.print_device_info()
    
    optimal_device = DeviceUtils.get_optimal_device()
    print(f"✅ Optimal Device: {optimal_device.upper()}")
    
    DeviceUtils.print_memory_usage()