import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Tuple, Optional
import gc
from pathlib import Path
import traceback
import os
from huggingface_hub import login as hf_login
import warnings

from .device_utils import DeviceUtils


class ModelManager:
    
    
    def __init__(self, cache_dir: str = "models/"):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = DeviceUtils.get_optimal_device()
        self.loaded_models = {}
        
        self._setup_hf_authentication()
        
        print(f"✓ ModelManager initialized")
        print(f"  Device: {self.device.upper()}")
        print(f"  Cache: {self.cache_dir}")
        print(f"  ✅ Memory Optimization: AGGRESSIVE")
    
    def _setup_hf_authentication(self):
        try:
            token = os.getenv('HUGGING_FACE_TOKEN')
            
            if token:
                print(f"✓ Using HuggingFace token from environment")
                hf_login(token=token, add_to_git_credential=False)
            else:
                hf_token_path = Path.home() / ".huggingface" / "token"
                if hf_token_path.exists():
                    print(f"✓ Found HuggingFace token")
        
        except Exception as e:
            print(f"⚠️  HuggingFace auth warning: {str(e)[:50]}")
    
    def load_model(
        self,
        config,
        quantization: Optional[str] = None,
        force_reload: bool = False
    ) -> Optional[Tuple[torch.nn.Module, AutoTokenizer]]:
       
        model_id = config.model_id
        
        # Check cache
        if model_id in self.loaded_models and not force_reload:
            print(f"✓ Using cached: {config.model_name}")
            cached = self.loaded_models[model_id]
            if cached is not None:
                return cached
        
        print(f"\n{'='*70}")
        print(f"📥 Loading {config.model_name}")
        print(f"   Model ID: {model_id}")
        print(f"   Size: {config.size}")
        print(f"   Disk: {config.disk_size_gb} GB")
        print(f"   RAM Needed: {config.min_ram_gb} GB")
        print(f"   Device: {self.device.upper()}")
        print(f"{'='*70}")
        
        print(f"\n  📍 Step 0: Memory cleanup...")
        self._aggressive_cleanup()
        
        # ✅ CHECK AVAILABLE MEMORY
        memory = DeviceUtils.get_memory_usage()
        available_ram = memory['ram_available_gb']
        needed_ram = config.min_ram_gb + 2  # +2GB safety buffer
        
        print(f"     Available RAM: {available_ram:.1f} GB")
        print(f"     Needed: {needed_ram:.1f} GB")
        
        if available_ram < needed_ram:
            print(f"\n❌ INSUFFICIENT MEMORY")
            print(f"   💡 SOLUTIONS:")
            print(f"      1. Close Chrome, VS Code, Discord")
            print(f"      2. Restart your PC")
            print(f"      3. Only run this script with nothing else")
            return None
        
        print(f"  ✓ Memory check passed")
        
        try:
            print(f"\n  📍 Step 1: Loading tokenizer...")
            tokenizer = self._load_tokenizer(model_id)
            
            if tokenizer is None:
                raise ValueError(f"Failed to load tokenizer for {model_id}")
            
            print(f"     ✓ Tokenizer loaded")
            
            self._aggressive_cleanup()
            
            print(f"\n  📍 Step 2: Configuring model loading...")
            load_kwargs = self._get_load_kwargs_aggressive()
            
            print(f"\n  📍 Step 3: Loading model...")
            print(f"     (This will take 1-2 minutes, please be patient...)")
            
            # ✅ Suppress the deprecation warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **load_kwargs
                )
            
            if model is None:
                raise ValueError(f"Failed to load model for {model_id}")
            
            print(f"     ✓ Model loaded successfully!")
            
            # Set to eval mode
            model.eval()
            
            # ✅ Move to device carefully
            try:
                print(f"     Moving to {self.device.upper()}...")
                model = model.to(self.device)
                print(f"     ✓ Moved to {self.device.upper()}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n❌ OUT OF MEMORY while moving model")
                    print(f"   Try: DistilGPT2 (82M) instead")
                    return None
                raise
            
            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Cache it
            self.loaded_models[model_id] = (model, tokenizer)
            
            print(f"\n  ✅ SUCCESS!")
            print(f"     Model: {config.model_name}")
            print(f"     Size: {config.size}")
            print(f"     Device: {self.device.upper()}")
            
            return (model, tokenizer)
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg:
                print(f"\n❌ OUT OF MEMORY")
                print(f"   Your system doesn't have enough RAM")
                print(f"   Try:")
                print(f"   1. Restart your PC")
                print(f"   2. Use DistilGPT2 instead (smallest model)")
                print(f"   3. Close all other programs")
            else:
                print(f"\n❌ RUNTIME ERROR: {str(e)[:150]}")
            
            self.loaded_models[model_id] = None
            return None
        
        except Exception as e:
            print(f"\n❌ FAILED to load {config.model_name}")
            print(f"  Error: {str(e)[:150]}")
            
            self._print_troubleshooting(model_id, str(e))
            
            self.loaded_models[model_id] = None
            return None
    
    def _get_load_kwargs_aggressive(self) -> dict:
       
        kwargs = {
            'cache_dir': self.cache_dir,
            'trust_remote_code': True,
        }
        
        if self.device == 'cuda':
            print(f"     ⚙️  GPU Configuration (8-bit Quantized)...")
            kwargs['device_map'] = 'auto'
            kwargs['torch_dtype'] = torch.float16
            kwargs['load_in_8bit'] = True
            print(f"     Applying 8-bit quantization (saves 75% memory)")
        
        else:
            print(f"     ⚙️  CPU Configuration (Memory Optimized)...")
            kwargs['device_map'] = None
            # ✅ KEY: Sequential loading doesn't load entire model at once
            kwargs['low_cpu_mem_usage'] = True
            kwargs['torch_dtype'] = torch.float32
            print(f"     Using sequential loading (low_cpu_mem_usage)")
            # ❌ NO quantization on CPU
        
        return kwargs
    
    def _aggressive_cleanup(self):
      
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Print memory after cleanup
        memory = DeviceUtils.get_memory_usage()
        print(f"     RAM available after cleanup: {memory['ram_available_gb']:.1f} GB")
    
    def _load_tokenizer(self, model_id: str) -> Optional[AutoTokenizer]:
        """Load tokenizer with fallbacks"""
        
        print(f"     Attempting to load tokenizer...")
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    use_auth_token=True
                )
            return tokenizer
        
        except Exception as e1:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
                return tokenizer
            
            except Exception as e2:
                print(f"     ❌ Failed to load tokenizer")
                print(f"     Error: {str(e2)[:100]}")
                return None
    
    def _print_troubleshooting(self, model_id: str, error_str: str):
        
        error_lower = error_str.lower()
        
        if "quantiz" in error_lower:
            print(f"  ℹ️  Quantization Error (GPU needed for 8-bit)")
            print(f"      Using CPU mode - no quantization available")
        
        elif "out of memory" in error_lower:
            print(f"  ℹ️  OUT OF MEMORY ERROR")
            print(f"      This model is too large for your RAM")
            print(f"      Try DistilGPT2 (82M) instead")
        
        elif "401" in error_lower or "unauthorized" in error_lower:
            print(f"  ℹ️  Authentication Error")
            print(f"      Run: huggingface-cli login")
        
        else:
            print(f"  ℹ️  Try these steps:")
            print(f"      1. Restart your PC")
            print(f"      2. Use smaller model (DistilGPT2)")
            print(f"      3. Update: pip install --upgrade torch transformers")
    
    def unload_model(self, model_id: str):
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            self._aggressive_cleanup()
            print(f"✓ Unloaded: {model_id}")
    
    def unload_all_models(self):
        self.loaded_models.clear()
        self._aggressive_cleanup()
        print(f"✓ All models unloaded and memory cleared")