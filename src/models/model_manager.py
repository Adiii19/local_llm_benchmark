"""
src/models/model_manager.py

Model manager with memory optimization for 1B models
"""

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

from .device_utils import DeviceUtils


class ModelManager:
    """
    Simple model manager for 1B models
    Optimized for low memory systems
    """
    
    def __init__(self, cache_dir: str = "models/"):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = DeviceUtils.get_optimal_device()
        self.loaded_models = {}
        
        self._setup_hf_authentication()
        
        print(f"✓ ModelManager initialized")
        print(f"  Device: {self.device.upper()}")
        print(f"  Cache: {self.cache_dir}")
        print(f"  ✅ Optimized for 1B models")
    
    def _setup_hf_authentication(self):
        """Setup HuggingFace authentication"""
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
        """
        Load 1B model with memory optimization
        ✅ OPTIMIZED FOR LOW MEMORY SYSTEMS
        """
        
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
        
        # ✅ AGGRESSIVE CLEANUP BEFORE LOADING
        self._cleanup_memory()
        
        # ✅ CHECK AVAILABLE MEMORY
        memory = DeviceUtils.get_memory_usage()
        available_ram = memory['ram_available_gb']
        needed_ram = config.min_ram_gb + 1  # +1GB buffer
        
        if available_ram < needed_ram:
            print(f"\n❌ INSUFFICIENT MEMORY")
            print(f"   Available: {available_ram:.1f} GB")
            print(f"   Needed: {needed_ram:.1f} GB")
            print(f"   💡 Close other programs and try again")
            return None
        
        print(f"  ✓ Memory check passed ({available_ram:.1f}GB available)")
        
        try:
            # Step 1: Load tokenizer
            print(f"\n  📍 Step 1: Loading tokenizer...")
            tokenizer = self._load_tokenizer(model_id)
            
            if tokenizer is None:
                raise ValueError(f"Failed to load tokenizer for {model_id}")
            
            print(f"     ✓ Tokenizer loaded")
            
            # Step 2: Build load kwargs
            print(f"  📍 Step 2: Configuring model loading...")
            load_kwargs = self._get_load_kwargs()
            
            # Step 3: Load model
            print(f"  📍 Step 3: Loading model (this may take 1-2 minutes)...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs
            )
            
            if model is None:
                raise ValueError(f"Failed to load model for {model_id}")
            
            print(f"     ✓ Model loaded")
            
            # Set to eval mode
            model.eval()
            
            # Move to device
            try:
                model = model.to(self.device)
                print(f"     ✓ Moved to {self.device.upper()}")
            except Exception as e:
                print(f"     ⚠️  Warning: {str(e)[:50]}")
            
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
        
        except Exception as e:
            print(f"\n  ❌ FAILED to load {config.model_name}")
            print(f"\n  Error: {str(e)[:150]}")
            
            self._print_troubleshooting(model_id, str(e))
            
            self.loaded_models[model_id] = None
            return None
    
    def _get_load_kwargs(self) -> dict:
   
      kwargs = {
        'cache_dir': self.cache_dir,
        'trust_remote_code': True,
        # ✅ CRITICAL: Sequential loading of model shards
        'low_cpu_mem_usage': True,
    }
    
      if self.device == 'cuda':
        print(f"     ⚙️  GPU Configuration...")
        kwargs['device_map'] = 'auto'
        # ✅ Use float16 on GPU (supported)
        kwargs['torch_dtype'] = torch.float16
    
      else:
        print(f"     ⚙️  CPU Configuration (Memory Optimized)...")
        kwargs['device_map'] = None
        # ✅ FIX: Use float32 on CPU (float16 not supported!)
        kwargs['torch_dtype'] = torch.float32
    
      return kwargs
    
    def _load_tokenizer(self, model_id: str) -> Optional[AutoTokenizer]:
        """Load tokenizer with fallbacks"""
        
        print(f"     Attempting to load tokenizer...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_auth_token=True
            )
            return tokenizer
        
        except Exception as e1:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                return tokenizer
            
            except Exception as e2:
                print(f"     ❌ Failed to load tokenizer")
                return None
    
    def _print_troubleshooting(self, model_id: str, error_str: str):
        """Print troubleshooting guide"""
        
        error_lower = error_str.lower()
        
        if "out of memory" in error_lower or "oom" in error_lower:
            print(f"  ℹ️  OUT OF MEMORY ERROR")
            print(f"      1. Close all other programs")
            print(f"      2. Restart your computer")
            print(f"      3. Run only this script")
            print(f"      4. Check RAM: taskmgr → Performance → Memory")
        
        elif "401" in error_lower or "unauthorized" in error_lower:
            print(f"  ℹ️  Authentication Error")
            print(f"      Run: huggingface-cli login")
        
        elif "tokenizer" in error_lower:
            print(f"  ℹ️  Tokenizer Error")
            print(f"      Check internet connection")
            print(f"      Update: pip install --upgrade transformers")
        
        else:
            print(f"  ℹ️  General Error")
            print(f"      Update: pip install --upgrade torch transformers")
    
    def unload_model(self, model_id: str):
        """Unload a model"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            self._cleanup_memory()
            print(f"✓ Unloaded: {model_id}")
    
    def unload_all_models(self):
        """Unload all models"""
        self.loaded_models.clear()
        self._cleanup_memory()
        print(f"✓ All models unloaded")
    
    def _cleanup_memory(self):
        """
        Aggressive memory cleanup
        ✅ CRITICAL FOR 1B MODELS ON LOW MEMORY
        """
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()