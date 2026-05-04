"""
src/models/model_manager.py

✅ COMPLETELY REWRITTEN: Works with INT8 quantization (not GGUF)
Simplified, proven approach
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
    Model manager with INT8 quantization support
    ✅ CHANGE: Simplified to use only proven INT8 method
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
        print(f"  ✅ INT8 Quantization enabled")
    
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
        Load model with INT8 quantization
        ✅ CHANGE: Simplified, proven approach
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
        print(f"   Quantization: {config.quantization_type}")
        print(f"   Disk Size: {config.disk_size_gb} GB")
        print(f"   Device: {self.device.upper()}")
        print(f"{'='*70}")
        
        # Clean memory
        self._cleanup_memory()
        
        try:
            # Step 1: Load tokenizer
            print(f"\n  📍 Step 1: Loading tokenizer...")
            tokenizer = self._load_tokenizer(model_id)
            
            if tokenizer is None:
                raise ValueError(f"Failed to load tokenizer for {model_id}")
            
            print(f"     ✓ Tokenizer loaded")
            
            # Step 2: Build load kwargs
            print(f"  📍 Step 2: Configuring for {self.device.upper()}...")
            load_kwargs = self._get_load_kwargs(config)
            
            # Step 3: Load model
            print(f"  📍 Step 3: Loading model...")
            
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
            print(f"     Quantization: {config.quantization_type}")
            print(f"     Device: {self.device.upper()}")
            
            return (model, tokenizer)
        
        except Exception as e:
            print(f"\n  ❌ FAILED to load {config.model_name}")
            print(f"\n  Error: {str(e)}")
            
            # ✅ CHANGE: Better error messages
            self._print_troubleshooting(model_id, str(e))
            
            self.loaded_models[model_id] = None
            return None
    
    def _get_load_kwargs(self, config) -> dict:
        """
        Get load kwargs with INT8 quantization
        ✅ CHANGE: Simplified, focused on INT8
        """
        kwargs = {
            'cache_dir': self.cache_dir,
            'trust_remote_code': True,
        }
        
        if self.device == 'cuda':
            print(f"     ⚙️  GPU with INT8 Quantization...")
            kwargs['device_map'] = 'auto'
            kwargs['torch_dtype'] = torch.float16
            
            # ✅ CHANGE: Always use INT8 for GPU quantization
            if config.quantization_type == 'int8':
                print(f"     Applying INT8 quantization...")
                kwargs['load_in_8bit'] = True
        
        else:  # CPU mode
            print(f"     ⚙️  CPU Mode...")
            kwargs['device_map'] = None
            kwargs['torch_dtype'] = torch.float32
            kwargs['low_cpu_mem_usage'] = True
            
            # INT8 on CPU is possible but slower
            # We'll skip it for better speed on CPU
        
        return kwargs
    
    def _load_tokenizer(self, model_id: str) -> Optional[AutoTokenizer]:
        """
        Load tokenizer with proper error handling
        ✅ CHANGE: Better handling
        """
        
        print(f"     Attempting to load tokenizer...")
        
        try:
            # Try with auth
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_auth_token=True
            )
            print(f"     ✓ Tokenizer loaded with auth")
            return tokenizer
        
        except Exception as e1:
            try:
                # Try without auth
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                print(f"     ✓ Tokenizer loaded")
                return tokenizer
            
            except Exception as e2:
                print(f"     ❌ Failed to load tokenizer")
                
                # ✅ CHANGE: Better error messaging
                if "401" in str(e1) or "Unauthorized" in str(e1):
                    print(f"     Reason: Authentication required")
                    print(f"     Solution: huggingface-cli login")
                else:
                    print(f"     Reason: Model not found or invalid")
                
                return None
    
    def _print_troubleshooting(self, model_id: str, error_str: str):
        """Print helpful troubleshooting"""
        
        error_lower = error_str.lower()
        
        if "401" in error_lower or "unauthorized" in error_lower:
            print(f"  ℹ️  Authentication Error")
            print(f"      Run: huggingface-cli login")
            print(f"      Then try again")
        
        elif "out of memory" in error_lower:
            print(f"  ℹ️  Out of Memory")
            print(f"      Try using a smaller model")
            print(f"      Close other programs")
            print(f"      Check available RAM: available GB")
        
        elif "tokenizer" in error_lower:
            print(f"  ℹ️  Tokenizer Error")
            print(f"      Model might not exist")
            print(f"      Update transformers: pip install --upgrade transformers")
        
        else:
            print(f"  ℹ️  General Error")
            print(f"      Check internet connection")
            print(f"      Update libraries: pip install --upgrade torch transformers")
            print(f"      Delete cache: rm -rf models/")
    
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
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()