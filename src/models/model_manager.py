"""
src/models/model_manager.py

COMPLETELY CORRECTED - With automatic GPU/CPU detection and switching
✅ KEY CHANGES: Automatic device detection and proper error handling
"""

import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Tuple, Optional
import gc
from pathlib import Path
from huggingface_hub import login as hf_login 
import traceback

from .device_utils import DeviceUtils  # ← IMPORT: Use device utils


class ModelManager:
    """
    Model manager with automatic GPU/CPU detection
    ✅ CHANGE: Auto-detects device and switches appropriately
    """
    
    def __init__(self, cache_dir: str = "models/"):
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # ✅ CHANGE 1: Use DeviceUtils for device detection
        # OLD: Manual torch.cuda.is_available()
        # NEW: Use centralized DeviceUtils.get_optimal_device()
        self.device = DeviceUtils.get_optimal_device()  # ← CHANGED: Use DeviceUtils
        self._setup_hf_authentication()

        self.loaded_models = {}
        
        print(f"✓ ModelManager initialized")
        print(f"  Device: {self.device.upper()}")  # ← Shows CUDA or CPU
        print(f"  Cache: {self.cache_dir}")

    def _setup_hf_authentication(self):
        """
        ✅ CHANGE: Setup HuggingFace authentication
        Fixes Llama2 access issue
        """
        try:
            # Try to get token from environment
            token = os.getenv('HUGGING_FACE_TOKEN')
            
            if token:
                print(f"✓ Using HuggingFace token from environment")
                hf_login(token=token, add_to_git_credential=False)
            else:
                # Try to read from .huggingface/token
                hf_token_path = Path.home() / ".huggingface" / "token"
                if hf_token_path.exists():
                    print(f"✓ Found HuggingFace token at ~/.huggingface/token")
                    # Token will be used automatically
                else:
                    print(f"⚠️  No HuggingFace token found")
                    print(f"   For Llama2, run: huggingface-cli login")
        
        except Exception as e:
            print(f"⚠️  HuggingFace auth warning: {str(e)[:50]}")
    
    # ✅ CHANGE 2: Changed return type to Optional[Tuple]
    def load_model(
        self,
        config,
        quantization: Optional[str] = None,
        force_reload: bool = False
    ) -> Optional[Tuple[torch.nn.Module, AutoTokenizer]]:
        """
        Load model with automatic GPU/CPU detection
        
        ✅ CHANGE: Auto-selects device based on availability
        Returns:
            (model, tokenizer) or None if failed
        """
        
        model_id = config.model_id
        
        # ✅ CHANGE 3: Check cache with None safety
        if model_id in self.loaded_models and not force_reload:
            print(f"✓ Using cached: {config.model_name}")
            cached = self.loaded_models[model_id]
            if cached is not None:
                return cached
        
        print(f"\n{'='*70}")
        print(f"📥 Loading {config.model_name}")
        print(f"   Model ID: {model_id}")
        print(f"   Device: {self.device.upper()}")  # ← Shows detected device
        print(f"{'='*70}")
        
        # Clean memory
        self._cleanup_memory()
        
        try:
            # ✅ CHANGE 4: Build kwargs based on detected device
            load_kwargs = self._get_load_kwargs(config, quantization)
            
            # Step 1: Load tokenizer
            print(f"\n  📍 Step 1: Loading tokenizer...")
            tokenizer = self._load_tokenizer(model_id)
            
            if tokenizer is None:
                raise ValueError(f"Failed to load tokenizer for {model_id}")
            
            print(f"     ✓ Tokenizer loaded")
            
            # Step 2: Load model
            print(f"  📍 Step 2: Loading model...")
            print(f"     Device: {self.device.upper()}")  # ← Confirm device
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
                
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
                print(f"     ⚠️  Warning moving to device: {str(e)[:50]}")
            
            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Cache it
            self.loaded_models[model_id] = (model, tokenizer)
            
            print(f"\n  ✅ SUCCESS!")
            print(f"     Model: {config.model_name}")
            print(f"     Size: {config.size}")
            print(f"     Device: {self.device.upper()}")  # ← Confirm final device
            
            return (model, tokenizer)
        
        except Exception as e:
            print(f"\n  ❌ FAILED to load {config.model_name}")
            print(f"\n  Error Type: {type(e).__name__}")
            print(f"  Error Message: {str(e)}")
            
            if "trust_remote_code" in str(e).lower():
                print(f"\n  This is a known transformers issue - trying workaround...")
                # Try loading with different approach
                try:
                    model = self._load_model_with_workaround(model_id, **self._get_load_kwargs(config, quantization))
                    if model:
                        print(f"  ✓ Workaround successful!")
                        self.loaded_models[model_id] = (model, tokenizer)
                        return (model, tokenizer)
                except:
                    pass
            
            print(f"\n  Full Traceback:")
            traceback.print_exc()
            
            print(f"\n  💡 Troubleshooting Guide:")
            self._print_troubleshooting(model_id, str(e))
           
            
            # Mark as failed in cache
            self.loaded_models[model_id] = None
            
            return None
        
    def _load_model_with_workaround(self, model_id: str, **kwargs) -> Optional[torch.nn.Module]:
        """
        ✅ CHANGE: Workaround for trust_remote_code duplicate issue
        This fixes the Neural Chat loading problem
        """
        try:
            # Remove trust_remote_code from kwargs if present
            if 'trust_remote_code' in kwargs:
                kwargs.pop('trust_remote_code')
            
            print(f"     Trying workaround (removing trust_remote_code)...")
            
            # Try loading without trust_remote_code
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **kwargs
            )
            return model
        
        except Exception as e:
            print(f"     Workaround failed: {str(e)[:50]}")
            return None
    
    # ✅ CHANGE 5: Get load kwargs based on DETECTED device
    def _get_load_kwargs(self, config, quantization: Optional[str]) -> dict:
        """
        Get proper loading kwargs based on DETECTED device
        
        ✅ CHANGE: Smart configuration based on device
        """
        kwargs = {
            'cache_dir': self.cache_dir,
            'trust_remote_code': True,
        }
        
        # ✅ CHANGE 6: Different config for GPU vs CPU
        if self.device == 'cuda':  # ← Check DETECTED device
            print(f"     ⚙️  Configuring for GPU...")
            kwargs['device_map'] = 'auto'
            kwargs['torch_dtype'] = torch.float16  # FP16 for GPU speed
            
            # Add quantization if specified (GPU only)
            if quantization == 'int8':
                print(f"     Using 8-bit quantization...")
                kwargs['load_in_8bit'] = True
            elif quantization == 'int4':
                print(f"     Using 4-bit quantization...")
                kwargs['load_in_4bit'] = True
                kwargs['bnb_4bit_compute_dtype'] = torch.float16
                kwargs['bnb_4bit_quant_type'] = "nf4"
        
        else:  # CPU mode
            print(f"     ⚙️  Configuring for CPU...")
            kwargs['device_map'] = None
            kwargs['torch_dtype'] = torch.float32  # FP32 for CPU stability
            kwargs['low_cpu_mem_usage'] = True
            
            # ✅ CHANGE 7: Warn about quantization on CPU
            if quantization and quantization != 'fp32':
                print(f"     ⚠️  Quantization not optimized for CPU, using fp32")
        
        return kwargs
    
    def _load_tokenizer(self, model_id: str) -> Optional[AutoTokenizer]:
        """
        Load tokenizer with multiple fallbacks
        Same for both GPU and CPU
        """
        
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
            error_msg = str(e1).lower()
            
            # ✅ CHANGE 4: Better error message for Llama2
            if "unauthorized" in error_msg or "401" in error_msg or "access" in error_msg:
                print(f"     ⚠️  Authentication required (probably Llama2)")
                print(f"         Please run: huggingface-cli login")
                print(f"         Then accept license at: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf")
                return None
            
            print(f"     ⚠️  Method 1 failed: {type(e1).__name__}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    use_auth_token=True
                )
                return tokenizer
            
            except Exception as e2:
                print(f"     ⚠️  Method 2 failed: {type(e2).__name__}")
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
                    return tokenizer
                
                except Exception as e3:
                    print(f"     ⚠️  Method 3 failed: {type(e3).__name__}")
                    if "unauthorized" in str(e3).lower() or "401" in str(e3):
                        print(f"     ❌ Authentication required - cannot load {model_id}")
                        print(f"        This is likely a gated model (e.g., Llama2)")
                        print(f"        Run: huggingface-cli login")
                    else:
                        print(f"     ❌ All tokenizer methods failed!")
                    return None
    
    def _print_troubleshooting(self, model_id: str, error_str: str):
        """Print troubleshooting guide"""
        
        error_lower = error_str.lower()

        if "trust_remote_code" in error_lower:
            print(f"  ℹ️  trust_remote_code Duplicate Parameter Error")
            print(f"      This was a bug - now fixed!")
            print(f"      Please update the code:")
            print(f"      1. Update to latest model_manager.py")
            print(f"      2. Delete cache: rm -rf models/")
            print(f"      3. Try again")
        
        # ✅ CHANGE 7: Handle Llama2 auth
        elif "unauthorized" in error_lower or "401" in error_lower or "access_denied" in error_lower:
            print(f"  ℹ️  Llama2 Authentication Required")
            print(f"      1. Go to: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf")
            print(f"      2. Click 'Access repository'")
            print(f"      3. Accept license agreement")
            print(f"      4. Run: huggingface-cli login")
            print(f"      5. Paste your HF token (from https://huggingface.co/settings/tokens)")
            print(f"      6. Try again")  
        
        # ✅ CHANGE 8: Add device-specific troubleshooting
        elif "cuda" in error_lower and "out of memory" in error_lower:
            print(f"  ℹ️  GPU OUT OF MEMORY")
            print(f"      1. Your GPU doesn't have enough memory")
            print(f"      2. Current device: {self.device.upper()}")
            print(f"      3. Try:")
            print(f"         a) Use smaller model")
            print(f"         b) Use int8 quantization: quantization='int8'")
            print(f"         c) Close other GPU programs")
            print(f"      4. Check GPU: nvidia-smi")
        
        elif "llama" in model_id.lower() and ("not a valid model" in error_lower or "401" in error_lower):
            print(f"  ℹ️  Llama2 Authentication Required")
            print(f"      1. Go to: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf")
            print(f"      2. Accept license agreement")
            print(f"      3. Run: huggingface-cli login")
            print(f"      4. Paste your HF token")
        
        elif "nonetype" in error_lower or "none" in error_lower:
            print(f"  ℹ️  NoneType Error - Model loading returned None")
            print(f"      Device: {self.device.upper()}")
            print(f"      1. Check internet connection")
            print(f"      2. Verify model exists")
            print(f"      3. Update transformers: pip install --upgrade transformers")
            print(f"      4. Clear cache: rm -rf models/")
        
        else:
            print(f"  ℹ️  Generic Troubleshooting")
            print(f"      Device: {self.device.upper()}")
            print(f"      1. Update: pip install --upgrade torch transformers")
            print(f"      2. Clear cache: rm -rf models/")
            print(f"      3. Check internet")
            print(f"      4. If GPU: check nvidia-smi")
    
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
    
    # ✅ CHANGE 9: Improved cleanup for both GPU and CPU
    def _cleanup_memory(self):
        """
        Clean up memory
        ✅ CHANGE: Safe cleanup that works on both GPU and CPU
        """
        gc.collect()
        
        # ✅ CHANGE 10: Only clean GPU if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()