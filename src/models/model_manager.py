"""
src/models/model_manager.py

Load and manage multiple models
Handle GPU memory, switching between models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional, Dict
import gc
from pathlib import Path

from .model_configs import ModelConfig, QUANTIZATION_OPTIONS
from .device_utils import DeviceUtils


class ModelManager:
    """
    Manages loading, unloading, and switching between models
    
    Key responsibility: Memory management for multiple large models
    """
    
    def __init__(self, cache_dir: str = "models/"):
        """
        Initialize model manager
        
        Args:
            cache_dir: Where to cache downloaded models
        """
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = DeviceUtils.get_optimal_device()
        self.loaded_models = {}  # Cache of loaded models
        self.current_model = None
        
        print(f"✓ ModelManager initialized (device: {self.device})")
    
    def load_model(
        self,
        config: ModelConfig,
        quantization: Optional[str] = None,
        force_reload: bool = False
    ) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """
        Load a model with optional quantization
        
        Args:
            config: ModelConfig specifying which model
            quantization: 'int4', 'int8', 'fp16', or None (fp32)
            force_reload: Reload even if cached
        
        Returns:
            (model, tokenizer) tuple
        """
        
        model_id = config.model_id
        
        # Check if already loaded
        if model_id in self.loaded_models and not force_reload:
            print(f"✓ Using cached model: {config.model_name}")
            return self.loaded_models[model_id]
        
        print(f"\n{'='*70}")
        print(f"📥 Loading {config.model_name}...")
        print(f"{'='*70}")
        
        # Clear GPU memory if switching models
        self._cleanup_memory()
        
        # Prepare loading config
        load_kwargs = {
            'device_map': 'auto' if self.device == 'cuda' else None,
            'cache_dir': self.cache_dir,
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32
        }
        
        # Add quantization if specified
        if quantization and quantization != 'fp32':
            q_config = QUANTIZATION_OPTIONS[quantization]
            print(f"  Quantization: {q_config}")
            
            if quantization == 'int8':
                load_kwargs['load_in_8bit'] = True
            elif quantization == 'int4':
                load_kwargs['load_in_4bit'] = True
                load_kwargs['bnb_4bit_compute_dtype'] = torch.float16
                load_kwargs['bnb_4bit_quant_type'] = "nf4"
        
        try:
            # Load tokenizer
            print(f"  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Load model
            print(f"  Loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
                trust_remote_code=True
            )
            
            model.eval()  # Set to evaluation mode
            
            # Cache it
            self.loaded_models[model_id] = (model, tokenizer)
            self.current_model = model_id
            
            # Show memory usage
            memory = DeviceUtils.get_memory_usage()
            print(f"  ✓ Model loaded successfully")
            print(f"  Memory used: {memory.get('gpu_used_gb', 0):.1f}GB GPU, "
                  f"{memory['ram_used_gb']:.1f}GB RAM")
            
            return model, tokenizer
        
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            raise
    
    def unload_model(self, model_id: str):
        """Unload a specific model to free memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            self._cleanup_memory()
            print(f"✓ Unloaded: {model_id}")
    
    def unload_all_models(self):
        """Unload all cached models"""
        self.loaded_models.clear()
        self.current_model = None
        self._cleanup_memory()
        print("✓ All models unloaded")
    
    def _cleanup_memory(self):
        """Clean up GPU and CPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print("  ✓ Memory cleaned")
    
    def get_model(self, config: ModelConfig) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Get model (load if not cached)"""
        if config.model_id not in self.loaded_models:
            return self.load_model(config)
        return self.loaded_models[config.model_id]


if __name__ == "__main__":
    from .model_configs import MISTRAL_7B, LLAMA2_13B
    
    # Show device info
    DeviceUtils.print_device_info()
    
    # Initialize manager
    manager = ModelManager(cache_dir="models/")
    
    # Load first model
    print("\n1️⃣ Loading first model...")
    model1, tokenizer1 = manager.load_model(MISTRAL_7B, quantization='int8')
    
    # Show memory
    memory = DeviceUtils.get_memory_usage()
    print(f"\nMemory after loading Mistral: {memory.get('gpu_used_gb', 0):.1f}GB GPU")
    
    # Load second model (should handle memory)
    print("\n2️⃣ Loading second model (will clean first)...")
    model2, tokenizer2 = manager.load_model(LLAMA2_13B)
    
    # Cleanup
    manager.unload_all_models()