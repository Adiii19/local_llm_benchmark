import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from typing import Tuple,Optional,Dict
import gc
from pathlib import Path

from .model_configs import ModelConfig, QUANTIZATION_OPTIONS
from .device_utils import DeviceUtils

class ModelManager:

    def __init__(self,cache_dir:str="models/"):
        
        self.cache_dir=cache_dir
        Path(cache_dir).mkdir(parents=True,exist_ok=True)

        self.device=DeviceUtils.get_optimal_device()
        self.loaded_model={}
        self.current_model=None

        print(f"Model Manager initialized (device:{self.device})")

    def load_model(
            
        self,
        config:ModelConfig,
        quantization:Optional[str]=None,
        force_reload:bool=False

    )->Tuple[torch.nn.Module,AutoTokenizer]:
        
        model_id=config.model_id

        if model_id in self.loaded_model and not force_reload:
            print(f"USing cache model:{config.model_name}")
            return self.loaded_model[model_id]
        
        print(f"\n{'='*70}")
        print(f"Loading {config.model_name}....")
        print(f"{'='*70}")

        self._cleanup_memory()

        load_kwargs={
            'device_map':'auto' if self.device=='cuda' else None,
            'cache_dir':self.cache_dir,
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32
        }

        if quantization and quantization !='fp32':
            q_config=QUANTIZATION_OPTIONS[quantization]
            print(f"Quantization:{q_config}")

            if quantization =='int8':
                print(f"  Quantization: {q_config}")

                if quantization =='int8':
                    load_kwargs['load_in_8bit']=True
                elif quantization =='int4':
                    load_kwargs['load_in_4bit']=True
                    load_kwargs['bnb_4bit_compute_dtype']=torch.float16
                    load_kwargs['bnb_4bit_quant_type']="nf4"

            try:

                    print(f"Loading tokenizer...")
                    tokenizer=AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )

                    model.eval()

                    self.loaded_models[model_id]=(model,tokenizer)
                    self.current_model=model_id

                    memory=DeviceUtils.get_memory_usage()
                    print(f"Model loaded successfully")
                    print(f"  Memory used: {memory.get('gpu_used_gb', 0):.1f}GB GPU, "
                  f"{memory['ram_used_gb']:.1f}GB RAM")
                    
                    return model,tokenizer
            
            except Exception as e:
                print(f"  ❌ Error loading model: {e}")
            raise

    def unload_model(self,model_id:str):
        if model_id in self.loaded_models:
            del self.loaded_model[model_id]
            self._cleanup_memory()
            print(f"✓ Unloaded: {model_id}")
    
    def unload_all_models(self):
        self.loaded_models.clear()
        self.current_model=None
        self._cleanup_memory()
        print("All models unloaded!!!")

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print("Memory Cleaned")

    def get_model(self,config:ModelConfig)->Tuple[torch.nn.Module,AutoTokenizer]:
        if config.model_id not in self.loaded_models:
            return self.load_model(config)
        return self.loaded_models[config.model_id]
    

if __name__ == "__main__":
    from .model_configs import MISTRAL_7B, LLAMA2_13B
    
   
    DeviceUtils.print_device_info()
    
    manager = ModelManager(cache_dir="models/")
    
     
    print("\n1️⃣ Loading first model...")
    model1, tokenizer1 = manager.load_model(MISTRAL_7B, quantization='int8')
    
    memory = DeviceUtils.get_memory_usage()
    print(f"\nMemory after loading Mistral: {memory.get('gpu_used_gb', 0):.1f}GB GPU")
    
    print("\n2️⃣ Loading second model (will clean first)...")
    model2, tokenizer2 = manager.load_model(LLAMA2_13B)
    
    
    manager.unload_all_models()
