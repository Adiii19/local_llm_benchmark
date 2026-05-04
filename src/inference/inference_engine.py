"""
src/inference/inference_engine.py

CORRECTED - Works on both GPU and CPU
✅ CHANGE: Auto-detects device and optimizes accordingly
"""

import torch
import time
import threading
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import numpy as np

from ..models.device_utils import DeviceUtils  # ← IMPORT: Use device utils


class InferenceConfig:
    """Inference configuration"""
    
    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        timeout_seconds: Optional[int] = None  # ← NEW: Timeout support
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.use_cache = use_cache
        self.timeout_seconds = timeout_seconds  # ← NEW: Timeout


class InferenceEngine:
    """
    Inference engine for CPU and GPU
    ✅ CHANGE: Auto-detects device
    """
    
    def __init__(self, device: str = None):
        # ✅ CHANGE 1: Use DeviceUtils if device not specified
        if device is None:
            self.device = DeviceUtils.get_optimal_device()  # ← Auto-detect
        else:
            self.device = device
        
        print(f"✓ InferenceEngine initialized")
        print(f"  Device: {self.device.upper()}")  # ← Show device
    
    def generate(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        prompt: str,
        config: InferenceConfig = None
    ) -> Tuple[str, Dict]:
        """
        Generate text from prompt
        Works on both CPU and GPU
        
        ✅ CHANGE: Device-aware memory tracking, CPU optimizations, progress logging
        """
        
        if config is None:
            config = InferenceConfig()
        
        # ✅ NEW: Adjust config for CPU if needed
        if self.device == 'cpu':
            # Reduce tokens for CPU to prevent very long runs
            if config.max_new_tokens > 128:
                print(f"     ℹ️  Reducing max_new_tokens from {config.max_new_tokens} to 128 for CPU")
                config.max_new_tokens = 128
        
        # Prepare input
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)  # ← Move to detected device
        
        input_length = inputs.input_ids.shape[1]
        
        # ✅ CHANGE 2: Device-aware memory tracking
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / (1024**3)
        else:
            memory_before = 0
        
        # Generate with progress tracking
        start_time = time.time()
        print(f"     ⚡ Starting generation: prompt_tokens={input_length}, max_new_tokens={config.max_new_tokens}")
        if self.device == 'cpu':
            print(f"     ℹ️  CPU mode: This may take 1-5 minutes for longer outputs. Do not interrupt.")
        
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    max_length=input_length + config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    use_cache=config.use_cache,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    min_length=10  # ← Ensure minimum output
                )
            
            elapsed_time = time.time() - start_time
            print(f"     ✓ Generation completed in {elapsed_time:.1f}s")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"     ✗ Generation failed after {elapsed_time:.1f}s: {str(e)[:100]}")
            return "", {
                'error': str(e),
                'latency_seconds': elapsed_time,
                'device': self.device
            }
        
        # Decode
        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Calculate metrics
        output_length = output_ids.shape[1] - input_length
        tokens_per_second = output_length / elapsed_time if elapsed_time > 0 else 0
        
        # ✅ CHANGE 3: Device-aware memory reporting
        if self.device == 'cuda':
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
        
        metrics = {
            'input_tokens': input_length,
            'output_tokens': output_length,
            'total_tokens': input_length + output_length,
            'latency_seconds': elapsed_time,
            'tokens_per_second': tokens_per_second,
            'memory_used_gb': memory_used,
            'device': self.device,  # ← Track which device
            'timestamp': time.time()
        }
        
        return generated_text, metrics
    
    def batch_generate(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        config: InferenceConfig = None
    ) -> Tuple[List[str], List[Dict]]:
        """
        Generate for multiple prompts
        """
        generated_texts = []
        all_metrics = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Generating [{i+1}/{len(prompts)}]...")
            
            text, metrics = self.generate(model, tokenizer, prompt, config)
            generated_texts.append(text)
            all_metrics.append(metrics)
        
        print(f"  ✓ Generated {len(prompts)} responses")
        
        return generated_texts, all_metrics
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
        """
        Aggregate metrics from multiple runs
        """
        if not metrics_list:
            return {}
        
        latencies = [m['latency_seconds'] for m in metrics_list]
        token_rates = [m['tokens_per_second'] for m in metrics_list]
        
        # ✅ CHANGE 4: Include device in aggregated metrics
        device = metrics_list[0].get('device', 'unknown') if metrics_list else 'unknown'
        
        return {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            
            'mean_tps': np.mean(token_rates),
            'median_tps': np.median(token_rates),
            'std_tps': np.std(token_rates),
            'min_tps': np.min(token_rates),
            'max_tps': np.max(token_rates),
            
            'device': device,  # ← Track device
            'num_runs': len(metrics_list)
        }