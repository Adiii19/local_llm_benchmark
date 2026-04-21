"""
src/inference/inference_engine.py

Core inference engine
Runs prompts and measures performance
"""

import torch
import time
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
import numpy as np


class InferenceConfig:
    """Configuration for inference"""
    
    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty


class InferenceEngine:
    """
    Run inference on models
    Measure latency, throughput, memory usage
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def generate(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        prompt: str,
        config: InferenceConfig = None
    ) -> Tuple[str, Dict]:
        """
        Generate text from a prompt
        
        Args:
            model: Loaded model
            tokenizer: Tokenizer
            prompt: Input prompt
            config: Inference configuration
        
        Returns:
            (generated_text, metrics) tuple
        """
        if config is None:
            config = InferenceConfig()
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Record memory before
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Generate with timing
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        elapsed_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Calculate metrics
        output_length = output_ids.shape[1] - input_length
        tokens_per_second = output_length / elapsed_time if elapsed_time > 0 else 0
        
        # Memory after
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (memory_after - memory_before) / (1024**3)  # GB
        
        metrics = {
            'input_tokens': input_length,
            'output_tokens': output_length,
            'total_tokens': input_length + output_length,
            'latency_seconds': elapsed_time,
            'tokens_per_second': tokens_per_second,
            'memory_used_gb': memory_used,
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
        
        Args:
            model: Loaded model
            tokenizer: Tokenizer
            prompts: List of prompts
            config: Inference configuration
        
        Returns:
            (generated_texts, metrics_list) tuple
        """
        generated_texts = []
        all_metrics = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Generating [{i+1}/{len(prompts)}]...", end='\r')
            
            text, metrics = self.generate(model, tokenizer, prompt, config)
            generated_texts.append(text)
            all_metrics.append(metrics)
        
        print(f"  ✓ Generated {len(prompts)} responses")
        
        return generated_texts, all_metrics
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
        """
        Aggregate metrics from multiple runs
        
        Args:
            metrics_list: List of individual metric dicts
        
        Returns:
            Aggregated statistics
        """
        if not metrics_list:
            return {}
        
        latencies = [m['latency_seconds'] for m in metrics_list]
        token_rates = [m['tokens_per_second'] for m in metrics_list]
        output_tokens = [m['output_tokens'] for m in metrics_list]
        
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
            
            'mean_output_tokens': np.mean(output_tokens),
            'total_tokens_generated': np.sum(output_tokens),
            'num_runs': len(metrics_list)
        }


if __name__ == "__main__":
    print("InferenceEngine module loaded")