"""
src/benchmarking/benchmark_suite.py

✅ UPDATED: Uses working INT8 quantized models
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import torch

# ✅ CHANGE: Import correct INT8 models
from ..models.model_configs import (
    MISTRAL_7B_INT8, NEURAL_CHAT_7B_INT8, DISTIL_GPT2
)
from ..models.model_manager import ModelManager
from ..inference.inference_engine import InferenceEngine, InferenceConfig
from ..models.device_utils import DeviceUtils


class BenchmarkingSuite:
    """
    Benchmarking framework for INT8 quantized models
    ✅ CHANGE: Proven working approach
    """
    
    def __init__(self, output_dir: str = "benchmark_results/", cache_dir: str = "models/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_manager = ModelManager(cache_dir=cache_dir)
        
        optimal_device = DeviceUtils.get_optimal_device()
        self.inference_engine = InferenceEngine(device=optimal_device)
        
        self.results = {}
        self.device_info = None
        self.failed_models = []
    
    def benchmark_model(
        self,
        config,
        prompts: List[str],
        quantization: str = None,
        num_runs: int = 1
    ) -> Optional[Dict]:
        """
        Benchmark a single model
        ✅ CHANGE: Simplified INT8 approach
        """
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARKING: {config.model_name}")
        print(f"{'='*70}")
        print(f"Disk Size: {config.disk_size_gb} GB")
        print(f"Quantization: {config.quantization_type}")
        print(f"Prompts: {len(prompts)}")
        
        # Load model
        model_result = self.model_manager.load_model(
            config,
            quantization=config.quantization_type
        )
        
        # Check if loaded
        if model_result is None:
            print(f"\n❌ SKIPPING {config.model_name}")
            self.failed_models.append({
                'model': config.model_name,
                'reason': 'Failed to load'
            })
            return None
        
        # Safe unpacking
        try:
            model, tokenizer = model_result
        except Exception as e:
            print(f"\n❌ ERROR unpacking: {e}")
            self.failed_models.append({
                'model': config.model_name,
                'reason': f'Unpack error'
            })
            return None
        
        # Run benchmarks
        try:
            all_metrics = []
            
            for run in range(num_runs):
                if num_runs > 1:
                    print(f"\n📍 Run {run + 1}/{num_runs}")
                
                _, metrics_list = self.inference_engine.batch_generate(
                    model,
                    tokenizer,
                    prompts
                )
                
                all_metrics.extend(metrics_list)
            
            # Aggregate
            aggregated = self.inference_engine.aggregate_metrics(all_metrics)
            
            result = {
                'model_name': config.model_name,
                'model_id': config.model_id,
                'model_size': config.size,
                'quality_score': config.quality_score,
                'quantization': config.quantization_type,
                'disk_size_gb': config.disk_size_gb,
                'ram_requirement_gb': config.min_ram_gb,
                'gpu_memory_requirement_gb': config.min_gpu_memory_gb,
                
                'benchmark_date': datetime.now().isoformat(),
                'num_prompts': len(prompts),
                'num_runs': num_runs,
                'total_runs': len(all_metrics),
                
                'metrics': aggregated,
                'device_used': aggregated.get('device', 'unknown'),
                'expected_tps_cpu': config.tokens_per_second_cpu,
                'expected_tps_gpu': config.tokens_per_second_gpu
            }
            
            self.results[config.model_name] = result
            
            # Display results
            self._print_benchmark_results(result)
            
            # Unload
            self.model_manager.unload_model(config.model_id)
            
            return result
        
        except Exception as e:
            print(f"\n❌ Benchmarking failed: {e}")
            self.failed_models.append({
                'model': config.model_name,
                'reason': f'Benchmark error'
            })
            self.model_manager.unload_model(config.model_id)
            return None
    
    def benchmark_all_models(
        self,
        prompts: List[str],
        models: List = None,
        quantizations: List[str] = None
    ) -> Dict:
        """
        Benchmark all models
        ✅ CHANGE: Use working INT8 models
        """
        if models is None:
            # ✅ CHANGE: Use INT8 models that actually work
            models = [MISTRAL_7B_INT8, NEURAL_CHAT_7B_INT8, DISTIL_GPT2]
        
        if quantizations is None:
            quantizations = ['auto']
        
        print("\n" + "="*70)
        print("🚀 LIGHTWEIGHT BENCHMARK SUITE")
        print("="*70)
        print("✅ INT8 Quantized models (3.5 GB each)")
        
        # Device info
        self.device_info = DeviceUtils.print_device_info()
        
        # Benchmark
        successful = 0
        for model_config in models:
            for quant in quantizations:
                result = self.benchmark_model(
                    model_config,
                    prompts,
                    quantization=quant,
                    num_runs=1
                )
                
                if result is not None:
                    successful += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Done: {successful}/{len(models)} models successful")
        print("="*70)
        
        # Save results
        self.save_results()
        
        # Display
        if self.results:
            self.display_comparison()
        else:
            print("\n⚠️  No successful benchmarks")
        
        return self.results
    
    def _print_benchmark_results(self, result: Dict):
        """Print results"""
        m = result['metrics']
        
        print(f"\n✅ RESULTS:")
        print(f"{'─'*70}")
        print(f"  Device:       {m.get('device', 'unknown').upper()}")
        print(f"  Disk Size:    {result['disk_size_gb']} GB")
        print(f"  Quantization: {result['quantization']}")
        print(f"  Latency:      {m['mean_latency']:.3f}s")
        print(f"  Throughput:   {m['mean_tps']:.1f} tokens/sec")
        print(f"  Quality:      {result['quality_score']:.2f}/1.0")
        print(f"{'─'*70}\n")
    
    def display_comparison(self):
        """Compare models"""
        if not self.results:
            return
        
        print("\n" + "="*130)
        print("📊 MODEL COMPARISON (INT8 Quantized, 25GB Storage)")
        print("="*130)
        
        print(
            f"{'Model':<30} "
            f"{'Quant':<12} "
            f"{'Disk':<10} "
            f"{'Device':<10} "
            f"{'Latency':<12} "
            f"{'Speed':<15} "
            f"{'Quality':<10}"
        )
        print("-"*130)
        
        for key, result in self.results.items():
            m = result['metrics']
            device = m.get('device', 'unknown').upper()
            
            print(
                f"{result['model_name']:<30} "
                f"{result['quantization']:<12} "
                f"{result['disk_size_gb']:.2f}GB{'':<5} "
                f"{device:<10} "
                f"{m['mean_latency']*1000:>6.1f}ms{'':<5} "
                f"{m['mean_tps']:>6.1f} tok/s{'':<8} "
                f"{result['quality_score']:.2f}⭐"
            )
        
        print("="*130)
        print(f"\n💾 Total Storage: {sum(r['disk_size_gb'] for r in self.results.values()):.2f} GB ✅")
    
    def save_results(self):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_{timestamp}.json"
        
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'device_info': self.device_info,
            'results': self.results,
            'failed_models': self.failed_models,
            'summary': {
                'total_models': len(self.results) + len(self.failed_models),
                'successful': len(self.results),
                'failed': len(self.failed_models),
                'total_disk_used_gb': sum(r.get('disk_size_gb', 0) for r in self.results.values())
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
        return filename