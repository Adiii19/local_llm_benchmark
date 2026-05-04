"""
src/benchmarking/benchmark_suite.py

CORRECTED - Key changes for GPU/CPU detection and handling
"""

import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import torch

from ..models.model_configs import MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_7B
from ..models.model_manager import ModelManager
from ..inference.inference_engine import InferenceEngine, InferenceConfig
from ..models.device_utils import DeviceUtils


class BenchmarkSuite:
    """
    Complete benchmarking framework with GPU/CPU detection
    ✅ CHANGE: Uses DeviceUtils for device detection
    """
    
    def __init__(self, output_dir: str = "benchmark_results/", cache_dir: str = "models/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_manager = ModelManager(cache_dir=cache_dir)
        
        # ✅ CHANGE 1: Use DeviceUtils for device detection
        # OLD: Manual torch.cuda.is_available()
        # NEW: Use centralized device detection
        optimal_device = DeviceUtils.get_optimal_device()  # ← Get optimal device
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
        ✅ CHANGE: Safe error handling with device info
        """
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARKING: {config.model_name}")
        print(f"{'='*70}")
        print(f"Quantization: {quantization or 'fp32'}")
        print(f"Prompts: {len(prompts)}")
        
        # Load model
        model_result = self.model_manager.load_model(
            config,
            quantization=quantization
        )
        
        # Check if model loaded
        if model_result is None:
            print(f"\n❌ SKIPPING {config.model_name}")
            print(f"   Could not load model - check errors above")
            self.failed_models.append({
                'model': config.model_name,
                'reason': 'Failed to load'
            })
            return None
        
        # Safe unpacking
        try:
            model, tokenizer = model_result
        except Exception as e:
            print(f"\n❌ ERROR unpacking model: {e}")
            self.failed_models.append({
                'model': config.model_name,
                'reason': f'Unpack error: {str(e)[:50]}'
            })
            return None
        
        # Run benchmarks
        try:
            if self.inference_engine.device == 'cpu' and config.model_size in ['7B', '13B', '70B']:
                print("\n⚠️  CPU generation for large models can be very slow. Please allow several minutes per prompt.")
            
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
            
            # Aggregate results
            aggregated = self.inference_engine.aggregate_metrics(all_metrics)
            
            result = {
                'model_name': config.model_name,
                'model_id': config.model_id,
                'model_size': config.size,
                'quality_score': config.quality_score,
                'ram_requirement_gb': config.min_ram_gb,
                'gpu_memory_requirement_gb': config.min_gpu_memory_gb,
                
                'benchmark_date': datetime.now().isoformat(),
                'num_prompts': len(prompts),
                'num_runs': num_runs,
                'total_runs': len(all_metrics),
                
                'metrics': aggregated,
                'device_used': aggregated.get('device', 'unknown'),  # ← Track device
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
                'reason': f'Benchmark error: {str(e)[:50]}'
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
        """
        if models is None:
            models = [MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_7B]
        
        if quantizations is None:
            quantizations = ['fp32']
        
        print("\n" + "="*70)
        print("🚀 STARTING BENCHMARK SUITE")
        print("="*70)
        
        # ✅ CHANGE 2: Show device info at start
        self.device_info = DeviceUtils.print_device_info()
        
        # Benchmark each model
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
        print(f"✅ Benchmarking complete: {successful}/{len(models)} models successful")
        
        if self.failed_models:
            print(f"\n⚠️  {len(self.failed_models)} models failed:")
            for fail in self.failed_models:
                print(f"   - {fail['model']}: {fail['reason']}")
        
        print("="*70)
        
        # Save results
        self.save_results()
        
        # Display comparison
        if self.results:
            self.display_comparison()
        else:
            print("\n⚠️  No successful benchmarks to compare")
        
        return self.results
    
    def _print_benchmark_results(self, result: Dict):
        """Print benchmark results"""
        m = result['metrics']
        
        print(f"\n✅ RESULTS:")
        print(f"{'─'*70}")
        print(f"  Device:               {m.get('device', 'unknown').upper()}")  # ← Show device
        print(f"  Latency (mean):       {m['mean_latency']:.3f}s")
        print(f"  Latency (median):     {m['median_latency']:.3f}s")
        print(f"  Throughput:           {m['mean_tps']:.1f} tokens/sec")
        print(f"  Quality score:        {result['quality_score']:.2f}/1.0")
        print(f"{'─'*70}\n")
    
    def display_comparison(self):
        """Compare models"""
        if not self.results:
            return
        
        print("\n" + "="*130)
        print("📊 MODEL COMPARISON (GPU vs CPU Performance)")
        print("="*130)
        
        print(
            f"{'Model':<25} "
            f"{'Device':<10} "
            f"{'Latency(ms)':<15} "
            f"{'Throughput':<15} "
            f"{'Quality':<10} "
            f"{'vs Expected':<20}"
        )
        print("-"*130)
        
        for key, result in self.results.items():
            m = result['metrics']
            device = m.get('device', 'unknown').upper()
            
            # Compare with expected
            if device == 'CUDA':
                expected = result['expected_tps_gpu']
            else:
                expected = result['expected_tps_cpu']
            
            ratio = (m['mean_tps'] / expected * 100) if expected > 0 else 0
            
            print(
                f"{result['model_name']:<25} "
                f"{device:<10} "
                f"{m['mean_latency']*1000:>6.1f}ms{'':<8} "
                f"{m['mean_tps']:>6.1f} tok/s{'':<8} "
                f"{result['quality_score']:.2f}⭐{'':<7} "
                f"{ratio:.0f}% of expected{'':<6}"
            )
        
        print("="*130)
    
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
                'total_models_attempted': len(self.results) + len(self.failed_models),
                'successful': len(self.results),
                'failed': len(self.failed_models)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
        return filename