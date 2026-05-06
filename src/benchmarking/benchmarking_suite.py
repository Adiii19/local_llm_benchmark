

import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from ..models.model_configs import TINYLLAMA_1B, DISTILGPT2
from ..models.model_manager import ModelManager
from ..inference.inference_engine import InferenceEngine
from ..models.device_utils import DeviceUtils


class BenchmarkSuite:
   
    
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
        """Benchmark a single 1B model"""
        
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARKING: {config.model_name}")
        print(f"{'='*70}")
        print(f"Size: {config.size}")
        print(f"Disk: {config.disk_size_gb} GB")
        print(f"RAM: {config.min_ram_gb} GB")
        print(f"Prompts: {len(prompts)}")
        
        model_result = self.model_manager.load_model(config)
        
        if model_result is None:
            print(f"\n❌ SKIPPING {config.model_name}")
            self.failed_models.append({'model': config.model_name, 'reason': 'Failed to load'})
            return None
        
        try:
            model, tokenizer = model_result
        except Exception as e:
            print(f"\n❌ ERROR unpacking: {e}")
            self.failed_models.append({'model': config.model_name, 'reason': 'Unpack error'})
            return None
        
        try:
            all_metrics = []
            
            for run in range(num_runs):
                if num_runs > 1:
                    print(f"\n📍 Run {run + 1}/{num_runs}")
                
                _, metrics_list = self.inference_engine.batch_generate(model, tokenizer, prompts)
                all_metrics.extend(metrics_list)
            
            aggregated = self.inference_engine.aggregate_metrics(all_metrics)
            
            result = {
                'model_name': config.model_name,
                'model_id': config.model_id,
                'model_size': config.size,
                'disk_size_gb': config.disk_size_gb,
                'ram_requirement_gb': config.min_ram_gb,
                'quality_score': config.quality_score,
                'metrics': aggregated,
                'device_used': aggregated.get('device', 'unknown'),
                'benchmark_date': datetime.now().isoformat(),
            }
            
            self.results[config.model_name] = result
            self._print_benchmark_results(result)
            self.model_manager.unload_model(config.model_id)
            
            return result
        
        except Exception as e:
            print(f"\n❌ Benchmarking failed: {e}")
            self.failed_models.append({'model': config.model_name, 'reason': 'Benchmark error'})
            self.model_manager.unload_model(config.model_id)
            return None
    
    def benchmark_all_models(
        self,
        prompts: List[str],
        models: List = None,
        quantizations: List[str] = None
    ) -> Dict:
        """Benchmark all 1B models"""
        
        if models is None:
            models = [TINYLLAMA_1B, DISTILGPT2, PHLORA_1B]
        
        if quantizations is None:
            quantizations = ['none']
        
        print("\n" + "="*70)
        print("🚀 1B MODEL BENCHMARK SUITE")
        print("="*70)
        print("✅ Using 1B models (2-3 GB each)")
        
        self.device_info = DeviceUtils.print_device_info()
        
        successful = 0
        for model_config in models:
            for quant in quantizations:
                result = self.benchmark_model(model_config, prompts, quantization=quant, num_runs=1)
                if result is not None:
                    successful += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Done: {successful}/{len(models)} models successful")
        print("="*70)
        
        self.save_results()
        
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
        print(f"  Device:    {m.get('device', 'unknown').upper()}")
        print(f"  Disk:      {result['disk_size_gb']} GB")
        print(f"  RAM:       {result['ram_requirement_gb']} GB")
        print(f"  Latency:   {m['mean_latency']:.3f}s")
        print(f"  Speed:     {m['mean_tps']:.1f} tokens/sec")
        print(f"  Quality:   {result['quality_score']:.2f}/1.0")
        print(f"{'─'*70}\n")
    
    def display_comparison(self):
        """Compare models"""
        if not self.results:
            return
        
        print("\n" + "="*140)
        print("📊 1B MODELS COMPARISON")
        print("="*140)
        
        print(
            f"{'Model':<30} "
            f"{'Size':<12} "
            f"{'Disk':<10} "
            f"{'RAM':<10} "
            f"{'Device':<10} "
            f"{'Latency':<12} "
            f"{'Speed':<15} "
            f"{'Quality':<10}"
        )
        print("-"*140)
        
        for key, result in self.results.items():
            m = result['metrics']
            device = m.get('device', 'unknown').upper()
            
            print(
                f"{result['model_name']:<30} "
                f"{result['model_size']:<12} "
                f"{result['disk_size_gb']:.2f}GB{'':<5} "
                f"{result['ram_requirement_gb']:.0f}GB{'':<6} "
                f"{device:<10} "
                f"{m['mean_latency']*1000:>6.1f}ms{'':<5} "
                f"{m['mean_tps']:>6.1f} tok/s{'':<8} "
                f"{result['quality_score']:.2f}⭐"
            )
        
        print("="*140)
        total = sum(r['disk_size_gb'] for r in self.results.values())
        print(f"\n💾 Total Storage: {total:.2f} GB ✅")
        print(f"   Available: 25 GB")
        print(f"   Free: {25 - total - 1:.2f} GB")
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_{timestamp}.json"
        
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'failed_models': self.failed_models,
            'summary': {
                'successful': len(self.results),
                'failed': len(self.failed_models),
                'total_disk_used_gb': sum(r.get('disk_size_gb', 0) for r in self.results.values())
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✓ Results saved to: {filename}")