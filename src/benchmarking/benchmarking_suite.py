"""
src/benchmarking/benchmark_suite.py

Complete benchmarking suite
Compare models on same hardware
"""

import json
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from ..models.model_configs import ModelConfig, MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_34B
from ..models.model_manager import ModelManager
from ..inference.inference_engine import InferenceEngine, InferenceConfig
from ..models.device_utils import DeviceUtils


class BenchmarkSuite:
    """
    Complete benchmarking framework
    Run models, collect metrics, compare results
    """
    
    def __init__(self, output_dir: str = "benchmark_results/", cache_dir: str = "models/"):
        """
        Initialize benchmark suite
        
        Args:
            output_dir: Where to save results
            cache_dir: Model cache directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_manager = ModelManager(cache_dir=cache_dir)
        self.inference_engine = InferenceEngine(
            device=DeviceUtils.get_optimal_device()
        )
        
        self.results = {}
        self.device_info = None
    
    def benchmark_model(
        self,
        config: ModelConfig,
        prompts: List[str],
        quantization: str = None,
        num_runs: int = 1
    ) -> Dict:
        """
        Benchmark a single model
        
        Args:
            config: ModelConfig for the model
            prompts: List of test prompts
            quantization: Quantization method ('int4', 'int8', 'fp16', None)
            num_runs: Number of iterations per prompt
        
        Returns:
            Benchmark results
        """
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARKING: {config.model_name}")
        print(f"{'='*70}")
        print(f"Quantization: {quantization or 'fp32'}")
        print(f"Prompts: {len(prompts)}")
        print(f"Runs per prompt: {num_runs}")
        print(f"{'='*70}\n")
        
        # Load model
        try:
            model, tokenizer = self.model_manager.load_model(
                config,
                quantization=quantization
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
        
        # Run benchmarks
        all_metrics = []
        
        for run in range(num_runs):
            if num_runs > 1:
                print(f"\n📍 Run {run + 1}/{num_runs}")
            
            # Generate for all prompts
            _, metrics_list = self.inference_engine.batch_generate(
                model,
                tokenizer,
                prompts
            )
            
            all_metrics.extend(metrics_list)
        
        # Aggregate results
        aggregated = self.inference_engine.aggregate_metrics(all_metrics)
        
        # Package results
        result = {
            'model_name': config.model_name,
            'model_id': config.model_id,
            'model_size': config.size,
            'quantization': quantization or 'fp32',
            'quality_score': config.quality_score,
            'ram_requirement_gb': config.min_ram_gb,
            'gpu_memory_requirement_gb': config.min_gpu_memory_gb,
            
            'benchmark_date': datetime.now().isoformat(),
            'device': DeviceUtils.get_optimal_device(),
            'num_prompts': len(prompts),
            'num_runs': num_runs,
            'total_runs': len(all_metrics),
            
            'metrics': aggregated,
            'expected_tps_cpu': config.tokens_per_second_cpu,
            'expected_tps_gpu': config.tokens_per_second_gpu
        }
        
        # Store
        self.results[config.model_name] = result
        
        # Display results
        self._print_benchmark_results(result)
        
        # Unload model to free memory
        self.model_manager.unload_model(config.model_id)
        
        return result
    
    def benchmark_all_models(
        self,
        prompts: List[str],
        models: List[ModelConfig] = None,
        quantizations: List[str] = None
    ) -> Dict:
        """
        Benchmark all models
        
        Args:
            prompts: Test prompts
            models: Which models to test (default: 3 benchmark models)
            quantizations: Which quantizations to test
        
        Returns:
            All results
        """
        if models is None:
            models = [MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_34B]
        
        if quantizations is None:
            quantizations = ['fp32']  # or ['fp32', 'int8'] for comparison
        
        print("\n" + "="*70)
        print("🚀 STARTING BENCHMARK SUITE")
        print("="*70)
        
        # Show device info
        self.device_info = DeviceUtils.print_device_info()
        
        # Benchmark each model
        for model_config in models:
            for quant in quantizations:
                result = self.benchmark_model(
                    model_config,
                    prompts,
                    quantization=quant,
                    num_runs=1
                )
                
                if result:
                    # Store with unique key
                    key = f"{model_config.model_name}_{quant}"
                    self.results[key] = result
        
        # Save results
        self.save_results()
        
        # Display comparison
        self.display_comparison()
        
        return self.results
    
    def _print_benchmark_results(self, result: Dict):
        """Print formatted benchmark results"""
        m = result['metrics']
        
        print(f"\n✅ RESULTS:")
        print(f"{'─'*70}")
        print(f"  Latency (P50):        {m['median_latency']:.3f}s")
        print(f"  Latency (P95):        {max(m['max_latency'], m['median_latency']):.3f}s")
        print(f"  Throughput:           {m['mean_tps']:.1f} tokens/sec")
        print(f"  Output tokens:        {m['mean_output_tokens']:.0f} avg")
        print(f"  Quality score:        {result['quality_score']:.2f}/1.0")
        print(f"{'─'*70}\n")
    
    def display_comparison(self):
        """Display side-by-side model comparison"""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*100)
        print("📊 MODEL COMPARISON")
        print("="*100)
        
        # Header
        print(f"{'Model':<25} {'Quantization':<15} {'Latency(ms)':<15} {'Throughput':<15} {'Quality':<10}")
        print("-"*100)
        
        # Results
        for key, result in self.results.items():
            m = result['metrics']
            print(
                f"{result['model_name']:<25} "
                f"{result['quantization']:<15} "
                f"{m['median_latency']*1000:>6.1f}ms{'':<8} "
                f"{m['mean_tps']:>6.1f} tok/s{'':<8} "
                f"{result['quality_score']:.2f}⭐"
            )
        
        print("="*100)
    
    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'device_info': self.device_info,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
        
        return filename


if __name__ == "__main__":
    # Sample test prompts
    sample_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms",
        "How do neural networks work?",
        "What is the difference between AI and ML?",
        "Explain blockchain technology"
    ]
    
    # Run benchmarks
    suite = BenchmarkSuite()
    results = suite.benchmark_all_models(sample_prompts)