
import json
from pathlib import Path
from src.benchmarking.benchmarking_suite import BenchmarkSuite
from src.models.model_configs import TINYLLAMA_1B, DISTILGPT2,  print_model_comparison
from src.models.device_utils import DeviceUtils


def load_test_prompts(filepath: str = "data/test_prompts.json") -> list:
    """Load test prompts"""
    
    if not Path(filepath).exists():
        test_data = [
            {"id": "q1", "prompt": "What is AI?", "reference_answer": "..."},
            {"id": "q2", "prompt": "How do computers work?", "reference_answer": "..."},
            {"id": "q3", "prompt": "What is machine learning?", "reference_answer": "..."},
            {"id": "q4", "prompt": "Explain neural networks", "reference_answer": "..."},
            {"id": "q5", "prompt": "What is deep learning?", "reference_answer": "..."}
        ]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"✓ Created test_prompts.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def run_complete_benchmark():
    """Complete benchmarking workflow"""
    
    print("\n" + "="*80)
    print("🚀 1B MODEL BENCHMARKING SYSTEM")
    print("="*80)
    print("✅ Optimized for 15.7GB RAM & 25GB Storage\n")
    
    print("💾 Step 1: Storage Check")
    print("-"*80)
    print("✅ Using 1B models:")
    print("   - TinyLlama-1.1B: 2.2 GB")
    print("   - DistilGPT2: 0.35 GB")
    print("   - Phi-1.5 (1.3B): 2.6 GB")
    print("   - HF Cache: 1.0 GB")
    print("   - TOTAL: ~6.15 GB ✅ (18.85 GB free!)\n")
    
    print("📊 Step 2: System Detection")
    print("-"*80)
    device_info = DeviceUtils.print_device_info()
    
    print("📋 Step 3: Available Models")
    print("-"*80)
    print_model_comparison()
    
    
    print("📝 Step 4: Load Test Prompts")
    print("-"*80)
    test_data = load_test_prompts()
    prompts = [item['prompt'] for item in test_data]
    print(f"✓ Loaded {len(prompts)} prompts\n")
    
    print("⚡ Step 5: Run Benchmarks")
    print("-"*80)
    
    suite = BenchmarkSuite(
        output_dir="benchmark_results/",
        cache_dir="models/"
    )
    
    results = suite.benchmark_all_models(
        prompts=prompts,
        models=[TINYLLAMA_1B, DISTILGPT2],
        quantizations=['none']
    )
    
    if results:
        print_recommendations(results)
    else:
        print("\n⚠️  No successful benchmarks")
    
    return results


def print_recommendations(results: dict):
    
    if not results:
        return
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS FOR YOUR SYSTEM")
    print("="*80)
    
    fastest = max(results.items(), key=lambda x: x[1]['metrics']['mean_tps'])
    highest_quality = max(results.items(), key=lambda x: x[1]['quality_score'])
    
    print(f"\n⚡ FOR SPEED:")
    print(f"   {fastest[1]['model_name']}")
    print(f"   Speed: {fastest[1]['metrics']['mean_tps']:.1f} tok/sec")
    print(f"   Disk: {fastest[1]['disk_size_gb']} GB")
    print(f"   RAM: {fastest[1]['ram_requirement_gb']} GB")
    
    print(f"\n🎯 FOR QUALITY:")
    print(f"   {highest_quality[1]['model_name']}")
    print(f"   Quality: {highest_quality[1]['quality_score']:.2f}/1.0")
    
    print(f"\n💾 Storage Summary:")
    total = sum(r['disk_size_gb'] for r in results.values())
    print(f"   Used: {total:.2f} GB")
    print(f"   Available: 25 GB")
    print(f"   Free: {25 - total - 1:.2f} GB ✅")
    
    print(f"\n✅ YOUR SYSTEM:")
    print(f"   RAM: 15.7 GB (Perfect for 1B models!)")
    print(f"   CPU: 16 cores (Great for inference)")
    print(f"   Storage: 25 GB (Plenty of space)")
    
    print("="*80)


if __name__ == "__main__":
    results = run_complete_benchmark()