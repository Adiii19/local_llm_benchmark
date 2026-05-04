"""
main.py

✅ UPDATED: Uses working INT8 quantized models
"""

import json
import os
from pathlib import Path
from src.benchmarking.benchmarking_suite import BenchmarkingSuite
from src.models.model_configs import (
    MISTRAL_7B_INT8, NEURAL_CHAT_7B_INT8, DISTIL_GPT2,  # ← Working models
    print_model_comparison
)
from src.models.device_utils import DeviceUtils


def load_test_prompts(filepath: str = "data/test_prompts.json") -> list:
    """Load test prompts"""
    
    if not Path(filepath).exists():
        test_data = [
            {"id": "q1", "prompt": "What is machine learning?", "reference_answer": "..."},
            {"id": "q2", "prompt": "How do neural networks work?", "reference_answer": "..."},
            {"id": "q3", "prompt": "Explain quantum computing", "reference_answer": "..."},
            {"id": "q4", "prompt": "What is deep learning?", "reference_answer": "..."},
            {"id": "q5", "prompt": "How do transformers work?", "reference_answer": "..."}
        ]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"✓ Created test_prompts.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def run_complete_benchmark():
    """
    Complete benchmarking workflow
    ✅ CHANGE: Uses working INT8 models
    """
    
    print("\n" + "="*80)
    print("🚀 LIGHTWEIGHT INFERENCE BENCHMARKING SYSTEM")
    print("="*80)
    print("✅ INT8 QUANTIZED MODELS - 25GB STORAGE\n")
    
    # Step 1: Storage
    print("💾 Step 1: Storage Optimization")
    print("-"*80)
    print("✅ Using INT8 quantized models:")
    print("   - Mistral 7B (INT8): 3.5 GB")
    print("   - Neural Chat 7B (INT8): 3.5 GB")
    print("   - DistilGPT2: 0.35 GB")
    print("   - HF Cache: 1.0 GB")
    print("   - TOTAL: ~8.35 GB (16.65 GB free!)\n")
    
    # Step 2: Hardware
    print("📊 Step 2: Hardware Detection")
    print("-"*80)
    device_info = DeviceUtils.print_device_info()
    
    optimal_device = DeviceUtils.get_optimal_device()
    print(f"✅ Device: {optimal_device.upper()}\n")
    
    # Step 3: Models
    print("\n📋 Step 3: Available Models")
    print("-"*80)
    print_model_comparison()
    
    # Step 4: Prompts
    print("\n📝 Step 4: Load Test Prompts")
    print("-"*80)
    test_data = load_test_prompts()
    prompts = [item['prompt'] for item in test_data]
    print(f"✓ Loaded {len(prompts)} prompts\n")
    
    # Step 5: Run
    print("\n⚡ Step 5: Run Benchmarks")
    print("-"*80)
    print(f"Device: {optimal_device.upper()}\n")
    
    suite = BenchmarkingSuite(
        output_dir="benchmark_results/",
        cache_dir="models/"
    )
    
    # ✅ CHANGE: Use working INT8 models
    results = suite.benchmark_all_models(
        prompts=prompts,
        models=[MISTRAL_7B_INT8, NEURAL_CHAT_7B_INT8, DISTIL_GPT2],
        quantizations=['auto']
    )
    
    # Step 6: Recommendations
    if results:
        print_recommendations(results, optimal_device)
    else:
        print("\n⚠️  No successful benchmarks")
    
    return results


def print_recommendations(results: dict, device: str):
    """Print recommendations"""
    
    if not results:
        return
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    print(f"Device: {device.upper()}\n")
    
    fastest = max(results.items(), key=lambda x: x[1]['metrics']['mean_tps'])
    highest_quality = max(results.items(), key=lambda x: x[1]['quality_score'])
    
    print(f"⚡ FOR SPEED:")
    print(f"   {fastest[1]['model_name']}")
    print(f"   Speed: {fastest[1]['metrics']['mean_tps']:.1f} tok/s")
    print(f"   Disk: {fastest[1]['disk_size_gb']} GB\n")
    
    print(f"🎯 FOR QUALITY:")
    print(f"   {highest_quality[1]['model_name']}")
    print(f"   Quality: {highest_quality[1]['quality_score']:.2f}/1.0\n")
    
    total = sum(r['disk_size_gb'] for r in results.values())
    print(f"💾 Storage: {total:.2f} GB used (leaves {25 - total - 1:.2f} GB free)")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 LIGHTWEIGHT BENCHMARKING")
    print("="*80)
    
    results = run_complete_benchmark()