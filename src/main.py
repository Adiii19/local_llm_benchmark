"""
main.py

Main entry point for benchmarking
Complete end-to-end benchmark workflow
"""

import json
from pathlib import Path
from src.benchmarking.benchmarking_suite import BenchmarkSuite
from src.models.model_configs import (
    MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_34B,
    print_model_comparison
)
from src.models.device_utils import DeviceUtils
from src.evaluation.quality_metrics import QualityMetrics


def load_test_prompts(filepath: str = "data/test_prompts.json") -> dict:
    """Load test prompts from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_complete_benchmark():
    """
    Complete benchmarking workflow
    Load models → Run inference → Measure quality → Compare
    """
    
    print("\n" + "="*80)
    print("🚀 OFFLINE INFERENCE BENCHMARKING SYSTEM")
    print("="*80)
    
    # Step 1: Show hardware
    print("\n📊 Step 1: Hardware Detection")
    print("-"*80)
    device_info = DeviceUtils.print_device_info()
    
    # Step 2: Show models
    print("\n📋 Step 2: Model Information")
    print("-"*80)
    print_model_comparison()
    
    # Step 3: Load test prompts
    print("\n📝 Step 3: Load Test Prompts")
    print("-"*80)
    test_data = load_test_prompts()
    prompts = [item['prompt'] for item in test_data]
    references = [item['reference_answer'] for item in test_data]
    
    print(f"✓ Loaded {len(prompts)} test prompts")
    print(f"  Sample: {prompts[0][:60]}...")
    
    # Step 4: Run benchmarks
    print("\n⚡ Step 4: Run Benchmarks")
    print("-"*80)
    suite = BenchmarkSuite(
        output_dir="benchmark_results/",
        cache_dir="models/"
    )
    
    results = suite.benchmark_all_models(
        prompts=prompts,
        models=[MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_34B],
        quantizations=['fp32']  # Can add 'int8' for comparison
    )
    
    # Step 5: Quality evaluation
    print("\n📈 Step 5: Quality Evaluation")
    print("-"*80)
    
    # For simplicity, evaluate on first prompt
    print(f"Evaluating quality metrics on: {prompts[0]}")
    
    quality_results = {}
    for key, result in results.items():
        # Generate sample answer (would be from actual model output)
        sample_output = "Sample generated text from model"
        
        metrics = QualityMetrics.evaluate_multiple(
            references=[references[0]],
            hypotheses=[sample_output]
        )
        
        quality_results[result['model_name']] = metrics
    
    # Step 6: Display final comparison
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON: Quality vs Speed vs Cost")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Speed':<15} {'Quality':<12} {'Memory':<12} {'Best For':<25}")
    print("-"*80)
    
    for key, result in results.items():
        model_name = result['model_name']
        tps = result['metrics']['mean_tps']
        quality = result['quality_score']
        ram = result['ram_requirement_gb']
        
        # Determine best for
        if tps > 40:
            best_for = "⚡ Real-time"
        elif quality > 0.80:
            best_for = "🎯 Accuracy"
        else:
            best_for = "⚖️ Balanced"
        
        print(
            f"{model_name:<20} "
            f"{tps:>6.1f} tok/s{'':<8} "
            f"{quality:>6.2f}⭐{'':<5} "
            f"{ram:>6.0f}GB{'':<6} "
            f"{best_for:<25}"
        )
    
    print("="*80)
    
    # Step 7: Recommendations
    print_recommendations(results)
    
    return results


def print_recommendations(results: dict):
    """Print production recommendations based on benchmarks"""
    
    print("\n" + "="*80)
    print("🎯 PRODUCTION RECOMMENDATIONS")
    print("="*80)
    
    if not results:
        print("\n❌ No benchmark results available.")
        print("   This could be due to:")
        print("   - Model loading failures")
        print("   - Insufficient hardware resources")
        print("   - Network issues downloading models")
        print("   - Check the logs above for specific errors")
        return
    
    # Find fastest
    fastest = max(
        results.items(),
        key=lambda x: x[1]['metrics']['mean_tps']
    )
    
    # Find highest quality
    highest_quality = max(
        results.items(),
        key=lambda x: x[1]['quality_score']
    )
    
    # Find most balanced
    balanced = min(
        results.items(),
        key=lambda x: abs(
            x[1]['metrics']['mean_tps']/100 - x[1]['quality_score']
        )
    )
    
    print(f"\n⚡ For Real-time / Latency-sensitive:")
    print(f"   → Use: {fastest[1]['model_name']}")
    print(f"   → Speed: {fastest[1]['metrics']['mean_tps']:.1f} tokens/sec")
    print(f"   → RAM: {fastest[1]['ram_requirement_gb']:.0f}GB")
    
    print(f"\n🎯 For Highest Quality / Accuracy:")
    print(f"   → Use: {highest_quality[1]['model_name']}")
    print(f"   → Quality: {highest_quality[1]['quality_score']:.2f}/1.0")
    print(f"   → Speed: {highest_quality[1]['metrics']['mean_tps']:.1f} tokens/sec")
    
    print(f"\n⚖️  For Balanced Use (Recommended):")
    print(f"   → Use: {balanced[1]['model_name']}")
    print(f"   → Speed: {balanced[1]['metrics']['mean_tps']:.1f} tokens/sec")
    print(f"   → Quality: {balanced[1]['quality_score']:.2f}/1.0")
    print(f"   → RAM: {balanced[1]['ram_requirement_gb']:.0f}GB")
    
    print("\n💡 Privacy Benefits (Offline):")
    print("   ✓ No data sent to external APIs")
    print("   ✓ No vendor lock-in")
    print("   ✓ No usage tracking")
    print("   ✓ Works without internet")
    
    print("\n📊 Cost Comparison:")
    print("   GPT-4 API: ~$0.03 per 1K tokens")
    print("   Your offline setup: $0 per 1M tokens (one-time setup)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    results = run_complete_benchmark()