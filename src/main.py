"""
main.py

CORRECTED - Entry point with GPU/CPU auto-detection
✅ CHANGE: Uses DeviceUtils for device detection
"""

import json
import os
from pathlib import Path
from src.benchmarking.benchmarking_suite import BenchmarkSuite
from src.models.model_configs import (
    MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_7B,
    print_model_comparison
)
from src.models.device_utils import DeviceUtils


def load_test_prompts(filepath: str = "data/test_prompts.json") -> list:
    """Load test prompts"""
    
    if not Path(filepath).exists():
        test_data = [
            {
                "id": "q1",
                "prompt": "What is machine learning?",
                "reference_answer": "Machine learning is a subset of artificial intelligence..."
            },
            {
                "id": "q2",
                "prompt": "How do neural networks work?",
                "reference_answer": "Neural networks are inspired by biological neurons..."
            },
            {
                "id": "q3",
                "prompt": "Explain quantum computing",
                "reference_answer": "Quantum computing uses quantum bits (qubits)..."
            },
            {
                "id": "q4",
                "prompt": "What is deep learning?",
                "reference_answer": "Deep learning is a type of machine learning..."
            },
            {
                "id": "q5",
                "prompt": "How do transformers work?",
                "reference_answer": "Transformers use self-attention mechanisms..."
            }
        ]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"✓ Created sample test_prompts.json")
    
    with open(filepath, 'r') as f:
        return json.load(f)
    
def check_llama2_access():
    """
    ✅ CHANGE: Check if Llama2 is accessible
    """
    print("\n" + "="*80)
    print("🔐 CHECKING LLAMA2 ACCESS")
    print("="*80)
    
    hf_token_path = Path.home() / ".huggingface" / "token"
    token_env = os.getenv('HUGGING_FACE_TOKEN')
    
    if token_env:
        print("\n✅ HuggingFace token found in environment variable")
        print("   Llama2 should work!")
    elif hf_token_path.exists():
        print("\n✅ HuggingFace token found at ~/.huggingface/token")
        print("   Llama2 should work!")
    else:
        print("\n⚠️  No HuggingFace token found")
        print("\n   To use Llama2, you need to:")
        print("   1. Go to: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf")
        print("   2. Click 'Access repository' button")
        print("   3. Accept the license agreement")
        print("   4. Get token from: https://huggingface.co/settings/tokens")
        print("   5. Run: huggingface-cli login")
        print("   6. Paste your token")
        print("\n   ℹ️  Mistral and Neural Chat don't require this!")
    
    print("="*80 + "\n")


def run_complete_benchmark():
    """
    Complete benchmarking workflow
    ✅ CHANGE: Now uses GPU/CPU auto-detection
    """
    
    print("\n" + "="*80)
    print("🚀 OFFLINE INFERENCE BENCHMARKING SYSTEM")
    print("="*80)
    print("✅ Automatic GPU/CPU Detection\n")
    
    # Step 1: Hardware Detection (GPU/CPU)
    print("📊 Step 1: Hardware Detection & Device Selection")
    print("-"*80)
    # ✅ CHANGE 1: Use DeviceUtils for comprehensive hardware info
    device_info = DeviceUtils.print_device_info()
    
    # ✅ CHANGE 2: Show which device will be used
    optimal_device = DeviceUtils.get_optimal_device()
    print(f"\n✅ Optimal device selected: {optimal_device.upper()}\n")
    
    # Step 2: Models
    print("\n📋 Step 2: Available Models")
    print("-"*80)
    print_model_comparison()
    
    # Step 3: Load prompts
    print("\n📝 Step 3: Load Test Prompts")
    print("-"*80)
    test_data = load_test_prompts()
    prompts = [item['prompt'] for item in test_data]
    print(f"✓ Loaded {len(prompts)} test prompts")
    for i, p in enumerate(prompts[:3], 1):
        print(f"  {i}. {p[:50]}...")
    
    # Step 4: Run benchmarks
    print("\n⚡ Step 4: Run Benchmarks")
    print("-"*80)
    print(f"Running on: {optimal_device.upper()}\n")  # ← Show device
    
    suite = BenchmarkSuite(
        output_dir="benchmark_results/",
        cache_dir="models/"
    )
    
    results = suite.benchmark_all_models(
        prompts=prompts,
        models=[MISTRAL_7B, NEURAL_CHAT_7B, LLAMA2_13B],
        quantizations=['fp32']  # Add 'int8' for GPU quantization
    )
    
    # Step 5: Recommendations
    if results:
        print_recommendations(results, optimal_device)
    else:
        print("\n⚠️  No successful benchmarks")
    
    return results


def print_recommendations(results: dict, device: str):
    """
    Print recommendations based on results
    ✅ CHANGE: Device-specific recommendations
    """
    
    if not results:
        return
    
    print("\n" + "="*80)
    print("🎯 PRODUCTION RECOMMENDATIONS")
    print("="*80)
    print(f"\n📍 Running on: {device.upper()}")
    
    # Find best for each category
    fastest = max(
        results.items(),
        key=lambda x: x[1]['metrics']['mean_tps']
    )
    
    highest_quality = max(
        results.items(),
        key=lambda x: x[1]['quality_score']
    )
    
    # ✅ CHANGE 3: Device-specific recommendations
    if device == 'cuda':
        print(f"\n⚡ FOR REAL-TIME (GPU - Fastest):")
        print(f"   Model: {fastest[1]['model_name']}")
        print(f"   Speed: {fastest[1]['metrics']['mean_tps']:.1f} tokens/sec")
        print(f"   GPU Memory: {fastest[1]['gpu_memory_requirement_gb']:.0f}GB")
        
        print(f"\n🎯 FOR QUALITY (GPU - Best):")
        print(f"   Model: {highest_quality[1]['model_name']}")
        print(f"   Quality: {highest_quality[1]['quality_score']:.2f}/1.0")
        print(f"   Speed: {highest_quality[1]['metrics']['mean_tps']:.1f} tok/sec")
    
    else:  # CPU mode
        print(f"\n✓ Running in CPU-ONLY mode")
        print(f"\n⚡ FOR REAL-TIME (CPU - Fastest):")
        print(f"   Model: {fastest[1]['model_name']}")
        print(f"   Speed: {fastest[1]['metrics']['mean_tps']:.1f} tokens/sec")
        print(f"   RAM: {fastest[1]['ram_requirement_gb']:.0f}GB")
        
        print(f"\n🎯 FOR QUALITY (CPU - Best):")
        print(f"   Model: {highest_quality[1]['model_name']}")
        print(f"   Quality: {highest_quality[1]['quality_score']:.2f}/1.0")
        
        print(f"\n💡 CPU Mode Tips:")
        print(f"   ✓ Mistral 7B: Recommended (3 tok/sec)")
        print(f"   ✓ Neural Chat 7B: Good (2.5 tok/sec)")
        print(f"   ⚠️  Llama2 13B: Slow (1 tok/sec)")
    
    print(f"\n💡 Privacy & Cost:")
    print(f"   ✓ Runs completely offline")
    print(f"   ✓ No API costs")
    print(f"   ✓ No data sent to vendors")
    print(f"   ✓ Works on {device.upper()}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # ✅ CHANGE 4: Show startup info
    print("\n" + "="*80)
    print("🚀 OFFLINE INFERENCE BENCHMARKING SYSTEM")
    print("="*80)
    print("Detecting hardware and device...")
    
    # Check HF token
    token = os.getenv('HUGGING_FACE_TOKEN')
    if not token and not Path(".huggingface/token").exists():
        print("\n⚠️  HuggingFace token not found")
        print("   For Llama2, run: huggingface-cli login")
    
    # Run benchmark
    results = run_complete_benchmark()