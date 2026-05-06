from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    
    model_id: str
    model_name: str
    size: str
    model_type: str
    min_ram_gb: float
    min_gpu_memory_gb: Optional[float]
    tokens_per_second_cpu: float
    tokens_per_second_gpu: float
    quality_score: float
    context_length: int
    description: str
    best_for: str
    quantization_type: str
    disk_size_gb: float


DISTILGPT2 = ModelConfig(
    model_id="distilgpt2",
    model_name="DistilGPT2",
    size="82M",
    model_type="gpt2",
    min_ram_gb=1,
    min_gpu_memory_gb=0.3,
    tokens_per_second_cpu=25,
    tokens_per_second_gpu=200,
    quality_score=0.55,
    context_length=1024,
    description="SMALLEST model. Fits easily in 15.7GB RAM.",
    best_for="Your system - fastest, lowest memory",
    quantization_type="fp32",
    disk_size_gb=0.35
)


TINYLLAMA_1B = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_name="TinyLlama-1.1B",
    size="1.1B",
    model_type="llama",
    min_ram_gb=3,  # ← INCREASED buffer
    min_gpu_memory_gb=0.5,
    tokens_per_second_cpu=12,
    tokens_per_second_gpu=120,
    quality_score=0.68,
    context_length=2048,
    description="Small model with better quality. Requires clean system.",
    best_for="Only if you close ALL other programs first",
    quantization_type="fp32",
    disk_size_gb=2.2
)

# ✅ ALTERNATIVE IF OTHERS FAIL
PHI_MINI = ModelConfig(
    model_id="microsoft/phi-2",
    model_name="Phi-2 (Mini)",
    size="2.7B",
    model_type="phi",
    min_ram_gb=3,
    min_gpu_memory_gb=1,
    tokens_per_second_cpu=8,
    tokens_per_second_gpu=80,
    quality_score=0.70,
    context_length=2048,
    description="Good quality but needs more RAM",
    best_for="Only if system is completely clean",
    quantization_type="fp32",
    disk_size_gb=5.4
)


def print_model_comparison():
    models = [DISTILGPT2, TINYLLAMA_1B, PHI_MINI]
    
    print("=" * 160)
    print("📊 MODELS FOR YOUR 15.7GB RAM SYSTEM")
    print("=" * 160)
    print(
        f"{'Model':<30} "
        f"{'Size':<12} "
        f"{'Disk':<10} "
        f"{'RAM':<12} "
        f"{'Quality':<12} "
        f"{'Speed':<15} "
        f"{'Recommendation':<35}"
    )
    print("-" * 160)
    
    recommendations = [
        "⭐ START HERE",
        "⚠️  If RAM freed",
        "❌ Avoid"
    ]
    
    for model, rec in zip(models, recommendations):
        print(
            f"{model.model_name:<30} "
            f"{model.size:<12} "
            f"{model.disk_size_gb:.2f}GB{'':<5} "
            f"{model.min_ram_gb:.1f}GB{'':<6} "
            f"{model.quality_score:.2f}⭐{'':<7} "
            f"{model.tokens_per_second_cpu:.1f} tok/s{'':<7} "
            f"{rec:<35}"
        )
    
    print("=" * 160)
    print("\n💡 RECOMMENDED APPROACH:")
    print("   1. START with DistilGPT2 (safest)")
    print("   2. If it works, close all programs and try TinyLlama")
    print("   3. Never use Phi-2 (too large for your RAM)")
    print()


if __name__ == "__main__":
    print_model_comparison()