"""
src/models/model_configs.py

Model configurations and specifications
Defines the 3 models we'll benchmark
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum


class ModelSize(Enum):
    """Model size categories"""
    TINY = "7B"      # Fast, lower quality
    SMALL = "13B"    # Balanced
    MEDIUM = "34B"   # Better quality, slower
    LARGE = "70B"    # Best quality, needs GPU


@dataclass
class ModelConfig:
    """Configuration for a model"""
    
    # Basic info
    model_id: str                    # HuggingFace model ID
    model_name: str                  # Display name
    size: str                        # Parameter count (7B, 13B, etc)
    model_type: str                  # Type (llama, mistral, etc)
    
    # Requirements
    min_ram_gb: float               # Minimum RAM needed
    min_gpu_memory_gb: Optional[float]  # Minimum GPU memory needed
    
    # Performance characteristics
    tokens_per_second_cpu: float    # Expected speed on CPU
    tokens_per_second_gpu: float    # Expected speed on GPU
    quality_score: float            # 0-1 quality rating
    
    # Features
    context_length: int             # Max token context
    supports_quantization: bool     # Can be quantized
    
    # Details
    description: str
    best_for: str                   # Use case
    
    def __str__(self):
        return f"{self.model_name} ({self.size})"


# ============================================================================
# THE 3 MODELS WE'LL BENCHMARK
# ============================================================================

# MODEL 1: FASTEST (Best for latency-sensitive apps)
MISTRAL_7B = ModelConfig(
    model_id="facebook/opt-1.3b",
    model_name="OPT 1.3B",
    size="1.3B",
    model_type="opt",
    min_ram_gb=4,
    min_gpu_memory_gb=1,
    tokens_per_second_cpu=5,      # ~5 tokens/sec on CPU
    tokens_per_second_gpu=100,    # ~100 tokens/sec on GPU
    quality_score=0.70,           # Decent quality
    context_length=2048,
    supports_quantization=True,
    description="Fast, efficient model. Great for real-time apps.",
    best_for="Chatbots, quick responses, edge devices"
)

# MODEL 2: BALANCED (Best for most use cases)
LLAMA2_13B = ModelConfig(
    model_id="facebook/opt-2.7b",
    model_name="OPT 2.7B",
    size="2.7B",
    model_type="opt",
    min_ram_gb=6,
    min_gpu_memory_gb=2,
    tokens_per_second_cpu=3,      # ~3 tokens/sec on CPU
    tokens_per_second_gpu=80,     # ~80 tokens/sec on GPU
    quality_score=0.75,           # Better quality
    context_length=2048,
    supports_quantization=True,
    description="Good balance of quality and speed.",
    best_for="General purpose, RAG systems, production apps"
)

# MODEL 3: HIGHEST QUALITY (Best for accuracy)
NEURAL_CHAT_34B = ModelConfig(
    model_id="facebook/opt-6.7b",
    model_name="OPT 6.7B",
    size="6.7B",
    model_type="opt",
    min_ram_gb=10,
    min_gpu_memory_gb=3,
    tokens_per_second_cpu=2,
    tokens_per_second_gpu=60,
    quality_score=0.80,
    context_length=32768,
    supports_quantization=True,
    description="Optimized for conversational quality.",
    best_for="Customer service, detailed responses, knowledge tasks"
)

# ============================================================================
# QUANTIZATION CONFIGS (Compress models for speed/memory)
# ============================================================================

@dataclass
class QuantizationConfig:
    """Model quantization settings"""
    method: str                     # 'int8', 'int4', 'fp16', 'fp32'
    bits: int                       # 4, 8, 16, 32
    reduction_ratio: float          # Memory reduction (e.g., 0.5 = 50% smaller)
    speed_improvement: float        # Speed improvement (e.g., 1.5 = 50% faster)
    quality_loss: float            # Quality degradation (0-1)
    
    def __str__(self):
        return f"{self.method} ({self.bits}-bit)"


QUANTIZATION_OPTIONS = {
    'fp32': QuantizationConfig(
        method='fp32',
        bits=32,
        reduction_ratio=1.0,
        speed_improvement=1.0,
        quality_loss=0.0
    ),
    'fp16': QuantizationConfig(
        method='fp16',
        bits=16,
        reduction_ratio=0.5,
        speed_improvement=1.2,
        quality_loss=0.02
    ),
    'int8': QuantizationConfig(
        method='int8',
        bits=8,
        reduction_ratio=0.25,
        speed_improvement=2.0,
        quality_loss=0.05
    ),
    'int4': QuantizationConfig(
        method='int4',
        bits=4,
        reduction_ratio=0.125,
        speed_improvement=3.5,
        quality_loss=0.10
    )
}


# ============================================================================
# MODEL COMPARISON TABLE
# ============================================================================

def print_model_comparison():
    """Print comparison of all 3 models"""
    models = [MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_34B]
    
    print("=" * 100)
    print("📊 MODEL COMPARISON")
    print("=" * 100)
    print(f"{'Model':<20} {'Size':<8} {'Quality':<10} {'Speed (GPU)':<15} {'RAM':<10} {'Best For':<25}")
    print("-" * 100)
    
    for model in models:
        print(
            f"{model.model_name:<20} "
            f"{model.size:<8} "
            f"{model.quality_score:.2f}⭐{'':<7} "
            f"{model.tokens_per_second_gpu:.0f} tok/s{'':<5} "
            f"{model.min_ram_gb:.0f}GB{'':<6} "
            f"{model.best_for:<25}"
        )
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print_model_comparison()
    
    print("\n📌 Model Specifications:\n")
    for model in [MISTRAL_7B, LLAMA2_13B, NEURAL_CHAT_34B]:
        print(f"\n{model.model_name}:")
        print(f"  ID: {model.model_id}")
        print(f"  Description: {model.description}")
        print(f"  Quality: {model.quality_score:.2f}/1.0")
        print(f"  Speed (CPU): {model.tokens_per_second_cpu:.1f} tok/s")
        print(f"  Speed (GPU): {model.tokens_per_second_gpu:.1f} tok/s")
        print(f"  Requirements: {model.min_ram_gb}GB RAM, {model.min_gpu_memory_gb or 'CPU only'}GB GPU")