from dataclasses import dataclass
from typing import Optional,Dict
from enum import Enum

class ModelSize(Enum):
    TINY="7B"
    SMALL="13B"
    MEDIUM="34B"
    LARGE="70B"

@dataclass
class ModelConfig:

    model_id:str
    model_name:str
    size:str
    model_type:str

    min_ram_gb:float
    min_gpu_memory_gb:Optional[float]

    tokens_per_second_cpu:float
    tokens_per_second_gpu:float
    quality_score:float

    context_length:int
    supports_quantization:bool

    description:str
    best_for:str

    def __str__(self):
        return f"{self.model_name} ({self.size})"
    

MISTRAL_7B=ModelConfig(

    model_id="mistral/Mistral-7B-Instruct-vO.1",
    model_name="Mistral 7B",
    size="7B",
    model_type="mistral",
    min_ram_gb=8,
    min_gpu_memory_gb=2,
    tokens_per_second_cpu=2,
    tokens_per_second_gpu=50,
    quality_score=0.75,
    context_length=32766,
    supports_quantization=True,
    description="Fastest  efficient model. Great for real-time apps.",
    best_for="Chatbots, quick responses, edge devices"


)

LLAMA2_13B=ModelConfig(

        model_id="meta-llama/Llama-2-13b-chat-hf",
        model_name="Llama 2 13B",
        size="13B",
        model_type="llama",
        min_ram_gb=12,
        min_gpu_memory_gb=4,
        tokens_per_second_cpu=1,
        tokens_per_second_gpu=30,
        quality_score=0.82,
        context_length=4096,
        supports_quantization=True,
        description="Good balance of quality and speed",
        best_for="General purpose, RAG system, production apps"

)

NEURAL_CHAT_34B = ModelConfig(
    model_id="Intel/neural-chat-7b-v3-1",  # Using 7B version for practicality
    model_name="Neural Chat 7B",
    size="7B",
    model_type="neural-chat",
    min_ram_gb=8,
    min_gpu_memory_gb=2,
    tokens_per_second_cpu=2,
    tokens_per_second_gpu=45,
    quality_score=0.78,
    context_length=32768,
    supports_quantization=True,
    description="Optimized for conversational quality.",
    best_for="Customer service, detailed responses, knowledge tasks"
)

@dataclass
class QuantizationCongfig:
    method:str
    bits=int
    reduction_ratio=float
    speed_improvement:float
    quality_loss:float

    def __str__(self):
        return f"{self.method} ({self.bits}-bit)"
    
QUANTIZATION_OPTIONS={

      'fp_32':QuantizationCongfig(
        method='fp32',
        bits=32,
        reduction_ratio=1.0,
        speed_improvement=1.0,
        quality_loss=0.0
    ),
    'fp16':QuantizationCongfig(
        method='fp16',
        bits=16,
        reduction_ratio=0.5,
        speed_improvement=1.2,
        quality_loss=0.02
    ),

   'int8': QuantizationCongfig(
        method='int8',
        bits=8,
        reduction_ratio=0.25,
        speed_improvement=2.0,
        quality_loss=0.05
    ),
    'int4': QuantizationCongfig(
        method='int4',
        bits=4,
        reduction_ratio=0.125,
        speed_improvement=3.5,
        quality_loss=0.10
    )

}
models=[MISTRAL_7B,LLAMA2_13B,NEURAL_CHAT_34B]

def print_model_comparison():
    models=[MISTRAL_7B,LLAMA2_13B,NEURAL_CHAT_34B]

    print("=" * 100)
    print("📊 MODEL COMPARISON")
    print("=" * 100)
    print(f"{'Model':<20} {'Size':<8} {'Quality':<10} {'Speed (GPU)':<15} {'RAM':<10} {'Best For':<25}")
    print("-" * 100)


    for model in models:
        print(

            f"{model.model_name:<20}"
            f"{model.size:<8} "
            f"{model.quality_score:.2f}⭐{'':<7} "
            f"{model.tokens_per_second_gpu:.0f} tok/s{'':<5} "
            f"{model.min_ram_gb:.0f}GB{'':<6} "
            f"{model.best_for:<25}"
        )
    
    print("=" * 100 + "\n")

if __name__=="__main__":
    print_model_comparison()

    print("\n📌 Model Specifications:\n")

    for model in models:
        print(f"\n{model.model_name}:")
        print(f"  ID: {model.model_id}")
        print(f"  Description: {model.description}")
        print(f"  Quality: {model.quality_score:.2f}/1.0")
        print(f"  Speed (CPU): {model.tokens_per_second_cpu:.1f} tok/s")
        print(f"  Speed (GPU): {model.tokens_per_second_gpu:.1f} tok/s")
        print(f"  Requirements: {model.min_ram_gb}GB RAM, {model.min_gpu_memory_gb or 'CPU only'}GB GPU")

    

