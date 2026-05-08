

---

```markdown
### **Local LLM Benchmarking System**

A comprehensive **performance benchmarking and evaluation framework** for running language models (1B models) on resource-constrained systems. Designed for CPU/GPU optimization with memory efficiency and detailed quality metrics.

> **Perfect for**: Evaluating LLMs on laptops, edge devices, and resource-limited environments

---

## ✨ Features

- 🚀 **Multi-Model Support**: TinyLlama, DistilGPT2, Phi-1.5, Mistral, Llama 2, and more
- 💾 **Memory Optimized**: Aggressive memory management for systems with 15-25GB RAM
- ⚡ **Hardware Detection**: Automatic CPU/GPU device selection and optimization
- 📊 **Comprehensive Benchmarking**: Speed (tokens/sec), quality metrics (BLEU, ROUGE-L), and resource usage
- 🔄 **Quantization Support**: INT8, FP32, and mixed precision inference
- 📈 **Quality Metrics**: BLEU scores, ROUGE-L, and detailed evaluation reports
- 📁 **Model Caching**: Efficient model reuse and management
- 🛡️ **Error Handling**: Graceful fallbacks and detailed troubleshooting guidance
- 🔐 **HuggingFace Integration**: Native support for model authentication and downloading

---

## 📋 System Requirements

### Minimum Specifications
- **RAM**: 15.7 GB (optimized for 25GB+ systems)
- **Storage**: 25 GB (for model caches and results)
- **GPU** (optional): NVIDIA GPU with CUDA support recommended
- **Python**: 3.8+

### Tested Models & Sizes
| Model | Size | Disk | RAM Needed |
|-------|------|------|-----------|
| DistilGPT2 | 82M | 0.35 GB | 2 GB |
| TinyLlama 1.1B | 1.1B | 2.2 GB | 4 GB |
| Phi-1.5 | 1.3B | 2.6 GB | 5 GB |
| Mistral 7B | 7B | 13 GB | 16 GB |
| Llama 2 13B | 13B | 24 GB | 32 GB |

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/local-llm-benchmarking.git
cd local-llm-benchmarking
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r src/requirements.txt
```

### 4. Run Complete Benchmark
```bash
python test.py
```

### 5. View Results
Results are saved to benchmark_results with JSON reports and performance metrics.

---

## 📖 Usage Guide

### Basic Inference
```python
from src.models.model_manager import ModelManager
from src.models.model_configs import DISTILGPT2
from src.inference.inference_engine import InferenceEngine, InferenceConfig

# Initialize components
model_manager = ModelManager(cache_dir="models/")
inference_engine = InferenceEngine()

# Load model
model, tokenizer = model_manager.load_model(DISTILGPT2)

# Configure inference
config = InferenceConfig(
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9
)

# Generate text
prompt = "What is artificial intelligence?"
output, metrics = inference_engine.generate(model, tokenizer, prompt, config)
print(f"Generated: {output}")
print(f"Speed: {metrics['tokens_per_second']:.2f} tok/sec")
```

### Run Benchmarks
```python
from src.benchmarking.benchmarking_suite import BenchmarkSuite
from src.models.model_configs import TINYLLAMA_1B, DISTILGPT2

suite = BenchmarkSuite(output_dir="benchmark_results/")

results = suite.benchmark_all_models(
    prompts=["Your prompt here"],
    models=[TINYLLAMA_1B, DISTILGPT2],
    quantizations=['none']
)
```

### Quality Evaluation
```python
from src.evaluation.quality_metrics import QualityMetrics

reference = "Artificial intelligence is a branch of computer science."
hypothesis = "AI is a computer science field."

bleu = QualityMetrics.bleu_score(reference, hypothesis)
rouge = QualityMetrics.rouge_l(reference, hypothesis)

print(f"BLEU: {bleu:.3f}")
print(f"ROUGE-L: {rouge:.3f}")
```

---

## 📁 Project Structure

```
local-llm-benchmarking/
├── src/
│   ├── main.py                  # Main entry point
│   ├── config.yaml              # Configuration file
│   ├── requirements.txt          # Python dependencies
│   ├── models/
│   │   ├── model_manager.py     # Model loading & management
│   │   ├── model_configs.py     # Predefined model configs
│   │   ├── device_utils.py      # Device detection & optimization
│   │   └── __init__.py
│   ├── benchmarking/
│   │   ├── benchmarking_suite.py # Benchmark orchestration
│   │   └── __init__.py
│   ├── inference/
│   │   ├── inference_engine.py  # Text generation engine
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── quality_metrics.py   # BLEU, ROUGE-L, etc.
│   │   └── __init__.py
│   └── __init__.py
├── models/                       # Model cache directory
├── data/
│   └── test_prompts.json        # Test prompts for benchmarking
├── benchmark_results/            # Output results & reports
├── test.py                       # Complete benchmark runner
└── README.md
```

---

## ⚙️ Configuration

Edit config.yaml to customize behavior:

```yaml
# Model settings
models:
  - name: "Mistral 7B"
    model_id: "mistralai/Mistral-7B-Instruct-v0.3"
    enabled: true

# Inference settings
inference:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.0

# Benchmark settings
benchmark:
  num_prompts: 5
  num_runs_per_prompt: 1
  quantizations:
    - "fp32"
    - "int8"

# Paths
paths:
  data_dir: "data/"
  model_cache: "models/"
  results_dir: "benchmark_results/"

# Hardware
hardware:
  use_gpu: true
  gpu_device: 0
  mixed_precision: true
```

---

## 🔑 Authentication

### HuggingFace Token Setup

For gated models (Llama 2, Mistral, etc.):

```bash
# Method 1: Environment Variable
export HUGGING_FACE_TOKEN="your_token_here"

# Method 2: HuggingFace CLI
huggingface-cli login

# Method 3: .env File
echo "HUGGING_FACE_TOKEN=your_token_here" > .env
```

Get your token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 📊 Benchmark Output

Benchmark results are saved as JSON with detailed metrics:

```json
{
  "model_name": "TinyLlama 1.1B",
  "timestamp": "2026-05-08T12:30:45",
  "metrics": {
    "mean_tps": 12.5,
    "median_tps": 12.3,
    "min_tps": 11.8,
    "max_tps": 13.2,
    "std_tps": 0.45
  },
  "quality_scores": {
    "bleu": 0.68,
    "rouge_l": 0.72
  },
  "memory_usage": {
    "peak_ram_gb": 4.2,
    "gpu_memory_gb": 0
  },
  "prompts_tested": 5,
  "successful_runs": 5
}
```

---

## 🔧 Key Components

### ModelManager
Handles model loading with aggressive memory optimization:
- Automatic device detection (CPU/GPU)
- 8-bit quantization for NVIDIA GPUs
- Sequential loading for CPU efficiency
- Memory cleanup and garbage collection
- Detailed error handling and troubleshooting

### InferenceEngine
Generates text with performance tracking:
- Configurable generation parameters
- Token-per-second measurement
- Automatic device placement
- Batch support

### BenchmarkSuite
Orchestrates comprehensive benchmarking:
- Multi-model evaluation
- Quantization comparison
- Quality metrics calculation
- Result aggregation and reporting

### QualityMetrics
Evaluates output quality:
- **BLEU Score**: N-gram overlap with reference
- **ROUGE-L**: Longest common subsequence
- **Extensible**: Add custom metrics

---

## 🚨 Troubleshooting

### Out of Memory
```
Error: CUDA out of memory

Solutions:
1. Close other applications (Chrome, VS Code, Discord)
2. Restart your PC
3. Use smaller model (DistilGPT2 - 82M)
4. Enable quantization (int8)
```

### Model Download Fails
```
Error: 401 Unauthorized

Solutions:
1. Run: huggingface-cli login
2. Export token: export HUGGING_FACE_TOKEN="your_token"
3. Check token permissions at huggingface.co
```

### Slow Inference
```
Solutions:
1. Verify GPU is being used (not CPU)
2. Check CUDA availability: torch.cuda.is_available()
3. Enable mixed precision (fp16)
4. Reduce max_new_tokens parameter
5. Use smaller model
```

---

## 📈 Performance Tips

1. **GPU Optimization**
   - Use 8-bit quantization to save 75% memory
   - Enable mixed precision (fp16)
   - Batch multiple inferences

2. **CPU Optimization**
   - Use sequential loading (`low_cpu_mem_usage=True`)
   - Reduce `max_new_tokens`
   - Close background applications

3. **Model Selection**
   - Start with DistilGPT2 (82M) for testing
   - Use TinyLlama (1.1B) for balanced quality/speed
   - Scale up based on system capacity

---

## 🔬 API Reference

### ModelManager.load_model()
```python
def load_model(
    config,
    quantization: Optional[str] = None,
    force_reload: bool = False
) -> Optional[Tuple[torch.nn.Module, AutoTokenizer]]
```
Loads a language model with memory optimization.

### InferenceEngine.generate()
```python
def generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    config: InferenceConfig = None
) -> Tuple[str, Dict]
```
Generates text and returns output with performance metrics.

### BenchmarkSuite.benchmark_all_models()
```python
def benchmark_all_models(
    prompts: List[str],
    models: List,
    quantizations: List[str] = ['none']
) -> Dict
```
Benchmarks multiple models with specified quantizations.

---

## 📝 Example: Complete Workflow

```python
import json
from src.benchmarking.benchmarking_suite import BenchmarkSuite
from src.models.model_configs import TINYLLAMA_1B, DISTILGPT2

# 1. Load test prompts
with open("data/test_prompts.json") as f:
    test_data = json.load(f)
    prompts = [item['prompt'] for item in test_data]

# 2. Create benchmark suite
suite = BenchmarkSuite(
    output_dir="benchmark_results/",
    cache_dir="models/"
)

# 3. Run benchmarks
results = suite.benchmark_all_models(
    prompts=prompts,
    models=[TINYLLAMA_1B, DISTILGPT2],
    quantizations=['none', 'int8']
)

# 4. Analyze results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Speed: {metrics['metrics']['mean_tps']:.2f} tok/sec")
    print(f"  Quality (BLEU): {metrics['quality_scores']['bleu']:.3f}")
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)
- [ROUGE Metric](https://en.wikipedia.org/wiki/ROUGE_(metric))

---

## 🙋 Support

- 📖 Read the documentation
- 🐛 Found a bug? Open an issue
- 💡 Have ideas? Start a discussion
- 📧 Email: your.email@example.com

---

## 🎯 Roadmap

- [ ] Support for larger models (7B+)
- [ ] Multi-GPU inference
- [ ] API server deployment
- [ ] Web dashboard for results visualization
- [ ] More quality metrics (METEOR, BERTScore)
- [ ] Streaming inference support
- [ ] Fine-tuning capabilities

---

**Built with ❤️ for LLM enthusiasts and researchers**
```

---

## 📋 Key Sections Included:

✅ **Features** - Highlights what makes your project special  
✅ **System Requirements** - Clear hardware requirements with model comparison table  
✅ **Quick Start** - 5-step setup for new users  
✅ **Usage Guide** - Code examples for common tasks  
✅ **Project Structure** - Directory organization  
✅ **Configuration** - How to customize behavior  
✅ **Authentication** - HuggingFace setup instructions  
✅ **Benchmark Output** - Example results format  
✅ **Troubleshooting** - Solutions for common issues  
✅ **Performance Tips** - Optimization strategies  
✅ **API Reference** - Key functions documented  
✅ **Complete Workflow** - Real-world example  
✅ **Contributing** - How to contribute  
✅ **Roadmap** - Future features  

Copy this content and paste it into your GitHub repository's README.md file!---

## 📋 Key Sections Included:

✅ **Features** - Highlights what makes your project special  
✅ **System Requirements** - Clear hardware requirements with model comparison table  
✅ **Quick Start** - 5-step setup for new users  
✅ **Usage Guide** - Code examples for common tasks  
✅ **Project Structure** - Directory organization  
✅ **Configuration** - How to customize behavior  
✅ **Authentication** - HuggingFace setup instructions  
✅ **Benchmark Output** - Example results format  
✅ **Troubleshooting** - Solutions for common issues  
✅ **Performance Tips** - Optimization strategies  
✅ **API Reference** - Key functions documented  
✅ **Complete Workflow** - Real-world example  
✅ **Contributing** - How to contribute  
✅ **Roadmap** - Future features  

Copy this content and paste it into your GitHub repository's README.md file!
