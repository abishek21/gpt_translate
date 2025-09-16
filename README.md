# GPT Translation System

A low-latency translation system using GPT OSS 20B model from Hugging Face for French to English translation with detailed latency measurements.

## Features

- **Low Latency Translation**: Optimized for minimal translation delay
- **GPU Acceleration**: Designed to run efficiently on GPU hardware
- **Detailed Metrics**: Comprehensive latency measurements including:
  - Time to first token
  - Time to last token
  - Total translation time
  - Tokenization time
  - Generation time
- **French Sample Sentences**: Includes 8 sample French sentences (20-30 words each)
- **Performance Analysis**: Detailed performance statistics and throughput metrics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abishek21/gpt_translate.git
cd gpt_translate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Mock Implementation (for testing/demonstration)

Run the mock implementation to see the system architecture and latency measurement approach:

```bash
python mock_translator.py
```

This will demonstrate:
- Translation system architecture
- Latency measurement methodology
- Performance metrics and analysis
- Sample French to English translations

### Production Implementation

For production use with actual GPT OSS 20B model:

```bash
python gpt_translator.py
```

**Note**: The production implementation requires:
- Internet access to download the GPT OSS 20B model
- Sufficient GPU memory (approximately 40GB+ for 20B model)
- Appropriate Hugging Face model access tokens if required

## Sample Output

```
=== PERFORMANCE SUMMARY ===
Average total time:        329.8ms
Average time to 1st token: 54.5ms
Average generation time:   318.1ms
Translation throughput:    3.0 sentences/sec
```

## Architecture

The system consists of:

1. **GPTTranslator Class**: Main translation engine
   - Model loading and initialization
   - GPU optimization
   - Latency measurement infrastructure

2. **Translation Pipeline**:
   - Input tokenization
   - Model inference with timing
   - Output generation and decoding
   - Performance metric collection

3. **Sample Data**: French sentences (20-30 words each)
   - Real-world complexity sentences
   - Technology, education, climate, and culture topics

## Performance Optimization

For production deployment, consider:

1. **Model Optimizations**:
   - Use `model.half()` for FP16 inference
   - Apply `torch.compile()` for faster execution
   - Enable KV-cache for sequential generation

2. **Infrastructure**:
   - Use high-end GPUs (A100, H100)
   - Implement model sharding for large models
   - Consider vLLM or TensorRT for deployment

3. **Batching**:
   - Implement dynamic batching for multiple requests
   - Use continuous batching for improved throughput

## Latency Measurements

The system measures:

- **Total Time**: End-to-end translation time
- **Time to First Token**: Latency until first output token
- **Time to Last Token**: Complete generation time
- **Tokenization Time**: Input preprocessing overhead
- **Generation Time**: Pure model inference time

## Files

- `gpt_translator.py`: Production translation system
- `mock_translator.py`: Mock implementation for demonstration
- `test_translator.py`: Test script for validation
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Model Notes

This implementation is designed for the GPT OSS 20B model. If using a different model:

1. Update the model name in the `GPTTranslator` constructor
2. Adjust memory requirements accordingly
3. Validate translation quality for your specific use case

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate tests
- Documentation is updated for significant changes

## License

This project is open source. Please check the license file for details.
