"""
Complete Example: GPT Translation System
Demonstrates all features of the low-latency translation system
"""

import time
import logging
from mock_translator import MockGPTTranslator, get_french_sample_sentences, analyze_latency_performance

def main():
    print("=" * 80)
    print("GPT TRANSLATION SYSTEM - COMPLETE EXAMPLE")
    print("=" * 80)
    
    print("\n1. SYSTEM INITIALIZATION")
    print("-" * 40)
    
    # Initialize the translator
    print("Initializing GPT Translation System...")
    translator = MockGPTTranslator("gpt-oss-20b")
    
    print("\n2. SINGLE TRANSLATION EXAMPLE")
    print("-" * 40)
    
    # Example single translation
    sample_text = "L'intelligence artificielle révolutionne l'industrie moderne avec des innovations remarquables et des possibilités infinies."
    print(f"Input: {sample_text}")
    
    result = translator.translate(sample_text)
    print(f"Output: {result['english_translation']}")
    print(f"Latency: {result['timing']['total_time']*1000:.1f}ms (First token: {result['timing']['time_to_first_token']*1000:.1f}ms)")
    
    print("\n3. BATCH TRANSLATION EXAMPLE")
    print("-" * 40)
    
    # Get sample sentences
    french_sentences = get_french_sample_sentences()
    print(f"Translating {len(french_sentences)} sample sentences...")
    
    # Translate all sentences
    all_results = []
    start_time = time.time()
    
    for i, sentence in enumerate(french_sentences, 1):
        print(f"  [{i}/{len(french_sentences)}] Translating...", end=" ")
        result = translator.translate(sentence)
        all_results.append(result)
        print(f"{result['timing']['total_time']*1000:.0f}ms")
    
    total_batch_time = time.time() - start_time
    
    print("\n4. PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Analyze performance
    perf_stats = analyze_latency_performance(all_results)
    
    print(f"Batch processing completed in {total_batch_time:.2f} seconds")
    print(f"Individual sentence metrics:")
    print(f"  • Average total time:     {perf_stats['avg_total_time']*1000:.1f}ms")
    print(f"  • Average first token:    {perf_stats['avg_first_token_time']*1000:.1f}ms") 
    print(f"  • Best latency:           {perf_stats['min_total_time']*1000:.1f}ms")
    print(f"  • Worst latency:          {perf_stats['max_total_time']*1000:.1f}ms")
    print(f"  • Throughput:             {perf_stats['throughput_sentences_per_second']:.1f} sentences/sec")
    
    print("\n5. DETAILED SAMPLE TRANSLATIONS")
    print("-" * 40)
    
    # Show detailed results for first 3 translations
    for i, result in enumerate(all_results[:3], 1):
        print(f"\nTranslation {i}:")
        print(f"  French:  {result['french_input']}")
        print(f"  English: {result['english_translation']}")
        timing = result['timing']
        print(f"  Timing:  {timing['total_time']*1000:.1f}ms total | "
              f"{timing['time_to_first_token']*1000:.1f}ms first token | "
              f"{timing['tokens_generated']} tokens")
    
    print("\n6. LATENCY DISTRIBUTION")
    print("-" * 40)
    
    # Latency distribution analysis
    total_times = [r['timing']['total_time']*1000 for r in all_results]
    total_times.sort()
    
    percentiles = [50, 75, 90, 95, 99]
    print("Latency percentiles:")
    for p in percentiles:
        idx = int(len(total_times) * p / 100)
        if idx >= len(total_times):
            idx = len(total_times) - 1
        print(f"  P{p:2d}: {total_times[idx]:6.1f}ms")
    
    print("\n7. MEMORY AND EFFICIENCY")
    print("-" * 40)
    
    # Model information
    model_info = all_results[0]['model_info']
    print(f"Model: {model_info['model_name']}")
    print(f"Device: {model_info['device']}")
    print(f"Note: {model_info['note']}")
    
    print("\n8. PRODUCTION RECOMMENDATIONS")
    print("-" * 40)
    
    print("For production deployment with actual GPT OSS 20B:")
    print("  • Use GPU with ≥40GB VRAM (A100/H100)")
    print("  • Enable FP16 precision for 2x speedup") 
    print("  • Apply torch.compile() for additional optimization")
    print("  • Implement request batching for higher throughput")
    print("  • Consider vLLM or TensorRT for serving")
    print("  • Monitor GPU memory usage and implement scaling")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("Replace MockGPTTranslator with ProductionGPTTranslator for real usage.")
    print("=" * 80)


if __name__ == "__main__":
    # Configure logging to be less verbose for the example
    logging.getLogger("mock_translator").setLevel(logging.WARNING)
    
    main()