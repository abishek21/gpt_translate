#!/usr/bin/env python3
"""
GPT Translation CLI
Command-line interface for the GPT translation system
"""

import argparse
import sys
import os
from typing import List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mock_translator import MockGPTTranslator, get_french_sample_sentences


def translate_text(text: str, model_type: str = "mock") -> dict:
    """Translate a single text"""
    if model_type == "mock":
        translator = MockGPTTranslator()
        return translator.translate(text)
    else:
        # For production, would use ProductionGPTTranslator
        try:
            from production_translator import ProductionGPTTranslator
            translator = ProductionGPTTranslator()
            return translator.translate(text)
        except Exception as e:
            print(f"Production model failed: {e}")
            print("Falling back to mock implementation...")
            translator = MockGPTTranslator()
            return translator.translate(text)


def translate_file(filename: str, model_type: str = "mock") -> List[dict]:
    """Translate texts from a file (one per line)"""
    results = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if model_type == "mock":
            translator = MockGPTTranslator()
        else:
            try:
                from production_translator import ProductionGPTTranslator
                translator = ProductionGPTTranslator()
            except Exception as e:
                print(f"Production model failed: {e}")
                print("Falling back to mock implementation...")
                translator = MockGPTTranslator()
        
        for i, line in enumerate(lines, 1):
            print(f"Translating line {i}/{len(lines)}...")
            result = translator.translate(line)
            results.append(result)
            print(f"  French: {line[:50]}...")
            print(f"  English: {result['english_translation'][:50]}...")
            print(f"  Time: {result['timing']['total_time']*1000:.1f}ms")
            print()
        
        return results
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def run_demo(model_type: str = "mock"):
    """Run the demo with sample sentences"""
    print("Running GPT Translation Demo")
    print("=" * 50)
    
    sentences = get_french_sample_sentences()
    
    if model_type == "mock":
        translator = MockGPTTranslator()
    else:
        try:
            from production_translator import ProductionGPTTranslator
            translator = ProductionGPTTranslator()
        except Exception as e:
            print(f"Production model failed: {e}")
            print("Falling back to mock implementation...")
            translator = MockGPTTranslator()
    
    results = []
    for i, sentence in enumerate(sentences, 1):
        print(f"\n--- Demo Translation {i}/{len(sentences)} ---")
        result = translator.translate(sentence)
        results.append(result)
        
        print(f"French: {sentence}")
        print(f"English: {result['english_translation']}")
        print(f"Latency: {result['timing']['total_time']*1000:.1f}ms")
    
    # Summary
    total_times = [r['timing']['total_time'] for r in results]
    print(f"\n--- Summary ---")
    print(f"Average latency: {sum(total_times)/len(total_times)*1000:.1f}ms")
    print(f"Best latency: {min(total_times)*1000:.1f}ms")
    print(f"Worst latency: {max(total_times)*1000:.1f}ms")


def benchmark_mode(iterations: int = 10, model_type: str = "mock"):
    """Run benchmark with multiple iterations"""
    print(f"Running Benchmark Mode ({iterations} iterations)")
    print("=" * 50)
    
    test_sentence = "La technologie moderne transforme notre façon de communiquer et de travailler ensemble."
    
    if model_type == "mock":
        translator = MockGPTTranslator()
    else:
        try:
            from production_translator import ProductionGPTTranslator
            translator = ProductionGPTTranslator()
        except Exception as e:
            print(f"Production model failed: {e}")
            print("Falling back to mock implementation...")
            translator = MockGPTTranslator()
    
    results = []
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...", end=" ")
        result = translator.translate(test_sentence)
        results.append(result)
        print(f"{result['timing']['total_time']*1000:.1f}ms")
    
    # Benchmark analysis
    total_times = [r['timing']['total_time'] for r in results]
    first_token_times = [r['timing']['time_to_first_token'] for r in results]
    
    print(f"\n--- Benchmark Results ---")
    print(f"Total time - Avg: {sum(total_times)/len(total_times)*1000:.1f}ms, "
          f"Min: {min(total_times)*1000:.1f}ms, "
          f"Max: {max(total_times)*1000:.1f}ms")
    print(f"First token - Avg: {sum(first_token_times)/len(first_token_times)*1000:.1f}ms, "
          f"Min: {min(first_token_times)*1000:.1f}ms, "
          f"Max: {max(first_token_times)*1000:.1f}ms")
    
    # Calculate percentiles
    total_times_sorted = sorted(total_times)
    p50 = total_times_sorted[len(total_times_sorted)//2] * 1000
    p95 = total_times_sorted[int(len(total_times_sorted)*0.95)] * 1000
    p99 = total_times_sorted[int(len(total_times_sorted)*0.99)] * 1000
    
    print(f"Latency percentiles - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="GPT Translation System CLI")
    parser.add_argument("--model", choices=["mock", "production"], default="mock",
                       help="Model type to use (default: mock)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate text")
    translate_parser.add_argument("text", help="French text to translate")
    
    # File command
    file_parser = subparsers.add_parser("file", help="Translate from file")
    file_parser.add_argument("filename", help="File containing French texts (one per line)")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample sentences")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    benchmark_parser.add_argument("--iterations", type=int, default=10,
                                 help="Number of benchmark iterations (default: 10)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "translate":
            result = translate_text(args.text, args.model)
            print(f"French: {args.text}")
            print(f"English: {result['english_translation']}")
            print(f"Latency: {result['timing']['total_time']*1000:.1f}ms")
            print(f"Time to first token: {result['timing']['time_to_first_token']*1000:.1f}ms")
            
        elif args.command == "file":
            results = translate_file(args.filename, args.model)
            if results:
                print(f"\nTranslated {len(results)} sentences successfully!")
                avg_time = sum(r['timing']['total_time'] for r in results) / len(results)
                print(f"Average latency: {avg_time*1000:.1f}ms")
            
        elif args.command == "demo":
            run_demo(args.model)
            
        elif args.command == "benchmark":
            benchmark_mode(args.iterations, args.model)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()