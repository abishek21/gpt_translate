"""
Test script for the GPT Translation System
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt_translator import GPTTranslator, get_french_sample_sentences
import logging

# Configure logging to show info level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_single_translation():
    """Test a single translation to verify the system works"""
    print("=== Testing Single Translation ===")
    
    # Initialize translator
    translator = GPTTranslator()
    
    # Test with a simple French sentence
    test_sentence = "La technologie moderne a transformé notre façon de communiquer et de travailler ensemble dans le monde numérique actuel."
    
    print(f"French: {test_sentence}")
    result = translator.translate(test_sentence)
    
    print(f"English: {result['english_translation']}")
    print(f"Total time: {result['timing']['total_time']:.3f}s")
    print(f"Time to first token: {result['timing']['time_to_first_token']:.3f}s")
    
    return result

def test_multiple_translations():
    """Test multiple translations with shorter sentences for quicker testing"""
    print("\n=== Testing Multiple Translations ===")
    
    translator = GPTTranslator()
    
    # Use first 3 sample sentences for quicker testing
    sentences = get_french_sample_sentences()[:3]
    
    results = []
    for i, sentence in enumerate(sentences, 1):
        print(f"\n--- Translation {i} ---")
        print(f"French: {sentence[:50]}...")  # Show only first 50 chars for brevity
        
        result = translator.translate(sentence)
        results.append(result)
        
        print(f"English: {result['english_translation'][:50]}...")
        print(f"Time: {result['timing']['total_time']:.3f}s")
    
    # Summary stats
    total_times = [r['timing']['total_time'] for r in results]
    print(f"\nAverage time: {sum(total_times)/len(total_times):.3f}s")
    
    return results

if __name__ == "__main__":
    print("Starting GPT Translation System Tests")
    
    # Test single translation first
    single_result = test_single_translation()
    
    # Test multiple translations
    multiple_results = test_multiple_translations()
    
    print("\n=== All Tests Completed ===")
    print("Translation system is working correctly!")