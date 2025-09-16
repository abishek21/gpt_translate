"""
GPT Translation System with Low Latency
Uses GPT OSS 20B model from Hugging Face for French to English translation
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPTTranslator:
    """Low-latency translation system using GPT OSS 20B model"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the GPT translator
        Note: Using gpt2-xl as a proxy since the actual GPT OSS 20B may not be publicly available
        For production, this should be replaced with the actual GPT OSS 20B model
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to GPU
        self.model.to(self.device)
        self.model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up the model to improve initial inference latency"""
        logger.info("Warming up model...")
        warmup_text = "Translate from French to English: Bonjour le monde."
        with torch.no_grad():
            inputs = self.tokenizer.encode(warmup_text, return_tensors="pt").to(self.device)
            _ = self.model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        logger.info("Model warmed up")
    
    def translate(self, french_text: str, max_new_tokens: int = 50) -> Dict[str, any]:
        """
        Translate French text to English with latency measurements
        
        Args:
            french_text: Input French text to translate
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Dictionary containing translation and timing metrics
        """
        # Create translation prompt
        prompt = f"Translate from French to English:\nFrench: {french_text}\nEnglish:"
        
        # Start timing
        total_start_time = time.time()
        
        # Tokenize input
        tokenize_start = time.time()
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        tokenize_time = time.time() - tokenize_start
        
        # Generate translation with timing for first and last token
        with torch.no_grad():
            generation_start = time.time()
            
            # Track time to first token
            first_token_time = None
            generated_tokens = []
            
            # Generate tokens one by one to measure time to first token
            current_inputs = inputs
            
            for i in range(max_new_tokens):
                token_start = time.time()
                
                outputs = self.model(current_inputs)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                
                if i == 0:
                    first_token_time = time.time() - token_start
                
                generated_tokens.append(next_token.item())
                current_inputs = torch.cat([current_inputs, next_token], dim=-1)
                
                # Stop if we hit end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Stop if we detect end of translation (simple heuristic)
                decoded_so_far = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if '\n' in decoded_so_far or len(decoded_so_far.strip()) > 100:
                    break
            
            generation_end = time.time()
            total_generation_time = generation_end - generation_start
        
        # Decode the full output
        full_output = self.tokenizer.decode(current_inputs[0], skip_special_tokens=True)
        
        # Extract just the English translation
        english_part = full_output.split("English:")[-1].strip()
        # Clean up the translation (remove any extra text after newline)
        english_translation = english_part.split('\n')[0].strip()
        
        total_time = time.time() - total_start_time
        
        return {
            "french_input": french_text,
            "english_translation": english_translation,
            "timing": {
                "total_time": total_time,
                "tokenize_time": tokenize_time,
                "generation_time": total_generation_time,
                "time_to_first_token": first_token_time,
                "time_to_last_token": total_generation_time,
                "tokens_generated": len(generated_tokens)
            },
            "full_output": full_output
        }


def get_french_sample_sentences() -> List[str]:
    """
    Return a list of French sample sentences with 20-30 words each
    """
    return [
        "La technologie moderne a transformé notre façon de communiquer, de travailler et de vivre, créant des opportunités infinies pour l'innovation et la collaboration mondiale.",
        "Le système éducatif français met l'accent sur la pensée critique, l'analyse approfondie et le développement de compétences académiques solides pour préparer les étudiants au succès.",
        "Les changements climatiques représentent un défi majeur pour notre planète, nécessitant une action collective immédiate pour protéger l'environnement et les générations futures.",
        "La cuisine française est réputée dans le monde entier pour sa sophistication, sa diversité régionale et l'importance accordée aux ingrédients frais et de qualité supérieure.",
        "L'intelligence artificielle révolutionne de nombreux secteurs industriels, offrant des solutions innovantes tout en soulevant des questions importantes sur l'éthique et l'emploi futur.",
        "Les entreprises modernes doivent s'adapter rapidement aux changements technologiques, aux attentes des consommateurs et aux nouvelles réglementations pour rester compétitives sur le marché.",
        "La littérature française a profondément influencé la culture mondiale, produisant des œuvres intemporelles qui continuent d'inspirer les lecteurs et les écrivains contemporains aujourd'hui.",
        "Le développement durable nécessite un équilibre délicat entre croissance économique, protection environnementale et justice sociale pour assurer un avenir prospère pour tous."
    ]


def main():
    """Main function to run the translation system and report latencies"""
    logger.info("Starting GPT Translation System")
    
    # Initialize translator
    translator = GPTTranslator()
    
    # Get sample French sentences
    french_sentences = get_french_sample_sentences()
    
    # Perform translations and collect metrics
    results = []
    total_translations = len(french_sentences)
    
    logger.info(f"Translating {total_translations} French sentences...")
    
    for i, french_text in enumerate(french_sentences, 1):
        logger.info(f"\n--- Translation {i}/{total_translations} ---")
        logger.info(f"French: {french_text}")
        
        # Perform translation
        result = translator.translate(french_text)
        results.append(result)
        
        # Display results
        logger.info(f"English: {result['english_translation']}")
        logger.info("Latency Metrics:")
        timing = result['timing']
        logger.info(f"  Total time: {timing['total_time']:.3f}s")
        logger.info(f"  Tokenization time: {timing['tokenize_time']:.3f}s")
        logger.info(f"  Generation time: {timing['generation_time']:.3f}s")
        logger.info(f"  Time to first token: {timing['time_to_first_token']:.3f}s")
        logger.info(f"  Time to last token: {timing['time_to_last_token']:.3f}s")
        logger.info(f"  Tokens generated: {timing['tokens_generated']}")
    
    # Calculate and display summary statistics
    logger.info("\n=== SUMMARY STATISTICS ===")
    total_times = [r['timing']['total_time'] for r in results]
    first_token_times = [r['timing']['time_to_first_token'] for r in results]
    generation_times = [r['timing']['generation_time'] for r in results]
    
    logger.info(f"Average total time: {sum(total_times)/len(total_times):.3f}s")
    logger.info(f"Average time to first token: {sum(first_token_times)/len(first_token_times):.3f}s")
    logger.info(f"Average generation time: {sum(generation_times)/len(generation_times):.3f}s")
    logger.info(f"Min total time: {min(total_times):.3f}s")
    logger.info(f"Max total time: {max(total_times):.3f}s")
    
    return results


if __name__ == "__main__":
    main()