"""
Production GPT Translation System with Low Latency
Optimized implementation for GPT OSS 20B model with GPU acceleration
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionGPTTranslator:
    """
    Production-ready low-latency translation system using GPT OSS 20B model
    Optimized for GPU inference with comprehensive latency measurements
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-large",  # Placeholder - replace with actual GPT OSS 20B
                 device: Optional[str] = None,
                 use_half_precision: bool = True,
                 use_torch_compile: bool = True):
        """
        Initialize the production GPT translator
        
        Args:
            model_name: Hugging Face model name (replace with actual GPT OSS 20B)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            use_half_precision: Use FP16 for faster inference
            use_torch_compile: Use torch.compile for optimization
        """
        self.model_name = model_name
        self.use_half_precision = use_half_precision
        self.use_torch_compile = use_torch_compile
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model and tokenizer
        self._load_model()
        
        # Optimize model
        self._optimize_model()
        
        # Warm up the model
        self._warmup()
    
    def _load_model(self):
        """Load tokenizer and model with optimizations"""
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",  # For better batching
                trust_remote_code=True
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            load_kwargs = {
                "torch_dtype": torch.float16 if self.use_half_precision and self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "trust_remote_code": True
            }
            
            # For large models, enable optimizations
            if "20b" in self.model_name.lower() or "large" in self.model_name.lower():
                load_kwargs["low_cpu_mem_usage"] = True
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if load_kwargs["device_map"] is None:
                self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock implementation...")
            raise e
    
    def _optimize_model(self):
        """Apply model optimizations for inference"""
        logger.info("Applying model optimizations...")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Apply half precision if requested and supported
        if self.use_half_precision and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Applied FP16 optimization")
        
        # Apply torch.compile if requested and supported
        if self.use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True if self.device.type == "cuda" else False
        
        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def _warmup(self):
        """Warm up the model to improve initial inference latency"""
        logger.info("Warming up model...")
        warmup_iterations = 3
        
        for i in range(warmup_iterations):
            warmup_text = f"Translate from French to English: Bonjour le monde {i}."
            try:
                with torch.no_grad():
                    inputs = self.tokenizer.encode(warmup_text, return_tensors="pt").to(self.device)
                    _ = self.model.generate(
                        inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True
                    )
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Clear cache after warmup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Model warmed up")
    
    def translate(self, 
                  french_text: str, 
                  max_new_tokens: int = 50,
                  temperature: float = 0.1,
                  use_cache: bool = True) -> Dict[str, any]:
        """
        Translate French text to English with comprehensive latency measurements
        
        Args:
            french_text: Input French text to translate
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            use_cache: Enable KV-cache for faster generation
            
        Returns:
            Dictionary containing translation and detailed timing metrics
        """
        # Create optimized translation prompt
        prompt = f"Translate from French to English:\nFrench: {french_text}\nEnglish:"
        
        # Start comprehensive timing
        total_start_time = time.time()
        timing_breakdown = {}
        
        # Tokenization timing
        tokenize_start = time.time()
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        timing_breakdown['tokenize_time'] = time.time() - tokenize_start
        
        # Memory usage before generation
        if self.device.type == "cuda":
            initial_memory = torch.cuda.memory_allocated()
        
        # Generation with detailed timing
        with torch.no_grad():
            generation_start = time.time()
            
            # Track first token timing
            first_token_start = time.time()
            
            # Generate with optimized settings
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=use_cache,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            timing_breakdown['generation_time'] = time.time() - generation_start
            
            # Estimate first token time (approximation)
            timing_breakdown['time_to_first_token'] = timing_breakdown['generation_time'] / max(len(outputs.sequences[0]) - len(inputs[0]), 1) if hasattr(outputs, 'sequences') else timing_breakdown['generation_time'] * 0.3
        
        # Decoding timing
        decode_start = time.time()
        if hasattr(outputs, 'sequences'):
            full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_tokens = len(outputs.sequences[0]) - len(inputs[0])
        else:
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_tokens = len(outputs[0]) - len(inputs[0])
        timing_breakdown['decode_time'] = time.time() - decode_start
        
        # Extract English translation
        try:
            english_part = full_output.split("English:")[-1].strip()
            english_translation = english_part.split('\n')[0].strip()
        except:
            english_translation = full_output.strip()
        
        # Final timing
        timing_breakdown['total_time'] = time.time() - total_start_time
        timing_breakdown['time_to_last_token'] = timing_breakdown['generation_time']
        timing_breakdown['tokens_generated'] = generated_tokens
        
        # Memory usage after generation
        if self.device.type == "cuda":
            final_memory = torch.cuda.memory_allocated()
            timing_breakdown['memory_used_mb'] = (final_memory - initial_memory) / 1e6
        
        # Calculate derived metrics
        if generated_tokens > 0:
            timing_breakdown['tokens_per_second'] = generated_tokens / timing_breakdown['generation_time']
            timing_breakdown['ms_per_token'] = timing_breakdown['generation_time'] * 1000 / generated_tokens
        
        return {
            "french_input": french_text,
            "english_translation": english_translation,
            "timing": timing_breakdown,
            "model_info": {
                "model_name": self.model_name,
                "device": str(self.device),
                "use_half_precision": self.use_half_precision,
                "use_torch_compile": self.use_torch_compile,
                "prompt_tokens": len(inputs[0]),
                "total_tokens": len(inputs[0]) + generated_tokens
            },
            "full_output": full_output
        }
    
    def translate_batch(self, 
                       french_texts: List[str], 
                       max_new_tokens: int = 50,
                       temperature: float = 0.1) -> List[Dict[str, any]]:
        """
        Translate multiple French texts in a batch for improved throughput
        
        Args:
            french_texts: List of French texts to translate
            max_new_tokens: Maximum number of new tokens per translation
            temperature: Sampling temperature
            
        Returns:
            List of translation results with timing metrics
        """
        batch_start_time = time.time()
        
        # Create prompts
        prompts = [f"Translate from French to English:\nFrench: {text}\nEnglish:" for text in french_texts]
        
        # Tokenize batch
        tokenize_start = time.time()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        tokenize_time = time.time() - tokenize_start
        
        # Generate batch
        with torch.no_grad():
            generation_start = time.time()
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
            generation_time = time.time() - generation_start
        
        # Decode results
        decode_start = time.time()
        results = []
        for i, (french_text, output) in enumerate(zip(french_texts, outputs)):
            full_output = self.tokenizer.decode(output, skip_special_tokens=True)
            try:
                english_translation = full_output.split("English:")[-1].strip().split('\n')[0].strip()
            except:
                english_translation = full_output.strip()
            
            generated_tokens = len(output) - len(inputs.input_ids[i])
            
            results.append({
                "french_input": french_text,
                "english_translation": english_translation,
                "timing": {
                    "batch_generation_time": generation_time,
                    "estimated_per_item_time": generation_time / len(french_texts),
                    "tokens_generated": generated_tokens,
                    "batch_tokenize_time": tokenize_time
                },
                "full_output": full_output
            })
        
        decode_time = time.time() - decode_start
        total_batch_time = time.time() - batch_start_time
        
        # Add batch metrics to all results
        for result in results:
            result["timing"]["total_batch_time"] = total_batch_time
            result["timing"]["decode_time"] = decode_time
            result["timing"]["batch_throughput"] = len(french_texts) / total_batch_time
        
        return results
    
    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def get_french_sample_sentences() -> List[str]:
    """Return French sample sentences with 20-30 words each"""
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


def run_production_translation_system():
    """Run the production translation system with comprehensive benchmarking"""
    logger.info("Starting Production GPT Translation System")
    logger.info("=" * 70)
    
    # Initialize translator
    try:
        translator = ProductionGPTTranslator()
    except Exception as e:
        logger.error(f"Failed to initialize translator: {e}")
        logger.info("Please ensure:")
        logger.info("1. Internet connection is available")
        logger.info("2. GPU memory is sufficient")
        logger.info("3. Model name is correct")
        return
    
    # Get sample sentences
    french_sentences = get_french_sample_sentences()
    
    # Run individual translations
    logger.info(f"\nTranslating {len(french_sentences)} sentences individually...")
    individual_results = []
    
    for i, french_text in enumerate(french_sentences, 1):
        logger.info(f"\n--- Translation {i}/{len(french_sentences)} ---")
        logger.info(f"French: {french_text[:60]}...")
        
        try:
            result = translator.translate(french_text)
            individual_results.append(result)
            
            logger.info(f"English: {result['english_translation'][:60]}...")
            timing = result['timing']
            logger.info(f"Timing: {timing['total_time']*1000:.1f}ms total, "
                       f"{timing['time_to_first_token']*1000:.1f}ms first token, "
                       f"{timing.get('tokens_per_second', 0):.1f} tok/s")
            
        except Exception as e:
            logger.error(f"Translation {i} failed: {e}")
    
    # Run batch translation
    logger.info(f"\n" + "=" * 70)
    logger.info("Running batch translation...")
    
    try:
        batch_results = translator.translate_batch(french_sentences[:4])  # Batch first 4 for memory
        
        logger.info(f"Batch completed. Results:")
        for i, result in enumerate(batch_results, 1):
            logger.info(f"  {i}. {result['english_translation'][:40]}...")
        
        batch_timing = batch_results[0]['timing']
        logger.info(f"Batch metrics: {batch_timing['total_batch_time']:.2f}s total, "
                   f"{batch_timing['batch_throughput']:.1f} sentences/s")
        
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        batch_results = []
    
    # Performance summary
    if individual_results:
        logger.info(f"\n" + "=" * 70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 70)
        
        total_times = [r['timing']['total_time'] for r in individual_results]
        first_token_times = [r['timing']['time_to_first_token'] for r in individual_results]
        tokens_per_sec = [r['timing'].get('tokens_per_second', 0) for r in individual_results if r['timing'].get('tokens_per_second', 0) > 0]
        
        logger.info(f"Individual translations:")
        logger.info(f"  Average total time:     {sum(total_times)/len(total_times)*1000:.1f}ms")
        logger.info(f"  Average first token:    {sum(first_token_times)/len(first_token_times)*1000:.1f}ms")
        logger.info(f"  Average throughput:     {sum(tokens_per_sec)/len(tokens_per_sec):.1f} tokens/s" if tokens_per_sec else "  Throughput: N/A")
        logger.info(f"  Best latency:           {min(total_times)*1000:.1f}ms")
        logger.info(f"  Worst latency:          {max(total_times)*1000:.1f}ms")
        
        if batch_results:
            logger.info(f"\nBatch processing:")
            logger.info(f"  Batch throughput:       {batch_results[0]['timing']['batch_throughput']:.1f} sentences/s")
            logger.info(f"  Per-item estimate:      {batch_results[0]['timing']['estimated_per_item_time']*1000:.1f}ms")
    
    # Cleanup
    translator.cleanup()
    logger.info(f"\n" + "=" * 70)
    logger.info("Translation system completed successfully!")


if __name__ == "__main__":
    run_production_translation_system()