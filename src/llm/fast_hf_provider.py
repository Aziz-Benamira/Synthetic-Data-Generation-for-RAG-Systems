"""
Fast HuggingFace Provider using transformers + flash-attention-2
Optimized for the pre-installed models in /home/ensta/data/
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FastHFProvider:
    """
    Fast HuggingFace LLM provider using optimizations:
    - Flash Attention 2
    - BFloat16 precision
    - Batched inference
    - KV cache
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 4096,
        use_flash_attention: bool = True
    ):
        """
        Initialize fast HF provider
        
        Args:
            model_path: Path to model (e.g., /home/ensta/data/Meta-Llama-3-8B-Instruct)
            device: Device to use (cuda/cpu)
            max_length: Maximum context length
            use_flash_attention: Use Flash Attention 2 for speedup
        """
        self.model_path = Path(model_path).expanduser()
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Initializing FastHFProvider")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Flash Attention: {use_flash_attention}")
        
        # Import heavy libraries only when needed
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig
            )
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install transformers torch accelerate")
            raise
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Add flash attention if requested
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("  Using Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
        
        # Load model
        logger.info("Loading model (this may take a minute)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **model_kwargs
        )
        self.model.eval()
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"  Parameters: ~{sum(p.numel() for p in self.model.parameters())/1e9:.1f}B")
        
        self.torch = torch
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate completion for messages
        
        Args:
            messages: List of {role: str, content: str}
            config: Generation config (temperature, max_tokens, etc.)
        
        Returns:
            Generated text
        """
        config = config or {}
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generation config
        gen_config = {
            "max_new_tokens": config.get("max_tokens", 1024),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "do_sample": config.get("temperature", 0.7) > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode (skip prompt)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'torch'):
                self.torch.cuda.empty_cache()
                logger.info("Model freed from memory")
        except:
            pass


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with Llama-3-8B
    provider = FastHFProvider(
        model_path="/home/ensta/data/Meta-Llama-3-8B-Instruct"
    )
    
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    response = provider.generate(messages)
    print(f"\nResponse: {response}")
