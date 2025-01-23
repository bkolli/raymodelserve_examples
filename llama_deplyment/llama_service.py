# llama_service.py

import os
import time
from typing import List

import ray
from ray import serve
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# ------------------- Configuration -------------------

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
USE_HF_AUTH = os.environ.get("USE_HF_AUTH", "false").lower() == "true"
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", "")

# Ray Serve configuration
NUM_REPLICAS = int(os.environ.get("NUM_REPLICAS", 1))
RAY_NAMESPACE = os.environ.get("RAY_NAMESPACE", "default")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 10))
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# ------------------- Deployment -------------------


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
    max_concurrent_queries=MAX_CONCURRENCY
)
class LlamaDeployment:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"Loading model: {MODEL_NAME}")
        start_time = time.time()

        try:
            if USE_HF_AUTH:
                print("Using Hugging Face authentication token.")
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_AUTH_TOKEN, device_map="auto", torch_dtype=torch.float16)
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_AUTH_TOKEN)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds.")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> str:
        """Generates text based on the given prompt."""
        try:
            start_time = time.time()
            sequences = self.generator(
                prompt,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generation_time = time.time() - start_time
            generated_text = sequences[0]['generated_text']
            cleaned_text = generated_text.replace(prompt, "").strip()
            print(f"Generated text in {generation_time:.2f} seconds.")
            return cleaned_text

        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error: {e}"

    async def __call__(self, http_request) -> str:
        try:
            data = await http_request.json()
            prompt = data["prompt"]
            max_new_tokens = data.get("max_new_tokens", 128)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 50)

        except:
            prompt = await http_request.body()
            prompt = prompt.decode()
            max_new_tokens = 128
            temperature = 0.7
            top_p = 0.9
            top_k = 50
        return self.generate(prompt, max_new_tokens, temperature, top_p, top_k)


deployment = LlamaDeployment.bind()
