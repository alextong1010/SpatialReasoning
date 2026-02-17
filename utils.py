import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration, AutoModel, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO

def load_model_and_helpers(model_name):
    """Load model and processor/tokenizer.
    
    Args:
        model_name: Model name
    """
    if model_name == "llava1_5_7b_hf":
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    elif model_name == "qwen3vl_8b_thinking":
        model_id = "Qwen/Qwen3-VL-8B-Thinking"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, 
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    elif model_name == "qwen3vl_8b_instruct":
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, 
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    elif model_name == "internvl3_5_2b":
        model_id = "OpenGVLab/InternVL3_5-2B"
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        return model, tokenizer
    elif model_name == "internvl3_5_8b":
        model_id = "OpenGVLab/InternVL3_5-8B"
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        return model, tokenizer
    else:
        raise NotImplementedError(f"Model type for '{model_name}' not implemented yet")

def load_image(image_input):
    """Load image from PIL Image, path, or URL"""
    # If already a PIL Image, just ensure it's RGB
    if isinstance(image_input, Image.Image):
        return image_input.convert('RGB')
    # Handle URL
    elif image_input.startswith('http://') or image_input.startswith('https://'):
        response = requests.get(image_input)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    # Handle file path
    else:
        image = Image.open(image_input).convert('RGB')
    return image