import torch
from PIL import Image


def expand2square(pil_img, background_color):
    """Expand image to square by padding with background color.

    LLaVA 1.5 pads non-square images to square using the CLIP image mean
    color before the vision encoder processes them.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_bg_color(processor):
    """Get the background color for padding from the processor's image mean.

    Returns a tuple of (R, G, B) integers in 0-255 range.
    """
    if hasattr(processor, "image_processor"):
        mean = processor.image_processor.image_mean
    else:
        mean = [0.48145466, 0.4578275, 0.40821073]  # Default CLIP mean
    return tuple(int(x * 255) for x in mean)


def preprocess_image(image, processor):
    """Pad image to square using the processor's mean color."""
    bg_color = get_bg_color(processor)
    return expand2square(image, bg_color)


def generate(model, processor, image, question, model_key=None):
    """Generate a response from a LLaVA model.

    Preprocessing pipeline (follows VLM-Visualizer):
    1. Pad image to square using CLIP mean color
    2. Build conversation using HuggingFace chat template
    3. Process inputs and generate with float16 precision
    """
    # Pad image to square with mean color (matches LLaVA 1.5 preprocessing)
    image = preprocess_image(image, processor)

    # Build conversation in HuggingFace LLaVA format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs with proper dtype
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        model.device, torch.float16
    )

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    # Decode only the generated tokens (skip the input)
    response = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    return response
