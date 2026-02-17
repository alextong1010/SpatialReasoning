import torch

GENERATION_CONFIGS = {
    "qwen3vl_8b_instruct": dict(
        max_new_tokens=256,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        repetition_penalty=1.0,
    ),
    "qwen3vl_8b_thinking": dict(
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.0,
    ),
}


def generate(model, processor, image, question, model_key=None):
    """Generate a response from a Qwen3-VL model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generation_config = GENERATION_CONFIGS.get(
        model_key, dict(max_new_tokens=256, do_sample=False)
    )
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response
