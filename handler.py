#!/usr/bin/env python
# Handler for RunPod Serverless Gemma 3 Image Captioning

import os
import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-4b-it")
CAPTION_PROMPT = os.environ.get("CAPTION_PROMPT", "Provide a short, single-line description of this image for training data.")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
HF_TOKEN = os.environ.get("HF_TOKEN", None)

print(f"Loading model: {MODEL_ID}")
print(f"Default caption prompt: {CAPTION_PROMPT}")
print(f"Default max tokens: {MAX_NEW_TOKENS}")

if HF_TOKEN:
    token_param = {"token": HF_TOKEN}
    print("Using configured Hugging Face token from environment variable")
else:
    token_param = {}
    print("No Hugging Face token provided (this will only work for non-gated models)")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, **token_param)
    print(f"Model loaded on {device}")
except Exception as e:
    if not HF_TOKEN:
        print("ERROR: Failed to load model. This may be a gated model that requires a token.")
    raise e

print("Model and processor loaded and ready for inference")

def caption_image(images_data, prompt=CAPTION_PROMPT, max_new_tokens=MAX_NEW_TOKENS):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {"type": "image", "image": img} for img in images_data
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_tokens = outputs[0][input_len:]
        caption = processor.decode(generated_tokens, skip_special_tokens=True)
        return caption.replace('\n', ' ').strip()

    except Exception as e:
        import traceback
        return f"Error processing image: {str(e)}\n{traceback.format_exc()}"

def handler(job):
    job_input = job["input"]

    if "images" not in job_input:
        return {"error": "No images provided in input"}

    prompt = job_input.get("prompt", CAPTION_PROMPT)
    max_new_tokens = job_input.get("max_new_tokens", MAX_NEW_TOKENS)
    images_input = job_input["images"]
    images_data = []

    try:
        for image in images_input:
            try:
                if isinstance(image, str) and image.startswith("data:image"):
                    base64_data = image.split(",")[1]
                    image_data = Image.open(io.BytesIO(base64.b64decode(base64_data)))
                elif isinstance(image, str) and image.startswith(('http://', 'https://')):
                    import requests
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    image_data = Image.open(io.BytesIO(response.content))
                elif isinstance(image, str) and len(image) > 100:
                    image_data = Image.open(io.BytesIO(base64.b64decode(image)))
                elif isinstance(image, str):
                    image_data = Image.open(image)
                else:
                    return {"error": "Invalid image format. Provide base64, URL, or path."}

                image_data = image_data.convert("RGB")
                images_data.append(image_data)
            except Exception as inner_e:
                return {"error": f"Failed to load image: {str(inner_e)}"}

        caption = caption_image(images_data, prompt, max_new_tokens)
        return {"caption": caption}

    except Exception as e:
        import traceback
        return {"error": f"Error processing image: {str(e)}", "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
