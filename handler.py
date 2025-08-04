#!/usr/bin/env python
# Handler for RunPod Serverless Gemma 3 Image Captioning

import os
import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# ===== USER MODIFIABLE SETTINGS =====
# Get model ID from environment variable with fallback to default
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-4b-it")

# Prompt for image captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = os.environ.get("CAPTION_PROMPT", 
                               "Provide a short, single-line description of this image for training data.")

# Maximum tokens to generate with fallback to default
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
# =====================================

# Set up Hugging Face token from environment variable 
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load the model once at startup, outside of the handler
print(f"Loading model: {MODEL_ID}")
print(f"Default caption prompt: {CAPTION_PROMPT}")
print(f"Default max tokens: {MAX_NEW_TOKENS}")

# Configure token parameters if provided
if HF_TOKEN:
    token_param = {"token": HF_TOKEN}
    print("Using configured Hugging Face token from environment variable")
else:
    token_param = {}
    print("No Hugging Face token provided (this will only work for non-gated models)")

# Configure quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model
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
    """Generate a caption for the given image."""
    try:
        #############################################################################
        ### Änderung hier: ##########################################################
        ### Übergibt mehrere Bilder statt nur eines an das Modell ###################
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

        # Process inputs
        processor.tokenizer.padding_side = "left"

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="longest",
            ############ Pad die Sequenzlängen zusätzlich auf ein Vielfaches von 8 # GPU-freundliche Ausrichtung, 
            ############ hat zuvor Fehler verursacht, daher aktiviert
            pad_to_multiple_of=8
        )

        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Track input length to extract only new tokens
        input_len = inputs["input_ids"].shape[-1]

        # Generate caption
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # Extract only the newly generated tokens
        generated_tokens = outputs[0][input_len:]

        # Decode the caption
        caption = processor.decode(generated_tokens, skip_special_tokens=True)

        # Ensure caption is a single line
        caption = caption.replace('\n', ' ').strip()
        return caption

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error processing image: {str(e)}\n{traceback_str}"

def handler(job):
    job_input = job["input"]

    # Basic input validation
    if "images" not in job_input:
        return {"error": "No images provided in input"}

    # Get the prompt (optional, use default if not provided)
    prompt = job_input.get("prompt", CAPTION_PROMPT)
    max_new_tokens = job_input.get("max_new_tokens", MAX_NEW_TOKENS)

    # Handle the image (base64, URL, or file path)
    images_input = job_input["images"]
    images_data = []

    try:
        for image in images_input:
            try:
                # Case 1: Base64 encoded image
                if isinstance(image, str) and image.startswith("data:image"):
                    # Extract base64 part after the comma
                    base64_data = image.split(",")[1]
                    image_data = Image.open(io.BytesIO(base64.b64decode(base64_data)))
                
                # Case 2: URL
                elif isinstance(image, str) and image.startswith(('http://', 'https://')):
                    import requests
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    image_data = Image.open(io.BytesIO(response.content))

                # Case 3: Pure base64 string
                elif isinstance(image, str) and len(image) > 100:
                    image_data = Image.open(io.BytesIO(base64.b64decode(image)))
                
                # Case 4: Local file path
                elif isinstance(image, str):
                    image_data = Image.open(image)
                
                else:
                    return {"error": "Invalid image format. Provide base64, URL, or path."}

                # Convert to RGB mode to ensure compatibility
                image_data = image_data.convert("RGB")

                # Collect image
                images_data.append(image_data)
            except Exception as inner_e:
                return {"error": f"Failed to load image: {str(inner_e)}"}

        # Process all images together to get one joint caption
        caption = caption_image(images_data, prompt, max_new_tokens)

        # Return the result
        return {"caption": caption}

    except Exception as e:
        import traceback
        return {"error": f"Error processing image: {str(e)}", "traceback": traceback.format_exc()}

# Start the serverless function
runpod.serverless.start({"handler": handler})
