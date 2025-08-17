from transformers import pipeline
from dotenv import load_dotenv
import os
import torch

load_dotenv()

HF_MODEL_PATH = os.getenv("HF_MODEL_PATH")

try:
    image_text_to_text_generator = pipeline(
        "image-text-to-text",
        model=HF_MODEL_PATH,
        device="cpu",
        torch_dtype=torch.bfloat16,
    )

    print("Model loaded successfully.")

    messages = [
        {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
        "role": "user",
        "content": [
            {"type": "image", "image": os.getenv("PHOTO_PATH")},
            {"type": "text", "text": "What animal is on the candy?"}
          ]
        }
    ]

    output = image_text_to_text_generator(text=messages, max_new_tokens=200)
    print(output[0]["generated_text"][-1]["content"])

except Exception as e:
    print(f"Error loading model: {e}")
