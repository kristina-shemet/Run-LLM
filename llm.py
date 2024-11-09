from huggingface_hub import login
import torch
import requests
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


# Authenticate with Hugging Face
huggingface_token = "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"
login(token=huggingface_token)


# Define model ID and load the model with specified dtype and device mapping
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Fetch image from URL
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define chat input with a storytelling prompt
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Imagine this rabbit is an explorer. Tell a short story about its latest adventure: "}
    ]}
]

# Process input with chat template
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

# Generate output with additional hyperparameters for creativity
output = model.generate(
    **inputs,
    max_new_tokens=100,      # Allows a bit longer output for storytelling
    temperature=0.8,         # Adds some creativity to the output
    top_k=50,                # Limits token options for diverse outputs
    top_p=0.9,               # Nucleus sampling for varied word choice
    repetition_penalty=1.2,  # Reduces repetitive phrases
    num_beams=5,             # Uses beam search to refine story quality
    do_sample=True,          # Enables sampling for more diverse outputs
    length_penalty=1.0       # Balances output length preference
)

# Decode and print the generated story
print(processor.decode(output[0]))



