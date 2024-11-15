from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Authenticate with Hugging Face
huggingface_token = "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"
login(token=huggingface_token)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an AI assistant with expertise in natural language processing and AI model configuration."},
    {"role": "user", "content": "What would happen to the responses if we set the temperature to a low value like 0.3 versus a high value like 1.0? Could you explain how it would change the style and predictability of the model's answers?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
