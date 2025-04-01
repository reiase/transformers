import torch

from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "Qwen/Qwen1.5-MoE-A2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

messages = [
    { "role":"user", "content": "轻轻的我走了，正如我轻轻的来；我轻轻的招手，作别西天的云彩。"},
]

input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
