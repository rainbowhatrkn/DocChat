from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "cerebras/Llama3-DocChat-1.0-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


system = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
instruction = "Please give a full and complete answer for the question."

document = """
# Cerebras Wafer-Scale Cluster

Exa-scale performance, single device simplicity

## AI Supercomputers

Condor Galaxy (CG), the supercomputer built by G42 and Cerebras, is the simplest and fastest way to build AI models in the cloud. With over 16 ExaFLOPs of AI compute, Condor Galaxy trains the most demanding models in hours rather than days. The terabyte scale MemoryX system natively accommodates 100 billion+ parameter models, making large scale training simple and efficient.

| Cluster  | ExaFLOPs | Systems  | Memory |
| -------- | -------- | -------- | ------ |
| CG1      | 4        | 64 CS-2s | 82 TB  |
| CG2      | 4        | 64 CS-2s | 82 TB  |
| CG3      | 8        | 64 CS-3s | 108 TB |
"""

question = "how the sun lightning earth?"

user_turn = f"""<context>
{document}
</context>
{instruction} {question}"""

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user_turn}
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
    max_new_tokens=256,
    eos_token_id=terminators,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))