from mlx_lm import load, generate

model, tokenizer = load("PleIAs/Baguettotron")

prompt = "Write a story about Einstein"

chat_template = """
{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}
"""

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, chat_template=chat_template
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)