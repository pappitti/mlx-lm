import json
import asyncio
import time
from typing import List, Dict, Union, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Rich Imports for TUI ---
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

# --- MLX Imports ---
# Ensure mlx_generate.py is in the same folder
from mlx_lm.generate import (
    load, 
    BatchGenerator, 
    make_sampler, 
    wired_limit, 
    generation_stream
)
import mlx.core as mx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_TEMPLATE = {
    "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-q8": """"
{%- set default_system_message = "You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYour knowledge base was last updated on 2023-10-01.\n\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. 'What are some good restaurants around me?' => 'Where are you?' or 'When is the next flight to Tokyo' => 'Where do you travel from?')" %}

{{- bos_token }}

{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = default_system_message %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}

{%- for message in loop_messages %}
    {%- if message['role'] == 'user' %}
	    {%- if message['content'] is string %}
            {{- '[INST]' + message['content'] + '[/INST]' }}
	    {%- else %}
		    {{- '[INST]' }}
		    {%- for block in message['content'] %}
			    {%- if block['type'] == 'text' %}
				    {{- block['text'] }}
			    {%- elif block['type'] == 'image' or block['type'] == 'image_url' %}
				    {{- '[IMG]' }}
				{%- else %}
				    {{- raise_exception('Only text and image blocks are supported in message content!') }}
				{%- endif %}
			{%- endfor %}
		    {{- '[/INST]' }}
		{%- endif %}
    {%- elif message['role'] == 'system' %}
        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- message['content'] + eos_token }}
    {%- else %}
        {{- raise_exception('Only user, system and assistant roles are supported!') }}
    {%- endif %}
{%- endfor %}
""",
"mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit" : """"
{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""",
"PleIAs/Baguettotron": """
{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}
"""
}

# Configuration
MODEL_PATH = "PleIAs/Baguettotron"
model = None
tokenizer = None

# Updated Pydantic model to accept Strings OR Chat Lists
class BatchRequest(BaseModel):
    model : str
    prompts: Union[List[str], List[List[Dict[str, Any]]]]
    max_tokens: int = 200
    temp: float = 0.2
    top_p: float = 1.0

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    print(f"Loading model: {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded. Ready.")

def create_dashboard(prompts_display: List[str], generated_text: Dict[int, str], finished_status: Dict[int, bool]) -> Table:
    """
    Creates a Rich Table where each cell is a Panel containing the 
    prompt and the streaming generated text.
    """
    table = Table(show_header=False, show_edge=False, box=None, expand=True)
    table.add_column("A", ratio=1)
    table.add_column("B", ratio=1)

    panels = []
    for uid, prompt_preview in enumerate(prompts_display):
        is_done = finished_status.get(uid, False)
        
        # Style: Green border while generating, White when done
        border_style = "green" if not is_done else "white"
        title_text = f"[b]ID {uid}[/b] | {prompt_preview}"
        if is_done:
            title_text += " [DONE]"

        content = generated_text.get(uid, "")
        # Truncate display content if it gets too long for the dashboard
        display_content = content[-1000:] if len(content) > 1000 else content

        p = Panel(
            display_content,
            title=title_text,
            border_style=border_style,
            height=24,
            padding=(0, 1)
        )
        panels.append(p)

    # Add panels to table grid
    for i in range(0, len(panels), 2):
        if i + 1 < len(panels):
            table.add_row(panels[i], panels[i+1])
        else:
            table.add_row(panels[i], "") 

    return table

@app.post("/generate_batch")
async def generate_batch(req: BatchRequest):
    
    # 1. Pre-process Prompts (Handle Chat Templates vs Raw Strings)
    processed_prompts = []
    display_prompts = [] # Short versions for the UI header

    if req.model in CHAT_TEMPLATE:
        chat_template = CHAT_TEMPLATE[req.model]

    for p in req.prompts:
        if isinstance(p, list): 
            # It is a chat history (List[Dict]) -> Apply Chat Template
            if req.model in CHAT_TEMPLATE:
                text_prompt = tokenizer.apply_chat_template(
                    p, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    chat_template=chat_template
                )
                processed_prompts.append(text_prompt)
                # Use the last user message for display
                last_user = next((m['content'] for m in reversed(p) if m['role'] == 'user'), "Chat")
                display_prompts.append(last_user[:30] + "...")
            else:
                # Fallback if no template
                processed_prompts.append(str(p))
                display_prompts.append("Raw Chat...")
        else:
            # It is a raw string
            processed_prompts.append(p)
            display_prompts.append(p[:30].replace("\n", " ") + "...")

    # 2. Tokenize
    prompt_tokens = [tokenizer.encode(p) for p in processed_prompts]
    
    # 3. Initialize Generator
    sampler = make_sampler(req.temp, req.top_p)
    gen = BatchGenerator(
        model=model,
        max_tokens=req.max_tokens,
        stop_tokens=tokenizer.eos_token_ids,
        sampler=sampler
    )
    
    # Insert into batcher
    uids = gen.insert(prompt_tokens, max_tokens=req.max_tokens)

    # 4. State Tracking
    generated_text = {uid: "" for uid in uids}
    finished_status = {uid: False for uid in uids}

    # 5. Run Generation Loop with Rich Visualization
    # Note: We are NOT returning a stream. We wait for completion.
    with wired_limit(model, [generation_stream]):
        with Live(create_dashboard(display_prompts, generated_text, finished_status), refresh_per_second=12, screen=False) as live:
            while True:
                # Blocking MLX step
                responses = gen.next()
                
                if not responses:
                    break
                
                for r in responses:
                    token_str = tokenizer.decode([r.token])
                    generated_text[r.uid] += token_str
                    
                    if r.finish_reason:
                        finished_status[r.uid] = True

                # Update Terminal
                live.update(create_dashboard(display_prompts, generated_text, finished_status))
                
                # Yield to event loop slightly to keep server responsive
                await asyncio.sleep(0)

    # 6. Format Response for Client
    # The client expects {"texts": ["...", "..."]}
    # Ensure results are ordered by original request index (uids are 0,1,2...)
    ordered_results = [generated_text[i] for i in range(len(processed_prompts))]
    
    return JSONResponse(content={"texts": ordered_results})

if __name__ == "__main__":
    import uvicorn
    # Important: access_log=False prevents HTTP logs from breaking the visual dashboard
    uvicorn.run(app, host="0.0.0.0", port=8081, access_log=False)