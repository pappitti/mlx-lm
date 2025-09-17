from mlx_lm.batch_server import ModelProvider, run

MODEL = "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-q8"

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
"lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-8bit" : """"
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""
}

class ServerArgs():
    def __init__(self):
        self.model = MODEL
        self.adapter_path = None
        self.draft_model = None
        self.num_draft_tokens = 0
        self.host = "127.0.0.1"
        self.port = 8080
        self.trust_remote_code = False
        self.log_level = "INFO"
        self.chat_template = CHAT_TEMPLATE.get(MODEL, CHAT_TEMPLATE["mlx-community/Mistral-Small-3.2-24B-Instruct-2506-q8"])
        self.use_default_chat_template = False
        self.temp = 0.15
        self.top_p = 1.0
        self.top_k = 0
        self.min_p = 0.8
        self.max_tokens = 512
        self.chat_template_args = {}

args = ServerArgs()

def main():
    print(f"Hello from mlx-explore! Using model: {args.model}")
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    main()
