from mlx_lm.server import ModelProvider, run

MODEL = "lmstudio-community/Olmo-3.1-32B-Instruct-MLX-8bit"# "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-q8" #"mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"

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
"lmstudio-community/Olmo-3.1-32B-Instruct-MLX-8bit": """
{%- set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 -%}{%- if not has_system -%}{{- '<|im_start|>system
You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai. ' -}}{%- if tools is none or (tools | length) == 0 -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
' -}}{%- else -%}{{- 'You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions.' -}}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions><|im_end|>
' -}}{%- endif -%}{%- endif -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system
' + message['content'] -}}{%- if tools is not none -%}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions>' -}}{%- elif message.get('functions', none) is not none -%}{{- ' <functions>' + message['functions'] + '</functions>' -}}{%- endif -%}{{- '<|im_end|>
' -}}{%- elif message['role'] == 'user' -%}{{- '<|im_start|>user
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'assistant' -%}{{- '<|im_start|>assistant
' -}}{%- if message.get('content', none) is not none -%}{{- message['content'] -}}{%- endif -%}{%- if message.get('function_calls', none) is not none -%}{{- '<function_calls>' + message['function_calls'] + '</function_calls>' -}}{% elif message.get('tool_calls', none) is not none %}{{- '<function_calls>' -}}{%- for tool_call in message['tool_calls'] %}{%- if tool_call is mapping and tool_call.get('function', none) is not none %}{%- set args = tool_call['function']['arguments'] -%}{%- set ns = namespace(arguments_list=[]) -%}{%- for key, value in args.items() -%}{%- set ns.arguments_list = ns.arguments_list + [key ~ '=' ~ (value | tojson)] -%}{%- endfor -%}{%- set arguments = ns.arguments_list | join(', ') -%}{{- tool_call['function']['name'] + '(' + arguments + ')' -}}{%- if not loop.last -%}{{ '
' }}{%- endif -%}{% else %}{{- tool_call -}}{%- endif %}{%- endfor %}{{- '</function_calls>' -}}{%- endif -%}{%- if not loop.last -%}{{- '<|im_end|>' + '
' -}}{%- else -%}{{- eos_token -}}{%- endif -%}{%- elif message['role'] == 'environment' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'tool' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- endif -%}{%- if loop.last and add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}{%- endfor -%}
""",
"mlx-community/Ministral-3-14B-Instruct-2512": """
{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = 'You are Ministral-3-14B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYou power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\nThe current date is {today}.\n\nWhen you\'re not sure about some information or when the user\'s request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don\'t have the information and avoid making up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\nNext sections describe the capabilities that you have.\n\n# WEB BROWSING INSTRUCTIONS\n\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n# MULTI-MODAL INSTRUCTIONS\n\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\nYou cannot read nor transcribe audio files or videos.\n\n# TOOL CALLING INSTRUCTIONS\n\nYou may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:\n\n1. When the request requires up-to-date information.\n2. When the request requires specific data that you do not have in your knowledge base.\n3. When the request involves actions that you cannot perform without tools.\n\nAlways prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.' %}
{#- Begin of sequence token. #}
{{- bos_token }}
{#- Handle system prompt if it exists. #}
{#- System prompt supports text content or text chunks. #}
{%- if messages[0]['role'] == 'system' %}
    {{- '[SYSTEM_PROMPT]' -}}
    {%- if messages[0]['content'] is string %}
        {{- messages[0]['content'] -}}
    {%- else %}        
        {%- for block in messages[0]['content'] %}
            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- else %}
                {{- raise_exception('Only text chunks are supported in system message contents.') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '[/SYSTEM_PROMPT]' -}}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
    {%- if default_system_message != '' %}
        {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}
    {%- endif %}
{%- endif %}
{#- Checks for alternating user/assistant messages. #}
{%- set ns = namespace(index=0) %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}
        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}
            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.') }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}
{#- Handle conversation messages. #}
{%- for message in loop_messages %}
    {#- User messages supports text content or text and image chunks. #}
    {%- if message['role'] == 'user' %}
        {%- if message['content'] is string %}
            {{- '[INST]' + message['content'] + '[/INST]' }}
        {%- elif message['content'] | length > 0 %}
            {{- '[INST]' }}
            {%- if message['content'] | length == 2 %}
                {%- set blocks = message['content'] | sort(attribute='type') %}
            {%- else %}
                {%- set blocks = message['content'] %}
            {%- endif %}
            {%- for block in blocks %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- elif block['type'] in ['image', 'image_url'] %}
                    {{- '[IMG]' }}
                {%- else %}
                    {{- raise_exception('Only text, image and image_url chunks are supported in user message content.') }}
                {%- endif %}
            {%- endfor %}
            {{- '[/INST]' }}
        {%- else %}
            {{- raise_exception('User message must have a string or a list of chunks in content') }}
        {%- endif %}
    {#- Assistant messages supports text content or text and image chunks. #}
    {%- elif message['role'] == 'assistant' %}
        {%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}
            {{- raise_exception('Assistant message must have a string or a list of chunks in content or a list of tool calls.') }}
        {%- endif %}
        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- elif message['content'] | length > 0 %}
            {%- for block in message['content'] %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- else %}
                    {{- raise_exception('Only text chunks are supported in assistant message contents.') }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        
        {%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}
            {%- for tool in message['tool_calls'] %}
                {%- set arguments = tool['function']['arguments'] %}
                {%- if arguments is not string %}
                    {%- set arguments = arguments|tojson|safe %}
                {%- elif arguments == '' %}
                    {%- set arguments = '{}' %}
                {%- endif %}
                {{- '[TOOL_CALLS]' + tool['function']['name'] + '[ARGS]' + arguments }}
            {%- endfor %}
        {%- endif %}
        {#- End of sequence token for each assistant messages. #}
        {{- eos_token }}
    {#- Tool messages only supports text content. #}
    {%- elif message['role'] == 'tool' %}
        {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}
    {#- Raise exception for unsupported roles. #}
    {%- else %}
        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role'] + '.') }}
    {%- endif %}
{%- endfor %}
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
