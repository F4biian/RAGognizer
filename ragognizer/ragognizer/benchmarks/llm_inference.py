import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_CURRENT_MODEL = None
_CURRENT_TOKENIZER = None
_CURRENT_MODEL_NAME = None

def get_fallback_template(model_name):
    name = model_name.lower()
    
    # Gemma / Gemma 2 / Gemma 3
    if "gemma" in name:
        return "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'assistant' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"

    # Qwen (ChatML)
    if "qwen" in name:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = 'You are a helpful assistant.' %}"
            "{% endif %}"
            "{{ '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}"
            "{% for message in loop_messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    
    # Vicuna
    if "vicuna" in name:
        return "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{{ system_message + ' ' }}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
    
    # Llama 2
    if "llama-2" in name:
        return "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '</s>' }}{% endif %}{% endfor %}"
    
    # Default
    return "{% for message in messages %}{{ message['role'].title() + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

def load_model_efficiently(model_name):
    global _CURRENT_MODEL, _CURRENT_TOKENIZER, _CURRENT_MODEL_NAME

    if _CURRENT_MODEL_NAME == model_name:
        return _CURRENT_MODEL, _CURRENT_TOKENIZER

    print(f"\ !!! Switching model: {_CURRENT_MODEL_NAME} -> {model_name} !!!")

    # Unload previous
    if _CURRENT_MODEL is not None:
        del _CURRENT_MODEL
        del _CURRENT_TOKENIZER
        _CURRENT_MODEL = None
        _CURRENT_TOKENIZER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Memory cleared.")

    torch_dtype = torch.float16
    if torch.cuda.is_bf16_supported():
        print("Bfloat16 supported. Using bfloat16 for stability.")
        torch_dtype = torch.bfloat16

    # Check for 4-bit requirement
    use_4bit = any(size in model_name.lower() for size in ["33b", "70b", "65b", "mixtral"])
    quantization_config = None
    
    if use_4bit:
        print(f"Detected large model ({model_name}). Enabling 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.chat_template is None:
            print(f"Warning: No chat_template found. Injecting fallback.")
            tokenizer.chat_template = get_fallback_template(model_name)

        if tokenizer.pad_token is None:
            if "qwen" in model_name.lower():
                tokenizer.pad_token_id = 151643
            else:
                tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        _CURRENT_MODEL = model
        _CURRENT_TOKENIZER = tokenizer
        _CURRENT_MODEL_NAME = model_name
        
        return model, tokenizer

    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        raise e

def generate_response(model_name, chat_messages, temperature=0.7, max_new_tokens=512):
    model, tokenizer = load_model_efficiently(model_name)

    input_ids = tokenizer.apply_chat_template(
        chat_messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)

    input_len = input_ids.shape[1]

    # Qwen Specific Settings
    use_cache = True
    eos_token_id = tokenizer.eos_token_id
    
    if "qwen" in model_name.lower() and "1.5" not in model_name:
        use_cache = False 
        eos_token_id = [151645, 151643] 

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache
        )

    generated_tokens = output_ids[0][input_len:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    del input_ids, output_ids, generated_tokens
    return decoded_output