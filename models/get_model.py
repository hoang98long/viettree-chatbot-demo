from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/llm/qwen2.5-3b"

AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=MODEL_PATH
)

AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=MODEL_PATH,
    device_map="auto",
    load_in_4bit=True
)
