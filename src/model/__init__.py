from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'graph_llm': GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    # 'llama2-7b': '/home/neo/Projects/LLMs/llama2/llama2-7b-hf',
    # 'llama2-13b': 'llama2/llama2-13b-hf',
    # 'llama2-7b-chat': 'llama2/llama2-7b-chat-hf',
    # 'llama2-13b-chat': 'llama2/llama2-13b-chat-hf',
    'llama3-8b-4bit': '/home/neo/LLMs/llama3/llama3-8b-bnb-4bit',
    # 'llama3-8b': '/home/neo/Projects/LLMs/llama3/Meta-Llama-3-8B',
}
