import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# 1. Load environment variables from .env
load_dotenv()

# 2. Define the model from Hugging Face Hub (e.g., Mistral-7B)
repo_id = "Qwen/Qwen2.5-7B-Instruct"

# 3. Initialize the HuggingFaceEndpoint (LLM)
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task='text-generation',
    max_new_tokens=10,
    temperature=0.1)

model=ChatHuggingFace(llm=llm)

result = model.invoke("Capital of India")
print(result.content)