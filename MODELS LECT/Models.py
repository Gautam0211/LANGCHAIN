import os
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="venv/.env")
# DEBUG CHECK
print("Loaded Key:", repr(os.getenv("OPENAI_API_KEY")))

# Fixed class name capitalization and model string format
llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("Mica Capital of India")

print(result)