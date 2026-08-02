import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 1. Load environment variables from .env
load_dotenv()

# 2. Define the model from Hugging Face Hub (e.g., Mistral-7B)
repo_id = "Qwen/Qwen2.5-7B-Instruct"

# 3. Initialize the HuggingFaceEndpoint (LLM)
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task='text-generation',
    max_new_tokens=100,
    temperature=0.1)

model=ChatHuggingFace(llm=llm)


parser = StrOutputParser()


prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})
print(result)

chain.get_graph().print_ascii()