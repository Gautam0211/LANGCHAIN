import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Setup model
repo_id = "Qwen/Qwen2.5-7B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task='text-generation',
    max_new_tokens=200,
    temperature=0.1
)
model = ChatHuggingFace(llm=llm)

# Define a PromptTemplate with placeholders
template = PromptTemplate(
    template="Summarize the paper titled '{paper_name}' in a {style} style within {length} sentences.",
    input_variables=["paper_name", "style", "length"]
)

# Format/Invoke the prompt
formatted_prompt = template.invoke({
    "paper_name": "Attention Is All You Need",
    "style": "simple and intuitive",
    "length": "3"
})

# Invoke the model
result = model.invoke(formatted_prompt)
print(result.content)