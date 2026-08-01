import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

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

chat_history=[SystemMessage(content='You are helpful AI assistant')]
while True:
    user_input=input('You:')
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result=model.invoke(chat_history)
    print('AI:',result.content)
    chat_history.append(AIMessage(result.content))

print(chat_history)    


# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# # 1. Load environment variables from .env
# load_dotenv()

# # 2. Define the model from Hugging Face Hub (e.g., Mistral-7B)
# repo_id = "Qwen/Qwen2.5-7B-Instruct"

# # 3. Initialize the HuggingFaceEndpoint (LLM)
# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     task='text-generation',
#     max_new_tokens=100,
#     temperature=0.1)

# model=ChatHuggingFace(llm=llm)

# chat_history=[]
# while True:
#     user_input=input('You:')
#     chat_history.append(user_input)
#     if user_input == 'exit':
#         break
#     result=model.invoke(chat_history)
#     print('AI:',result.content)
#     chat_history.append(result.content)

# print(chat_history)    


