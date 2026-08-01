import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables (Make sure HUGGINGFACEHUB_API_TOKEN is set in your .env)
load_dotenv()

# 1. Define the model repository from Hugging Face Hub
repo_id = "Qwen/Qwen2.5-7B-Instruct"

# 2. Initialize the HuggingFaceEndpoint (LLM)
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=500,  # Increased from 100 to prevent summary truncation
    temperature=0.1,
)

# 3. Wrap with ChatHuggingFace for compatibility with LangChain chat templates
model = ChatHuggingFace(llm=llm)

# --- Streamlit UI ---
st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
)

template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | model
    
    with st.spinner("Generating summary..."):
        result = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )
        st.write(result.content)