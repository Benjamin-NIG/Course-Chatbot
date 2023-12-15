import torch
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index.prompts.prompts import SimpleInputPrompt
#from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
#from llama_index.llms import HuggingFaceLLM
#from llama_index.embeddings import LangchainEmbedding
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings


st.set_page_config(page_title="Chat CourseBot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# st.secrets["db_username"]
openai.api_key = st.secrets["openai_key"]

st.title("Chat with BCN_4787 CourseBot 💬🦙")
st.info("", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about BCN_4787 Course!"}
    ]

# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         load_in_8bit_fp32_cpu_offload=True
#     )

# @st.cache_resource(show_spinner=False)
# def load_data():
#     #  # Define variable to hold llama2 weights naming 
#     # name = "meta-llama/Llama-2-7b-chat-hf"

#     # # Create tokenizer
#     # tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)
#     # model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
#     #                         , use_auth_token=auth_token, torch_dtype=torch.float16, 
#     #                         quantization_config=bnb_config, device_map='auto') 
    
#     system_prompt = """<s>[INST] <<SYS>>
#     You are a helpful, respectful and honest assistant. Always answer as 
#     helpfully as possible, while being safe. Your answers should not include
#     any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
#     Please ensure that your responses are socially unbiased and positive in nature.

#     If a question does not make any sense, or is not factually coherent, explain 
#     why instead of answering something not correct. If you don't know the answer 
#     to a question, please don't share false information.

#     Your goal is to provide answers relating to the course information.<</SYS>>"""

    # query prompt wrapper
    #query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    # Create a HF LLM using the llama index wrapper 
    # llm = HuggingFaceLLM(context_window=4096,
    #                     max_new_tokens=256,
    #                     system_prompt=system_prompt,
    #                     query_wrapper_prompt=query_wrapper_prompt,
    #                     model_name=model,
    #                     tokenizer_name=tokenizer)

    # # Create and dl embeddings instance  
    # embeddings=LangchainEmbedding(
    #     HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # )

    # # Create new service context instance to allow llmindex work with huggingface

    # service_context = ServiceContext.from_defaults(
    #     chunk_size=1024,
    #     chunk_overlap=20,
    #     llm=llm,
    #     embed_model=embeddings
    # )

    # reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    # docs = reader.load_data()

    # index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    # return index

def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        system_prompt="""You are an expert on the BCN_4787C course content and your job is to answer technical questions.
                        Assume that all questions are related to the Course information.
                        Keep your answers technical and based on facts – do not hallucinate features."""
        
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                                                                   system_prompt=system_prompt))
        
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question about BCN_4787C Course information"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
