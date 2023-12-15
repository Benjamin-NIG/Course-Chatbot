
# Course-chatbot-using-LLM
A chatbot is developed by RAG approach. The LLM used is llama2 by Meta with llamaIndex framework.

The external libraries used for this work include:

1) streamlit
2) OpenAi
3) llama-index, and
4) langchain

The procedure adopted in the work as follows:

1) the splitting of the document into chunk size, application of embedding, and storing the obtained vector into a vector database. 
2) Adopted gpt 3.5 turbo large language model. 
3) implementation of retrieval augmentation (RAG) approach by creating a link between the document, the LLM model, and the desired system behaviour and query response using the llama-index framework
4) development of frontend application for the chatbot using streamlit library (main.py)
5) The required dependencies for the replication of the work were documented in the requirements.txt file
