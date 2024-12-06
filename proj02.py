
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

# Streamlit Settings
st.set_page_config(page_title="Your AI assistant 🤖", page_icon="🤖")
st.title("Your AI assistant 🤖")

model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]

## Model Providers
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
  llm = HuggingFaceEndpoint(
      repo_id=model,
      temperature=temperature,
      max_new_tokens=512,
      return_full_text=False,
      #model_kwargs={
      #    "max_length": 64,
      #    #"stop": ["<|eot_id|>"],
      #}
  )
  return llm

def model_openai(model="gpt-4o-mini", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
        # other parameters...
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm


def model_response(user_query, chat_history, model_class):

    ## Loading the LLM
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    ## Prompt definition
    system_prompt = """
    You are a helpful assistant answering general questions. Please respond in {language}.
    """
    # corresponds to the language we want in our output
    language = "the same language the user is using to chat" #or, force to answer in a specific language e.g. english

    # Adapting to the pipeline (for open source model with hugging face)
    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = "{input}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])

    ## Creating the Chain
    chain = prompt_template | llm | StrOutputParser()

    ## Response return / Stream
    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, I'm your virtual assistant! How can I help you?"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Enter your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))
        print(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=resp))
