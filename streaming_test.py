from hf_model.hf_chatmodel_loader import HF_LLM
from web_agent import WebAgent
from kb_agent import KBAgent
from langgraph.graph import StateGraph, MessagesState

model_name_or_path = "./model_files/Meta-Llama-3.1-8B-Instruct"
llm = HF_LLM(
    model_name_or_path,
    model_config={"device_map": "auto"},
    generate_config={"temperature": 0.01},
)

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async  def call_llm(state):
    input_messages = state['messages']
    response = await llm.ainvoke(input_messages)

    return {'messages':response}


workflow = StateGraph(GraphState)

workflow.add_node("llm", call_llm)

workflow.add_edge('__start__', "llm")
workflow.add_edge('llm', "__end__")

app = workflow.compile()

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

from langchain_core.messages import HumanMessage
from  time import time 

async def response_generator():
    inputs = {
        "messages": [
            ("human", "你好")
        ]
    }   

    async for event in app.astream_events(inputs, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                # print(content, end="|")
                # print(content)

                yield content
                # time.sleep(0.05)

import asyncio
from typing import AsyncGenerator
# def to_sync_generator(async_gen: AsyncGenerator):
#     while True:
#         try:
#             yield asyncio.run(anext(async_gen))
#         except StopAsyncIteration:
#             break

def to_sync_generator(async_gen: AsyncGenerator):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            try:
                yield loop.run_until_complete(anext(async_gen))
            except StopAsyncIteration:
                break
    finally:
        loop.close()

import streamlit as st
import random
import time

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("user"):
    st.write("Hello 👋")


with st.chat_message("assistant"):
    response = st.write_stream(to_sync_generator(response_generator()))
    # response = st.write_stream(llm.stream('你好'))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

