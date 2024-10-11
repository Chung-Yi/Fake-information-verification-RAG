from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_anthropic import ChatAnthropic

from IPython.display import Image, display
from PIL import Image
from io import BytesIO


llm = ChatAnthropic(model="claude-3-haiku-20240307")

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def main():

    graph_builder = StateGraph(State)
    graph_builder.add_node(chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile()

    # fig, ax = plt.subplots()

    # graph.get_graph().draw_mermaid_png()

    image = Image.open(BytesIO(graph.get_graph().draw_mermaid_png()))

    try:

        # plt.imshow(image)
        image.show()
    except Exception:
        # This requires some extra dependencies and is optional
        pass


    

    



if __name__ == "__main__":
    main()