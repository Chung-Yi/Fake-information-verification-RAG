from typing import Literal

from hf_model.hf_chatmodel_loader import HF_LLM
from google_scraper import verify_events_with_tfc

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

class info_verification_with_web:
    
    def __init__(self, llm, tool_specific_param = {}):
        
        tools = [verify_events_with_tfc]
        self.tool_node = ToolNode(tools)

        self.llm = llm
        self.llm_with_tools = llm.bind_tools(tools)

        self.tool_specific_param = tool_specific_param

    def llm_engine_with_tool(self, state: MessagesState):
        system_prompt = "你是查證專家，根據查證工具提供的資訊進行確認事實的真實性"
        
        if state["messages"][0].type != 'system':
            system_message = [('system',system_prompt)]
            state["messages"] = system_message + state["messages"]
        else:
            state["messages"][0] =  [('system',system_prompt)]

        messages = state["messages"]        
        response = self.llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def llm_engine(self, state: MessagesState):
        system_prompt  = ''''
            你是彙整專家，根據對話內容彙整查證結果，請依循下列規劃輸出結果:
            (1) 請先確認查證的問題與資訊是否符合，若不是請統一回覆「查無消息」
            (2) 輸出內容需要包含查證結果、資訊來源及資訊網址，定義如下:
                - 查證結果: 此部分根據查證工具回傳的資訊內容進行彙整，並說明事件的真實性
                - 資訊來源: 提供資訊的來源
                - 資訊網址: 提供資訊來源的相關網址
            (3) 請根據下方格式進行回覆:

            【查證結果】
            {此地方填入查證的結果說明及相關資訊內容，並提供正確資訊或方法給使用者}

            資訊來源:{此地方填入資訊來源}
            資訊網址:{此地方填入資訊網址}
        '''.replace(' ','').strip()

        if state["messages"][0].type != 'system':
            system_message = [('system',system_prompt)]
            state["messages"] = system_message + state["messages"]
        
        else:
            state["messages"][0] =  [('system',system_prompt)]

        messages = state["messages"]        
        response = self.llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def should_continue(self, state: MessagesState) -> Literal["tools", "summary"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:

            tool_calls = []
            for tool in last_message.tool_calls:
                if tool['name'] == 'verify_events_with_tfc':
                    tool['args'].update(self.tool_specific_param)
                tool_calls.append(tool)

            state["messages"][-1] = AIMessage(content="",tool_calls=tool_calls)

            return "tools"
        return "summary"
    
    def graph(self):

        agent_workflow = StateGraph(MessagesState)
        
        agent_workflow.add_node("tool_agent", self.llm_engine_with_tool)
        agent_workflow.add_node("summary", self.llm_engine)
        agent_workflow.add_node("tools", self.tool_node)

        agent_workflow.add_edge("__start__", "tool_agent")
        agent_workflow.add_conditional_edges(
            "tool_agent",
            self.should_continue,
        )
        agent_workflow.add_edge("tools", "tool_agent")
        agent_workflow.add_edge("summary", "__end__")

        return agent_workflow.compile()

if __name__ == "__main__":
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("aigo Tracing")
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
    # mlflow.langchain.autolog(log_models=True, log_input_examples=True)
    # from langchain_community.callbacks.mlflow_callback import MlflowCallbackHandler
    # mlflow_callback = MlflowCallbackHandler(
    #     tracking_uri = "http://127.0.0.1:8080"
    # )

    ####  model loading
    model_name_or_path = "./model_files/Meta-Llama-3.1-8B-Instruct"    
    llm = HF_LLM(model_name_or_path, model_config = {'device_map':'auto'}, generate_config = {'temperature':0.4})

    #### parameter setting
    tool_specific_param = {
        # 'zh_sites': ['tfc'],
        'zh_sites': ['tfc'],
        'en_sites':['snopes']
        # 'en_sites':[]
    }

    search_agent_with_web = info_verification_with_web(
        llm = llm,
        tool_specific_param = tool_specific_param
    ).graph()

    from IPython.display import Image, display
    display(Image(search_agent_with_web.get_graph().draw_mermaid_png()))

    inputs = {
        'messages':[
            ("human", "近期賴清德致詞，背後出現日本軍旗，請問這個是真的嗎?")
            # ("human", "近期川普中槍，請問這個是真的嗎?")
        ]
    }
    config = {
         'callbacks' : [MlflowLangchainTracer()]
    }
    
    async for event, chunk in search_agent_with_web.astream(inputs, stream_mode=["updates"], config  = config):
        # print(f"Receiving new event of type: {event}...")
        # print(chunk)
        node_name = list(chunk.keys())[0]
        chunk[node_name]['messages'][0].pretty_print()
