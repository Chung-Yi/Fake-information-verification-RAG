import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# 設定LLM
llm = Ollama(base_url='http://1.173.2.45:11434', model='llama3', temperature=0.1, top_p=0.9)



#===============================================================================================================================================#
# 定義Pydantic模型
class json_output_bad(BaseModel):
    bad_type: str = Field(description="不良類別")
    bad_desc: str = Field(description="不良推論")

# 系統提示設定
system_prompt_bad =     """請根據以下定義判斷'{content}'文本中是否包含不良資訊，並以繁體中文格式返回結果，格式需遵守及固定為以下兩行，勿出現其他多餘語句：
    - 不良類別: [類別1, 類別2, 類別3]
    - 不良推論: 原因描述
    
    不良類別字段應該是一個包含'色情,毒品,賭博,暴力,自殺,武器'類別的列表，這些類別用英文逗號分隔。 
    不良推論字段應該是一個描述整體推論是否包含不良類別的原因的字符串。
    
    '色情'類別定義：文本中若明確出現性行為描寫、性暗示或裸露的描述，或提及應召站、春色場所或色情服務，這包括詳細描寫性活動的語句、性暗示的用語，以及任何涉及裸露身體部位的描述。
    '毒品'類別定義：文本中若提到毒品的持有、查獲、吸食、使用、販售、交易、走私或製造，則歸類於此類別。
    '賭博'類別定義：文本中若描述賭博活動的進行、行為描述、宣傳、參與、推廣或相關內容，則歸類於此類別。
    '暴力'類別定義：文本中若涉及暴力行為的描述或血腥的內容，例如打鬥、攻擊、傷害、毆打等，具體的身體暴力行為或傷害的描寫，或威脅他人會遭受身體傷害的言語或行為、精神虐待，則歸類於此類別。
    '自殺'類別定義：文本中若含有自殺行為、對身體傷害的詳細描寫、鼓勵自殺、自傷或自殘的內容，則歸類於此類別。
    '武器'類別定義：文本中若含有槍枝武器的製造、持有、販售或使用，或提及任何武器的使用和效果，則歸類於此類別。
    
    當上述皆不符合時，不良類別顯示'未含不良資訊'。
    """

prompt_messages_bad = [
    SystemMessage(content=system_prompt_bad),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template='{input}',
            input_variables=["input"]
        )
    ),
]
prompt_bad = ChatPromptTemplate(messages=prompt_messages_bad)
chain_bad = prompt_bad | llm

#===============================================================================================================================================#
# 定義Pydantic模型
class json_output_fake(BaseModel):
    fake_account_type: str = Field(description="假帳號類別")
    fake_account_desc: str = Field(description="假帳號推論")

# 系統提示設定
system_prompt_fake = '''
你是一名專業的內容審核員，負責判斷文本是否包含以下兩種假帳號資訊，請依照下列類別進行精準識別。
1. 【兩種假帳號類別定義】：假帳號資訊請依照下列類別進行判斷，且需要文本中非常明顯提到該類別相關的內容，才能判斷為含有假帳號資訊。若文本未明確判斷出涉及以下兩個假帳號類別，請直接回覆"假帳號類別:未含假帳號資訊。假帳號推論:{這個地方填入推論原因}。"
   (1). 假活動或身份：若文本中提到偽造的活動、虛假身份或任何偽裝成真實個人的內容，判定為假活動或身份。
   (2). 不正當行為：若文本中涉及投資詐騙、投資內線分享、欺詐行為或其他非法活動的內容，判定為不正當行為。

2. 【假帳號資訊的判斷方式】：
   (1). 假帳號資訊識別需基於文本內容進行判斷，請勿過度揣測。
   (2). 需要非常明顯提及相關內容，才能判斷為該類別。

3. 【輸出格式定義】：
   (1). 回覆一律為字串，包含"假帳號類別"及"推論原因"的key值，其key值定義如下:
        - 假帳號類別: 可同時輸出單個或多個假帳號類別，以逗號分隔，並以句號結束；記錄是否判斷出含有上述兩種假帳號資訊，若同時判別出符合多個假帳號類別，則輸出時一起提供。
        - 推論原因: 輸出格式為字串，並以句號結束；說明識別假帳號類別的原因及相關推論的過程，若判斷出的假帳號類別有多個時，需綜合推論原因。

   (2). 【含假帳號資訊時的輸出方式】：
        - 假帳號類別(fake_type)及推論原因(fake_desc)的輸出內容皆不能為空。
        - 請以字串輸出結果，並以多標籤的判別方式進行識別及輸出，即若文本中含有多個假帳號類別時皆需判斷出，請不要輸出其他多餘字元。
        - 例如：假設判斷出文本中同時判斷出有兩個假帳號類別，則輸出格式如下:假帳號推論:假活動或身份、不正當行為。假帳號推論:{這個地方填入整體的推論原因}。"
        - 輸出的字串中，內容僅能為"繁體中文"，若為簡體中文時請協助翻譯為繁體中文後再輸出。

   (3).【無假帳號資訊時的輸出方式】：
        - 若文本沒有判別出現上述兩個假帳號類別，則統一以繁體中文輸出如下:"假帳號類別:未含假帳號資訊。假帳號推論:文本内容未明顯涉及假活動、身份或不正當行為相關的假帳號訊息。"
'''



prompt_messages_fake = [
    # System message
    SystemMessage(content=(system_prompt_fake)),
    # Instructions for the chatbot to set context and actions
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template='{input}',
            input_variables=["input"]
        )
    ),
]

prompt_fake = ChatPromptTemplate(messages=prompt_messages_fake)
chain_fake = prompt_fake | llm

#===============================================================================================================================================#
import streamlit as st
from PIL import Image
import base64
import os
import time

# 將圖片轉為 base64
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error converting image to base64: {str(e)}")
        return None

def response_generator(response):

    for word in str(response).split():
        yield word + " "
        time.sleep(0.5)

def main():
    st.set_page_config(
        page_title="內容審核系統",
        page_icon=":guardsman:",
        layout="wide",
        initial_sidebar_state="auto"  # 預設狀態為展開
    )

    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 2px #333333,  /* 深灰色 */
                0 0 4px #333333,  /* 深灰色 */
                0 0 6px #333333,  /* 深灰色 */
                0 0 8px #333333,  /* 深灰色 */
                0 0 10px #333333, /* 深灰色 */
                0 0 12px #333333, /* 深灰色 */
                0 0 14px #333333; /* 深灰色 */
            position: relative;
            z-index: -1;
            border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("內容審核系統")

    # 初始化 session_state
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'text' not in st.session_state:
        st.session_state.text = ''
    if 'options' not in st.session_state:
        st.session_state.options = []

    # 顯示圖像和控件在左側
    with st.sidebar:
        # 設置圖片路徑
        current_directory = os.path.dirname(__file__)
        img_path = os.path.join(current_directory, 'picture/detective.jpg')  # 確保這是正確的圖片路徑
        img_base64 = img_to_base64(img_path)

        if img_base64:
            st.markdown(
                f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
                unsafe_allow_html=True,
            )

        st.subheader("選擇需要執行的檢查項目")
        
        # 初始化複選框狀態
        verifier_checkbox = st.checkbox('假消息查證員', value=st.session_state.get('verifier_checkbox', False))
        fake_checkbox = st.checkbox('假帳號偵測員', value=st.session_state.get('fake_checkbox', False))
        bad_checkbox = st.checkbox('色情守門員', value=st.session_state.get('bad_checkbox', False))
        
        st.session_state.verifier_checkbox = verifier_checkbox
        st.session_state.fake_checkbox = fake_checkbox
        st.session_state.bad_checkbox = bad_checkbox

        with st.form(key='form_buttons'):
            submit_button = st.form_submit_button('提交', help="提交")
            clear_button = st.form_submit_button('清空', help="清空")  # Update button text to '清空'
        
        if submit_button:
            if st.session_state.text.strip() == '':
                st.session_state.results = {"Error": "請輸入內容！"}
            else:
                results = {}
                if st.session_state.verifier_checkbox:
                    results['假消息查證員'] = st.write_stream(response_generator("等待重億增加"))
                if st.session_state.fake_checkbox:
                    response_fake = chain_fake.invoke({"input": st.session_state.text})
                    results['假帳號偵測員'] = st.write_stream(response_generator(response_fake))

                if st.session_state.bad_checkbox:
                    response_bad = chain_bad.invoke({"input": st.session_state.text})
                    results['色情守門員'] = st.write_stream(response_generator(response_bad))
                st.session_state.results = results

        if clear_button:
            st.session_state.results = {}
            st.session_state.text = ''
            st.session_state.options = []
            st.session_state.verifier_checkbox = False
            st.session_state.fake_checkbox = False
            st.session_state.bad_checkbox = False

    # 主內容區域
    text = st.text_area("輸入要檢查的內容", value=st.session_state.text)
    if text != st.session_state.text:
        st.session_state.text = text

    # 顯示結果
    if st.session_state.results:
        st.write("檢查結果:")
        for key, value in st.session_state.results.items():
            st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()
