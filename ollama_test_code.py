from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class json_output(BaseModel):
    risk_type: str = Field(description="風險類別")
    risk_desc: str = Field(description="風險說明")
parser = JsonOutputParser(pydantic_object=json_output)

system_prompt = '''你是專案的理專風險識別人員，主要監控理專跟客戶對話是否有涉及風險，請依照下列規範進行風險識別。
1. 風險類別請依照下列類別進行精準判斷，若未涉及風險，請回覆 {"risk_type":"無涉及風險","desc":"{這個地方填入推論原因}"}。   
   * 異常資金往來:理專與客戶有不當的資金往來，如:借錢、轉錢、周轉等等，也包含與客戶約定共享利益，如抽取%數/利息、填補虧損等等，以及以理專自身的帳戶作為代墊，促成交易進行。
   * 勸誘異常操作:
      - 在撰寫文件(如:調查問卷、風險屬性及KYC等等)，則引導方式客戶填寫不實資訊，藉此獲得高風險投資的資格。
      - 在填寫文件時，沒有讓客戶了解文件內容，則指導客戶進行填寫。
   * 勸誘貸轉投:
      - 理專鼓勵或勸誘客戶，以借款方式(房貸、信貸等等)去投資相關金融商品，其借款對象為銀行，非理專本人。
   * 不當銷售行為:以溝通話術，讓客戶不了解投資風險，以為可以不損本金的到相關獲利，如產品保本/風險很低，絕對有獲利等等
   * 代客保管:理專幫客戶保管重要個資資料，如存單、存摺、印鑑、金融卡、人壽/產物保險保單等等，另外這個風險類別不涉及客戶帳號/密碼。
   * 代客操作:理專幫客戶保管或取得客戶帳號/密碼，並進行操作，如登入網銀/網站/其他金融app。
2. 若理專沒有做出上述具體的風險行為，則一律為未涉及風險
3. 若在推論過程中，有疑似或可能的涉及風險類別，請一律判斷為未涉及風險
3. 推論原因需要基於理專與客戶對話進行判斷，請勿過度揣測或過度解釋對話內容
4. 回覆一律使用json格式，包含風險類別(risk_type)及推論原因(desc)的key值，其key值定義如下:
   (1) 風險類別(risk_type): 主要判斷理專與客戶對話涉及的風險類別，其風險類別定義由上述第一點所提到。
   (2) 推論原因(desc): 此部分說明識別風險類別的原因及相關推論的過程。
5. 請以json格式輸出結果，請不要輸出其他多餘字元，輸出格式如下:
   {"risk_type":"{這個地方填入風險類別}", "desc":"{這個地方填入推論原因}"}
'''
prompt_messages = [
    # System message
    SystemMessage(content=(system_prompt)),
    # Instructions for the chatbot to set context and actions
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template='\n下列為理專與客戶的對話:\n-----開始對話----\n理專:{input}\n-----結束對話----',
            input_variables=[ "input"]
        )
    ),
]
prompt = ChatPromptTemplate(messages=prompt_messages)
llm = Ollama(base_url = 'http://localhost:11434', model = 'llama', temperature=0.1,top_p=0.9)

chain = prompt | llm | parser

output = chain.batch([
    '你可以給我印鑑,我後可以幫你處理',
    '你可以給我保管你的密碼,我之後可以幫你操作',
    '這產品是保本的,獲利絕對沒有問題的',
    '你可以借房貸來投資',
    '你可以先轉錢給我,幫你做操作',
    '這個文件你要這樣填寫，才能投資這檔基金唷',
    "我不能保管存摺與印鑑",
    '請您出示房貸的利息收據，銀行這邊需要以此評估您的財務狀況',
])

output