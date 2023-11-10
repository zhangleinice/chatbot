from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from sse_starlette.sse import EventSourceResponse
import threading
import os

llm = ChatOpenAI(    
    openai_api_key= os.environ["OPENAI_API_KEY"],
    temperature=0, 
    streaming=True
)

# 流式输出（非LLMChain）
def stream_output(input="你好"):
    messages= [ChatMessage(role="user", content=input)]
    # generator object
    result = llm._stream(messages)

    return result

# res = stream_output('hello')
# print('res', res)
# <generator object ChatOpenAI._stream at 0x116b66700>


# 自定义langchain callback
class My_StreamingStdOutCallbackHandler():

    tokens = []
    finish = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.finish = True

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.tokens.append(str(error))

    def generate_tokens(self):
        while not self.finish:  # or self.tokens:
            if self.tokens:
                token = self.tokens.pop(0)
                yield {'data': token}
            else:
                pass
                # time.sleep(0.02)  # wait for a new token


def f1(llm, query):
    llm.predict(query)

def f2(llm_chain, query):
    llm_chain.run(query)

# sse流式输出
def sse_output(query='你好'):
    callback = My_StreamingStdOutCallbackHandler()

    # 创建一个新线程
    thread = threading.Thread(target=f1, args=(llm, query))
    thread.start()

    return EventSourceResponse(callback.generate_tokens(), media_type="text/event-stream")


res = sse_output()
print('res', res)
# <sse_starlette.sse.EventSourceResponse object at 0x12ae676d0>