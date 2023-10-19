# 订单查询
import re
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import tool
from models.model import llama2

ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"

ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date": "2023-01-03",
    "estimated_delivered_date": "2023-01-05",
}

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date": None,
    "estimated_delivered_date": None,
}

answer_order_info = PromptTemplate(
    template="请把下面的订单信息回复给用户： \n\n {order}?", input_variables=["order"]
)

# 查询订单
answer_order_llm = LLMChain(
    llm=llama2,
    prompt=answer_order_info
)


@tool("Search Order", return_direct=True)
def search_order(input: str) -> str:

    # 加提示语：找不到订单的时候，防止重复调用 OpenAI 的思考策略，来敷衍用户
    """useful for when you need to answer questions about customers orders"""
    pattern = r"\d+[A-Z]+"
    match = re.search(pattern, input)

    order_number = input
    if match:
        order_number = match.group(0)
    else:
        return "请问您的订单号是多少？"
    if order_number == ORDER_1:
        return answer_order_llm.run(json.dumps(ORDER_1_DETAIL))
    elif order_number == ORDER_2:
        return answer_order_llm.run(json.dumps(ORDER_2_DETAIL))
    else:
        return f"对不起，根据{input}没有找到您的订单"
