# @Version   : 1.0
# @Author    : HanSir
# @File      : streaming.py
# @Time      : 2026/11/17
# @Desc      : 为流式课程构造可注入模型的图
"""默认使用本地模型；调用者可显式传入 init_llm 中的真实模型。"""

from langgraph.graph import END, START, MessagesState, StateGraph

from release_support.models import text_model


def build_chat_graph(model=None):
    """同步和异步节点分别调用对应模型接口，保留取消传播。"""
    model = model if model is not None else text_model()

    async def answer(state: MessagesState):
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    def answer_sync(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    # RunnableLambda 接受同步和异步实现；同步 stream 也能正常工作。
    from langchain_core.runnables import RunnableLambda

    builder = StateGraph(MessagesState)
    builder.add_node("answer", RunnableLambda(answer_sync, afunc=answer))
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()
