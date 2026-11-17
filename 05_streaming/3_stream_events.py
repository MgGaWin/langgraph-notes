# @Version   : 2.0
# @Author    : HanSir
# @File      : 3_stream_events.py
# @Time      : 2026/11/17
# @Desc      : v3 的类型化投影及协议事件信封
"""运行：python -m 05_streaming.3_stream_events；默认离线，不需要 API Key。

v3 的原始事件使用 method/params/seq。
旧 on_chain_start/on_chat_model_stream 字典属于 astream_events(version="v2")。
"""

from release_support.streaming import build_chat_graph


def main():
    stream = build_chat_graph().stream_events(
        {"messages": [("user", "介绍 LangGraph")]},
        version="v3",
    )
    for channel, item in stream.interleave("values", "messages"):
        if channel == "messages":
            print("模型输出：", str(item.text))
        else:
            print("状态快照：", item)
    print("最终状态：", stream.output)
    print("interleave 同时消费多个投影，不需要重新调用模型。")


if __name__ == "__main__":
    main()
