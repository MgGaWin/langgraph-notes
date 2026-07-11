# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_form_filling.py
# @Time      : 2026/6/1 10:00
# @Desc      : 表单填写演示 —— 收集结构化人工输入与数据验证

"""
表单填写概念：
在许多应用场景中，需要引导用户逐步填写结构化表单数据。
例如：
1. 用户注册：收集姓名、年龄、邮箱等个人信息
2. 订单填写：收货地址、支付方式、备注等
3. 调查问卷：多道题目逐项作答
本示例展示如何使用 interrupt() 逐步收集用户资料（姓名、年龄、偏好），
并对用户输入进行验证，确保数据的有效性。
验证失败时会重新请求输入，直到数据合法为止。
"""

# ========== 1. 导入依赖 ===========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将上级目录加入路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ========== 2. 定义状态结构 ===========
class FormState(TypedDict):
    """表单填写的状态定义"""
    name: str               # 用户姓名
    age: int                # 用户年龄
    preferences: list       # 用户偏好列表
    bio: str                # 个人简介
    validation_errors: list # 验证错误信息
    is_complete: bool       # 表单是否完成
    collected_fields: list  # 已收集的字段名


# ========== 3. 数据验证函数 ===========
def validate_name(name: str) -> tuple:
    """
    验证姓名字段。
    规则：非空且长度在2-20个字符之间。
    返回 (是否有效, 错误信息)
    """
    if not name or not name.strip():
        return False, "姓名不能为空"
    if len(name.strip()) < 2:
        return False, "姓名至少需要2个字符"
    if len(name.strip()) > 20:
        return False, "姓名不能超过20个字符"
    return True, ""


def validate_age(age_str: str) -> tuple:
    """
    验证年龄字段。
    规则：必须是1-150之间的整数。
    返回 (转换后的年龄, 是否有效, 错误信息)
    """
    try:
        age = int(age_str)
    except (ValueError, TypeError):
        return 0, False, "年龄必须是整数"
    if age < 1:
        return age, False, "年龄不能小于1"
    if age > 150:
        return age, False, "年龄不能超过150"
    return age, True, ""


def validate_preferences(prefs_str: str) -> tuple:
    """
    验证偏好字段。
    规则：使用逗号分隔，至少选择1个，最多5个。
    返回 (偏好列表, 是否有效, 错误信息)
    """
    if not prefs_str or not prefs_str.strip():
        return [], False, "偏好不能为空"
    # 使用逗号分隔偏好项
    prefs = [p.strip() for p in prefs_str.split(",") if p.strip()]
    if len(prefs) < 1:
        return [], False, "至少需要选择1个偏好"
    if len(prefs) > 5:
        return prefs, False, "偏好不能超过5个"
    return prefs, True, ""


def validate_bio(bio: str) -> tuple:
    """
    验证个人简介字段。
    规则：长度在5-200个字符之间。
    返回 (是否有效, 错误信息)
    """
    if not bio or not bio.strip():
        return False, "个人简介不能为空"
    if len(bio.strip()) < 5:
        return False, "个人简介至少需要5个字符"
    if len(bio.strip()) > 200:
        return False, "个人简介不能超过200个字符"
    return True, ""


# ========== 4. 定义表单填写节点函数 ===========
def collect_name(state: FormState) -> dict:
    """
    收集姓名节点：使用 interrupt() 获取用户姓名。
    支持验证失败后重新输入。
    """
    print("[表单-姓名] 请输入您的姓名")

    # 循环收集直到验证通过
    while True:
        # interrupt：请求用户输入姓名
        name_input = interrupt({
            "字段": "姓名",
            "提示": "请输入您的姓名（2-20个字符）",
            "当前已填写": {
                "姓名": state.get("name", "未填写"),
                "年龄": state.get("age", "未填写"),
                "偏好": state.get("preferences", []),
                "简介": state.get("bio", "未填写")
            }
        })

        # 验证输入
        is_valid, error_msg = validate_name(name_input)
        if is_valid:
            print(f"[表单-姓名] 姓名验证通过: {name_input.strip()}")
            return {
                "name": name_input.strip(),
                "collected_fields": state.get("collected_fields", []) + ["name"]
            }
        else:
            print(f"[表单-姓名] 验证失败: {error_msg}，重新请求输入")
            # 验证失败，将错误信息传给下一次 interrupt
            # 通过返回空字典不更新状态，但需要再次 interrupt
            # 这里我们直接继续循环，再次调用 interrupt
            continue


def collect_age(state: FormState) -> dict:
    """
    收集年龄节点：使用 interrupt() 获取用户年龄。
    验证年龄必须为合法整数。
    """
    print(f"[表单-年龄] 正在为 {state['name']} 收集年龄")

    while True:
        # interrupt：请求用户输入年龄
        age_input = interrupt({
            "字段": "年龄",
            "提示": "请输入您的年龄（1-150之间的整数）",
            "已填写": {"姓名": state.get("name", "")}
        })

        # 验证年龄
        age, is_valid, error_msg = validate_age(str(age_input))
        if is_valid:
            print(f"[表单-年龄] 年龄验证通过: {age}")
            return {
                "age": age,
                "collected_fields": state.get("collected_fields", []) + ["age"]
            }
        else:
            print(f"[表单-年龄] 验证失败: {error_msg}，重新请求输入")
            continue


def collect_preferences(state: FormState) -> dict:
    """
    收集偏好节点：使用 interrupt() 获取用户偏好。
    用户以逗号分隔的方式输入多个偏好。
    """
    print(f"[表单-偏好] 正在为 {state['name']} 收集偏好")

    while True:
        # interrupt：请求用户输入偏好
        prefs_input = interrupt({
            "字段": "偏好",
            "提示": "请输入您的偏好，用逗号分隔（1-5个，如: 阅读,编程,音乐）",
            "已填写": {
                "姓名": state.get("name", ""),
                "年龄": state.get("age", "")
            }
        })

        # 验证偏好
        prefs, is_valid, error_msg = validate_preferences(str(prefs_input))
        if is_valid:
            print(f"[表单-偏好] 偏好验证通过: {prefs}")
            return {
                "preferences": prefs,
                "collected_fields": state.get("collected_fields", []) + ["preferences"]
            }
        else:
            print(f"[表单-偏好] 验证失败: {error_msg}，重新请求输入")
            continue


def collect_bio(state: FormState) -> dict:
    """
    收集个人简介节点：使用 interrupt() 获取用户简介。
    验证简介长度在合法范围内。
    """
    print(f"[表单-简介] 正在为 {state['name']} 收集个人简介")

    while True:
        # interrupt：请求用户输入简介
        bio_input = interrupt({
            "字段": "个人简介",
            "提示": "请输入您的个人简介（5-200个字符）",
            "已填写": {
                "姓名": state.get("name", ""),
                "年龄": state.get("age", ""),
                "偏好": state.get("preferences", [])
            }
        })

        # 验证简介
        is_valid, error_msg = validate_bio(str(bio_input))
        if is_valid:
            print(f"[表单-简介] 简介验证通过（长度: {len(bio_input.strip())}）")
            return {
                "bio": bio_input.strip(),
                "collected_fields": state.get("collected_fields", []) + ["bio"]
            }
        else:
            print(f"[表单-简介] 验证失败: {error_msg}，重新请求输入")
            continue


def confirm_form(state: FormState) -> dict:
    """
    确认表单节点：汇总所有填写内容，让用户最终确认。
    使用 interrupt() 获取确认或修改请求。
    """
    print("[表单-确认] 所有字段填写完毕，等待用户确认")

    # 汇总所有已填写信息
    summary = {
        "姓名": state.get("name", ""),
        "年龄": state.get("age", ""),
        "偏好": state.get("preferences", []),
        "个人简介": state.get("bio", "")
    }

    # interrupt：让用户确认或修改
    confirm_input = interrupt({
        "表单汇总": summary,
        "提示": "以上信息是否正确？输入 'confirm' 确认提交 / 'restart' 重新填写"
    })

    if confirm_input == "confirm":
        print("[表单-确认] 用户确认提交")
        return {"is_complete": True}
    else:
        print("[表单-确认] 用户选择重新填写")
        return {"is_complete": False}


# ========== 5. 条件路由函数 ===========
def after_confirm(state: FormState) -> str:
    """确认后的路由：确认->结束，重新填写->回到开始"""
    if state.get("is_complete", False):
        return "end"
    return "restart"


# ========== 6. 构建表单填写图 ===========
def build_form_graph():
    """构建表单填写流程图"""
    builder = StateGraph(FormState)

    # 添加所有节点
    builder.add_node("collect_name", collect_name)
    builder.add_node("collect_age", collect_age)
    builder.add_node("collect_preferences", collect_preferences)
    builder.add_node("collect_bio", collect_bio)
    builder.add_node("confirm_form", confirm_form)

    # 定义线性流程
    builder.add_edge(START, "collect_name")                      # 起点 -> 姓名
    builder.add_edge("collect_name", "collect_age")              # 姓名 -> 年龄
    builder.add_edge("collect_age", "collect_preferences")       # 年龄 -> 偏好
    builder.add_edge("collect_preferences", "collect_bio")       # 偏好 -> 简介
    builder.add_edge("collect_bio", "confirm_form")              # 简介 -> 确认

    # 确认后的条件分支
    builder.add_conditional_edges(
        "confirm_form",
        after_confirm,
        {
            "restart": "collect_name",   # 重新填写 -> 回到姓名
            "end": END                    # 确认提交 -> 结束
        }
    )

    # 创建检查点并编译图
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 7. 辅助函数：恢复执行并检查状态 ===========
def resume_form(graph, config, value, step_name):
    """
    辅助函数：恢复表单填写流程并检查是否再次暂停。
    返回最终结果或下一次 interrupt 的信息。
    """
    print(f"\n{'*' * 40}")
    print(f"填写 [{step_name}]，输入值: {value}")
    print(f"{'*' * 40}")

    result = graph.invoke(Command(resume=value), config)

    # 检查是否再次暂停在 interrupt
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        field = interrupt_info.value.get("字段", "未知")
        print(f"表单暂停，等待填写: {field}")
        return result, True  # 仍在 interrupt 中
    else:
        print(f"表单填写完毕")
        return result, False  # 执行完成


# ========== 8. 主程序入口 ===========
if __name__ == "__main__":
    # 构建表单图
    graph = build_form_graph()
    config = {"configurable": {"thread_id": "form_filling_demo"}}

    # --- 第一轮调用：启动表单流程，暂停在姓名填写 ---
    print("*" * 40)
    print("启动用户资料填写流程")
    print("*" * 40)

    initial_state = {
        "name": "",
        "age": 0,
        "preferences": [],
        "bio": "",
        "validation_errors": [],
        "is_complete": False,
        "collected_fields": []
    }

    result = graph.invoke(initial_state, config)

    # 检查是否暂停在第一个字段
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"\n等待填写: {interrupt_info.value}")

    # --- 模拟用户逐步填写表单 ---
    print("\n" + "*" * 40)
    print("模拟用户填写表单")
    print("*" * 40)

    # 填写姓名
    result, paused = resume_form(graph, config, "李四", "姓名")

    # 填写年龄
    if paused:
        result, paused = resume_form(graph, config, "28", "年龄")

    # 填写偏好
    if paused:
        result, paused = resume_form(graph, config, "编程,阅读,音乐", "偏好")

    # 填写简介
    if paused:
        result, paused = resume_form(graph, config, "一名热爱技术的软件工程师，喜欢探索新技术", "简介")

    # 确认提交
    if paused:
        result, paused = resume_form(graph, config, "confirm", "确认提交")

    # --- 打印最终结果 ---
    if not paused:
        print("\n" + "*" * 40)
        print("表单填写完成！用户资料")
        print("*" * 40)
        print(f"  姓名: {result.get('name', '')}")
        print(f"  年龄: {result.get('age', '')}")
        print(f"  偏好: {result.get('preferences', [])}")
        print(f"  简介: {result.get('bio', '')}")
        print(f"  已收集字段: {result.get('collected_fields', [])}")
