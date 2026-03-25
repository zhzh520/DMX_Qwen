# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = r"D:\Users\huihu\Desktop\2026\damoxing\Qwen\qwen\Qwen3-0.6B"


def _get_args():
    parser = ArgumentParser(description="Qwen3-Instruct web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8003, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    # 使用官方模型名称（会自动从 HuggingFace 下载）
    model_name = "Qwen/Qwen3-0.6B"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    ).eval()

    # 设置生成参数
    model.generation_config.max_new_tokens = 2048
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.do_sample = True

    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    # 构建对话历史
    messages = []
    for query_h, response_h in history:
        messages.append({"role": "user", "content": query_h})
        messages.append({"role": "assistant", "content": response_h})
    messages.append({"role": "user", "content": query})

    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        timeout=60.0,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "do_sample": True,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)
            yield _chatbot
            full_response = response

        print(f"Qwen: {full_response}")
        _task_history.append((_query, full_response))

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        # ========== 这里是修复的核心：更换了可用的 Qwen3 Logo 链接 ==========
        gr.Markdown("""
        <p align="center"><h1 style="color:#0078D4; font-size:40px;">🤖 Qwen1.5 Chat Bot</h1></p>""")

        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen3-0.6B-Instruct, developed by Alibaba Cloud. \
(本WebUI基于Qwen3-0.6B-Instruct打造，实现聊天机器人功能。)</center>"""
        )
        gr.Markdown("""\
<center><font size=4>
Qwen3-0.6B-Instruct | 
<a href="https://github.com/QwenLM/Qwen3">Github</a></center>""")

        chatbot = gr.Chatbot(label="Qwen3", elem_classes="control-height")
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen3. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen3的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()