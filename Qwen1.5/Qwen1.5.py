# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = r"D:\Users\huihu\Desktop\2026\damoxing\Qwen\qwen\Qwen1.5-0.5B-Chat"


def _get_args():
    parser = ArgumentParser(description="Qwen1.5-Instruct web chat demo.")
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
        "--server-port", type=int, default=8001, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    # 检查本地路径是否存在
    if os.path.exists(args.checkpoint_path):
        model_name = args.checkpoint_path
        print(f"Loading from local path: {model_name}")
    else:
        model_name = "Qwen/Qwen1.5-0.5B-Instruct"
        print(f"Loading from HuggingFace: {model_name}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,  # 添加 trust_remote_code
        resume_download=True,
        padding_side="left",  # 设置 padding side
    )

    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.cpu_only:
        device_map = "cpu"
        torch_dtype = torch.float32
    else:
        device_map = "auto"
        # 使用 float16 如果 GPU 可用
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 加载模型 - 关键修改：移除 resume_download 参数
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        # resume_download=True,  # 已移除，model.from_pretrained 不接受此参数
        low_cpu_mem_usage=True,  # 添加内存优化
    ).eval()

    # 设置生成配置
    model.generation_config.max_new_tokens = 2048
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.9
    model.generation_config.do_sample = True
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    # 构建对话历史
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})

    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    # 编码输入
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    # 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        timeout=60.0,
        skip_special_tokens=True
    )

    # 生成参数
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # 启动生成线程
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 流式输出
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
        try:
            for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
                response += new_text
                _chatbot[-1] = (_query, response)
                yield _chatbot
                full_response = response
        except Exception as e:
            print(f"Error during generation: {e}")
            _chatbot[-1] = (_query, f"生成出错: {str(e)}")
            yield _chatbot
            return

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
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen1.5_logo.png" style="height: 120px"/><p>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen1.5-0.5B-Instruct, developed by Alibaba Cloud. \
(本WebUI基于Qwen1.5-0.5B-Instruct打造，实现聊天机器人功能。)</center>"""
        )
        gr.Markdown("""\
<center><font size=4>
Qwen1.5-0.5B-Instruct | 
<a href="https://github.com/QwenLM/Qwen1.5">Github</a></center>""")

        chatbot = gr.Chatbot(label="Qwen1.5", elem_classes="control-height")
        query = gr.Textbox(lines=2, label="Input", placeholder="请输入您的问题...")
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
<font size=2>Note: This demo is governed by the original license of Qwen2.5. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen1.5的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    print("=" * 50)
    print("Loading Qwen1.5-0.5B-Instruct model...")
    print("=" * 50)

    model, tokenizer = _load_model_tokenizer(args)

    print("Model loaded successfully!")
    print(f"Device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print("=" * 50)
    print("Starting web interface...")
    print("=" * 50)

    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()