# Qwen

## 项目简介

该项目围绕阿里云通义千问（Qwen）系列轻量化大语言模型展开，主要实现了 Qwen1.5-0.5B-Chat、Qwen2.5-0.5B-Instruct、Qwen3-0.6B 三款模型的下载、加载、推理及交互式 ChatBot 部署，旨在提供轻量化、易部署的大模型本地运行解决方案。

### 核心特性

1. **环境适配**：基于 Python 3.8 + PyTorch 2.11.0 构建，依托 CUDA 12.4 和 Accelerate 1.13.0 实现 GPU 加速，支持 GPU/CPU 自动适配，确保模型高效推理。
2. **模型下载**：通过魔搭社区（ModelScope）的`snapshot_download`工具便捷下载三款轻量化 Qwen 模型权重，支持自定义本地保存路径。
3. **推理实现**：提供三款模型的标准化加载与推理代码，适配不同版本 Qwen 模型的交互格式，支持自定义生成参数（如`max_new_tokens`、`temperature`、`top_p`等），满足多样化推理需求。
4. **交互式部署**：支持将三款模型分别部署为 Web 端 ChatBot，通过修改模型路径和端口号（8001/8002/8003）避免端口冲突，可通过网页端实现实时问答交互。

### 项目依赖

| 项目             | 版本号 | 说明                          |
| ---------------- | ------ | ----------------------------- |
| **Python**       | 3.8    | conda 环境 Qwen               |
| **PyTorch**      | 2.11.0 | 自带 GPU 支持                 |
| **Transformers** | 5.3.0  | 大模型加载库                  |
| **CUDA**         | 12.4   | PyTorch 内置，GPU 加速        |
| **Accelerate**   | 1.13.0 | GPU 推理加速                  |
| **GPU 状态**     | 可用   | 可运行 Qwen1.5、2.5、3.0 模型 |

### 下载模型

下载Qwen1.5-0.5B-Chat模型权重⽂件

下载Qwen2.5-0.5B-Instruct模型权重⽂件

下载Qwen3-0.6B模型权重⽂件

（可以去[首页 · 魔搭社区](https://www.modelscope.cn/home)搜索Qwen3.0其他模型下载）

```
from modelscope import snapshot_download

# ===================== 下载 1：Qwen1.5-0.5B-Chat =====================
print("正在下载 Qwen1.5-0.5B-Chat...")
snapshot_download(
    "qwen/Qwen1.5-0.5B-Chat",
    local_dir="./Qwen1.5-0.5B-Chat"
)

# ===================== 下载 2：Qwen2.5-0.5B-Instruct =====================
print("\n正在下载 Qwen2.5-0.5B-Instruct...")
snapshot_download(
    "qwen/Qwen2.5-0.5B-Instruct",
    local_dir="./Qwen2.5-0.5B-Instruct"
)

# ===================== 下载 3：Qwen3-0.6B =====================
print("\n正在下载 Qwen3-0.6B...")
snapshot_download(
    "qwen/Qwen3-0.6B",
    local_dir="./Qwen3-0.6B"
)

print("\n 全部三个模型下载完成！")
print("保存路径：")
print("- ./Qwen1.5-0.5B-Chat")
print("- ./Qwen2.5-0.5B-Instruct")
print("- ./Qwen3-0.6B")
```

![image-20260325111012689](C:\Users\huihu\AppData\Roaming\Typora\typora-user-images\image-20260325111012689.png)

### 案例实现

#### 使用Qwen1.5-0.5B的加载以及推理

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './qwen/Qwen1.5-0.5B-Chat'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print(text)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
    )
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids,
    generated_ids)
    ]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)



```

![image-20260325111920130](C:\Users\huihu\AppData\Roaming\Typora\typora-user-images\image-20260325111920130.png)

#### 使用Qwen2.5-0.5B的加载以及推理

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 设备判断（GPU/CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你的模型路径
model_path = r"D:\Users\huihu\Desktop\2026\damoxing\Qwen\qwen\Qwen2.5-0.5B-Instruct"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()  # 推理模式

# 输入问题
prompt = "简单介绍一下你自己"
messages = [{"role": "user", "content": prompt}]

# 构造输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 输入送到GPU
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 生成（关键：加参数！）
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True
    )

# 解码输出
response = tokenizer.decode(
    generated_ids[0][len(model_inputs.input_ids[0]):],
    skip_special_tokens=True
)

# 打印结果
print("用户：", prompt)
print("模型：", response)
```

![image-20260325113029301](C:\Users\huihu\AppData\Roaming\Typora\typora-user-images\image-20260325113029301.png)

### 使⽤模型构建⼀个ChatBot

##### 更改DEFAULT_CKPT_PATH路径

分别将DEFAULT_CKPT_PATH改成自己模型的路径：

```
DEFAULT_CKPT_PATH = r"D:\Users\huihu\Desktop\2026\damoxing\Qwen\qwen\Qwen1.5-0.5B-Chat"
```

```
DEFAULT_CKPT_PATH = r"D:\Users\huihu\Desktop\2026\damoxing\Qwen\qwen\Qwen2.5-0.5B-Instruct"
```

```
DEFAULT_CKPT_PATH = r"D:\Users\huihu\Desktop\2026\damoxing\Qwen\qwen\Qwen3-0.6B"
```

为避免冲突将parser.add_argument中的default分别改成8001、8002、8003

```
parser.add_argument(
        "--server-port", type=int, default=8001, help="Demo server port."
    )
```

##### 运行代码，点进网页就可以进行问答了![屏幕截图 2026-03-24 164540](C:\Users\huihu\Pictures\Screenshots\屏幕截图 2026-03-24 164540.png)

##### 

![屏幕截图 2026-03-24 164931](C:\Users\huihu\Pictures\Screenshots\屏幕截图 2026-03-24 164931.png)



![image-20260325114916849](C:\Users\huihu\AppData\Roaming\Typora\typora-user-images\image-20260325114916849.png)
