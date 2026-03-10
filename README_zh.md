# VidKV：面向视频大语言模型的即插即用式 _1.x-Bit_ KV Cache 量化

[Keda Tao](), [Haoxuan You](https://hxyou.github.io/), [Yang Sui](https://eclipsess.github.io/yangsui.github.io/), [Can Qin](https://canqin.tech/), [Huan Wang](https://huanwang.tech/), "Plug-and-Play _1.x-Bit_ KV Cache Quantization for Video Large Language Models"

[[论文](https://arxiv.org/abs/2503.16257)]

#### 🔥🔥🔥 最新动态

- **2025-03-21:** 本仓库已发布。
- **2025-03-21:** 论文已发布。

![overview](figures/method.png)

> **摘要：** 视频大语言模型（VideoLLMs）已经展现出处理更长视频输入并进行复杂推理与分析的能力。然而，由于视频帧会产生数千个视觉 token，KV cache 会显著增加内存需求，进而成为推理速度与显存占用的瓶颈。KV cache 量化是解决这一问题的常见方法。我们发现，对 VideoLLMs 进行 2-bit KV 量化几乎不会损伤模型性能，但更低比特下的量化极限尚未被系统研究。为此，我们提出 VidKV，这是一种即插即用的 KV cache 量化方法，可将 KV cache 压缩到低于 2 bit。具体而言，(1) 对于 key，我们提出通道维混合精度量化策略：异常通道采用 2-bit 量化，普通通道采用 1-bit 量化并结合 FFT；(2) 对于 value，我们实现了 1.58-bit 量化，并通过选择性过滤语义显著的视觉 token 进行定向保留，以获得精度与性能之间更优的平衡。重要的是，我们的研究表明，VideoLLMs 的 value cache 应采用逐通道量化，而不是以往 LLM KV cache 量化工作中常用的逐 token 量化方式。实验上，在 LLaVA-OV-7B 和 Qwen2.5-VL-7B 上的六个基准测试结果显示：相较于 FP16，VidKV 可以将 KV cache 有效压缩到 1.5-bit 与 1.58-bit，且几乎无性能下降。

## ⚒️ TODO

* [x] 发布论文
* [x] 发布代码
* [ ] 支持更多模型

## 安装
##### 1. **克隆仓库并进入项目目录：**
```bash
git clone https://github.com/KD-TAO/VidKV.git
cd VidKV
```

##### 2. **安装推理环境：**
```bash
conda create -n vidkv python=3.10 -y
conda activate vidkv
pip install --upgrade pip  # 启用 PEP 660 支持
pip install -e ".[train]"
# Transformers
bash env_setup.sh
```

## VidKV 使用说明

我们基于 **transformers** 实现了 VidKV 量化框架。  
##### 1. **满足以下两种条件之一：**
- 使用我们提供的 `vidkv` 环境。
- 在你当前环境中安装本仓库提供的 **transformers** 版本。
```bash
cd transformers
pip install .
cd ..
```

##### 2. **参数设置与生成调用：**

```python
# 首先，需要配置 QUANTIZATION_CONFIG
### -> 2-bit 量化:
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheLM', 'nbits': 2, 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### -> 1.5-bit + 2-bit 量化: [1.5, 2] 表示 K-1.5-bit, V-2-bit
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheLM', 'nbits': [1.5, 2], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### -> 1.5-bit + 1.58-bit 量化:
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLM', 'nbits': [1.5, 1.58], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### -> 带 STP 的量化:
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLMSTP', 'nbits': [1.5, 1.58], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1, 'vidkv_stp': 0.2}"
'''
# 目前 QuantizedCacheVLMSTP 仅支持 llava-onevision 模型，其他模型后续支持

# 然后，在模型生成时传入:
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config=QUANTIZATION_CONFIG)
```

## Demo
- 我们提供了基于 LLaVA-OV 模型的推理脚本：
```bash
python demo.py
```
- 你可以在此处修改量化设置：
```python
...
outputs = model.generate(
input_ids,
attention_mask=attention_mask, # Add attention mask
images=image_tensors,
image_sizes=image_sizes,
do_sample=False,
temperature=0,
max_new_tokens=1000,
modalities=["video"],
output_attentions=False,
use_cache = True,
return_dict_in_generate=True,
output_hidden_states=True,
cache_implementation="quantized",
# VidKV
cache_config={"backend": "QuantizedCacheVLM", "nbits": [1.5, 1.58], "q_group_size": 32, "residual_length": 128,"axis_key":-1, "axis_value":-1},
)
...
```

## 评测
评测说明即将发布。

## 联系方式

如果你有任何问题，欢迎联系：KD.TAO@outlook.com

## 引用

如果这项工作对你的研究有帮助，欢迎引用我们的论文：

```bibtex
@article{vidkv,
  title={Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models},
  author={Tao, Keda and You, Haoxuan and Sui, Yang and Qin, Can and Wang, Huan},
  journal={arXiv preprint arXiv:2503.16257},
  year={2025}
}
```
