# 大语言模型 GPTQ 量化与 vLLM 推理部署实验

## 摘要

这是一个大语言模型 GPTQ 量化与 vLLM 推理部署的实验。
本实验在 Google Colab 环境下，使用 **gptqmodel** 对通义千问 Qwen2.5-1.5B-Instruct 模型进行了 4-bit GPTQ 量化，并成功使用 **vLLM** 推理框架进行高性能部署。实验完整经历了模型选择、环境配置、量化执行、兼容性调试及最终 GPU 加速推理等环节。量化后的线性层权重为 torch.int32 打包格式，证明量化成功。最终在 Tesla T4 GPU 上，模型实现了 **86.75 tokens/s** 的输出速度，验证了 4-bit 量化的压缩效果与 vLLM 的推理性能。

---

## 一、实验背景与目标

大语言模型（LLM）参数规模日益庞大，给部署带来了显存和计算的双重压力。模型量化通过降低权重精度（例如从 FP16 降至 INT4）可显著减少显存占用并加速推理。GPTQ 是一种训练后量化（PTQ）方法，利用二阶信息补偿量化误差，在 4-bit 下仍能保持较高精度。

本次实验目标：
1. 在 Google Colab 上对开源 LLM 进行 GPTQ 4-bit 量化；
2. 使用 vLLM 推理框架加载量化模型并进行性能测试；
3. 通过实践熟悉量化工具链、解决环境兼容性问题，并验证量化模型的实际加速效果。

---

## 二、实验环境与工具

| 项目 | 规格 / 版本 |
| :--- | :--- |
| 硬件 | Google Colab (NVIDIA Tesla T4, 16GB 显存) |
| Python 版本 | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| 量化工具 | **gptqmodel** 7.1.0 |
| 推理框架 | **vLLM** 0.18.0 |
| Transformers | 4.43.4（最终兼容版本） |
| 量化算法 | GPTQ (4-bit, group_size=128, desc_act=False, sym=True) |
| 校准数据集 | **allenai/c4** (前256条) |

---

## 三、实验过程与关键问题

### 3.1 模型选择与量化执行

- 最初尝试使用 LLaMA-2-7B，但我的 HF-token 配置错误，且使用该模型需要向 Meta 申请权限，改为开源模型 **Qwen/Qwen2.5-1.5B-Instruct**。
- 使用 **gptqmodel** 库进行 4-bit 量化，校准数据取自 c4 数据集前 256 条文本。量化配置：bits=4, group_size=128, desc_act=False, sym=True。
- 量化完成后，我们打印了 model.layers.0.mlp.down_proj.qweight 的数据类型为 torch.int32 及其前10个权重，通过检查权重文件确认线性层已转换为 torch.int32 打包格式，证明量化成功。

### 3.2 环境兼容性问题与解决

实验过程中遇到的主要障碍是 **Colab Python 3.12 + CUDA 12.8 与 GPTQ 高性能 kernel（Marlin）的即时编译不兼容**。具体问题包括：
- **auto-gptq** 安装时出现 metadata-generation-failed；
- **gptqmodel** 加载模型时尝试编译 Marlin kernel，但编译过程失败或卡顿；
- vLLM 与新版 Transformers 存在 tokenizer 兼容性错误（AttributeError: 'list' object has no attribute 'keys'）。

**解决方案**：
1. 改用 **gptqmodel** 进行量化，避免 auto-gptq 的安装问题。
2. 在推理阶段，先将 Transformers 降级至 4.37.2 以解决 tokenizer 错误，后又升级至 4.43.4 以满足 vLLM 对 Gemma3Config 的导入需求。
3. 安装 vLLM **0.18.0**，并设置环境变量 VLLM_USE_MARLIN=0 等禁用部分 kernel，最终成功编译 vLLM （编译耗时约5分钟）。
4. 采用 enforce_eager=True 参数避免 CUDA 图优化带来的额外编译延迟。

### 3.3 vLLM 推理部署与性能

成功加载量化模型后，使用官方对话模板进行推理：

```python
from vllm import LLM, SamplingParams
llm = LLM(model="./qwen2.5-1.5b-gptq-4bit", trust_remote_code=True, enforce_eager=True)
prompt = "<|im_start|>user\n请简单解释一下什么是人工智能。<|im_end|>\n<|im_start|>assistant\n"
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

**推理性能**：
- 编译耗时：约 5 分钟。
- 输出速度：**86.75 tokens/s**（Tesla T4 GPU，1.5B 模型，4-bit 量化）。
- 模型输出质量：生成文本连贯、准确，与基座模型能力一致。

**性能分析**：86.75 tokens/s 的生成速度对于 T4 这样的入门级 GPU 和 1.5B 参数量级而言非常理想，充分体现了 4-bit 量化和 vLLM 高效调度带来的加速效果。

---

## 四、实验结果总结

| 指标 | 结果 |
| :--- | :--- |
| 模型 | Qwen2.5-1.5B-Instruct (4-bit GPTQ) |
| 量化后线性层数据类型 | torch.int32 (打包格式) |
| 模型文件大小 | 约 1.0 GB（相比 FP16 压缩约 75%） |
| 推理吞吐 | 86.75 tokens/s |
| 推理延迟 | < 2.5 秒生成长度为 200 个 token 的回复 |
| 推理正确性 | 通过人工评估，回答正确且流畅 |

---

## 五、实验心得与展望

1. **量化工具链的成熟度**：GPTQ 算法本身已非常成熟，但开源工具（auto-gptq, gptqmodel, vLLM）与不同 Python/CUDA 版本的兼容性仍是实际部署中的主要挑战。掌握环境调试能力至关重要。
2. **量化效果的验证**：除了运行推理，检查权重数据类型（如 qweight 是否为 int32）是最可靠的量化成功标志。
3. **性能收益**：4-bit 量化在 T4 GPU 上仍能获得 80+ tokens/s 的生成速度，证明量化对于资源受限环境具有极高实用价值。
4. **后续方向**：可进一步尝试 AWQ 量化、探索更低位宽（如 3-bit、2-bit）的精度损失与速度权衡，或使用 Triton 推理服务器进行生产级部署。此外，我计划学习 nano-vllm，希望对 vLLM 有更深了解。目前已对 PagedAttention 与 Continuous Batching 有初步认识。
5. **科研前沿**：在 ICLR 2026 上，来自 NVIDIA、MIT 等机构的研究者提出了 **QeRL（量化增强强化学习）** 框架，首次系统性地探索了量化技术在强化学习中的巨大潜力。这为我后续的学习方向提供了极具启发性的新视角，我计划在后续研究中，将强化学习作为新的切入点，并逐步向具身智能和机器人领域延伸，探索“高效计算 + 智能决策”的交叉方向。

---

## 附录：关键依赖版本（最终可用环境）

```text
Python               3.12.13
torch                2.11.0+cu128
transformers         4.43.4
vllm                 0.18.0
gptqmodel            7.1.0
optimum              2.1.0
accelerate           0.33.0
```
