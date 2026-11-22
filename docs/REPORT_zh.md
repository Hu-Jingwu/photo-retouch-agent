# 代码报告：AI 修图 Agent

*第一个 agent 项目，写给学习用。内容截至 2026-11-22。*

## 1. 一句话概括

**模型做决策，工具执行，规则打分。**

多模态大模型（hy4-flash，通过腾讯 Tokenhub 的 OpenAI 兼容接口调用）看照片和工具结果，负责做决定。
基于 OpenCV、rawpy、MediaPipe 的 Python 工具负责测量和改像素。确定性检查和评估 harness 判断结果好不好。
模型从不直接改像素，也不在 harness 里给自己打分。

这是整个项目最核心的一课。模型擅长判断（"哪条线才是真正水平的？""哪个裁剪讲清了故事？"），
不擅长精确数字和可重复的像素操作；代码正好相反。agent 就是让两者各做擅长的事。

## 2. 架构

```
 照片 + 指令（+ 可选：样片 / 风格）
                │
                ▼
 ┌───────────────────────────────┐      ┌──────────────────────────────┐
 │ 编排 agent（agent.py）         │◄─────│ 风格记忆（styles.py）         │
 │ hy4-flash，工具调用循环         │      │ 11 种样片库                   │
 │ 阶段 A → B → C → D            │      │ 只在用户要求风格时使用          │
 └──────────────┬────────────────┘      └──────────────────────────────┘
                │ 工具调用（JSON）            ▲ 图片 + JSON 结果
                ▼                            │
 ┌─────────────────────────────────────────────────────────────────────┐
 │ 工具（agent_tools/，由 registry.py 分发）                              │
 │ A 水平矫正   estimate_tilt · apply_rotation · check_level              │
 │ B 构图裁剪   propose_crops · apply_crop                                │
 │ C 色彩       analyze_color · preview_variants · preview_looks ·        │
 │              render_adjustments · apply_color · match_reference ·      │
 │              suggest_styles · apply_style                              │
 │ D 人像       detect_people · preview_retouch · apply_retouch           │
 └──────────────┬──────────────────────────────────────────────────────┘
                ▼
 ┌───────────────────────────────┐   通过   ┌──────────────────────────┐
 │ 渲染 + 评估                    │─────────►│ 导出：16 位 TIFF、         │
 │ 工具内检查 + harness           │          │ JPEG 预览、参数 JSON       │
 └──────────────┬────────────────┘          └──────────────────────────┘
                │ 不通过：问题返回给模型（最多重试 2–3 次）
                └──────────────► 编排 agent
```

## 3. 文件

| 文件 | 行数 | 作用 |
|---|---|---|
| `agent.py` | ~400 | agent 本体：系统提示词、工具调用循环、路径固定、最终 JSON |
| `agent_tools/registry.py` | ~345 | 工具清单（名称、说明、JSON Schema）和 `dispatch()` |
| `agent_tools/horizon.py` | ~395 | A：用线段测倾斜、旋转、残余倾斜检查 |
| `agent_tools/composition.py` | ~675 | B：人脸、显著性、显著物体、候选裁剪、构图评分 |
| `agent_tools/color.py` | ~990 | C：测量、色彩流水线、富士 LUT、样片追色、人脸色偏 |
| `agent_tools/styles.py` | ~205 | 风格记忆：11 种样片风格和风格判断 |
| `agent_tools/people.py` | ~590 | D：MediaPipe 蒙版、整体补光、磨皮、瘦脸瘦身、人像检查 |
| `agent_tools/face.py` | ~485 | 仅作函数库（人脸子 agent 已删除）：像素蛋糕式五官工具 |
| `agent_tools/io_utils.py` | ~77 | 读 RAW（NEF，用 rawpy）、TIFF、JPEG；写 8/16 位 |
| `evals/harness.py` | ~245 | 在一组照片上跑 agent，并给每个阶段打分 |
| `evals/ground_truth.py` | ~57 | 用 SIFT 匹配，从你的 Lightroom 导出反推你的旋转和裁剪 |
| `tests/` | ~520 | 63 个 pytest 测试 |
| `samples/` | | 样片库和 `SOURCES.md`（来源与版权） |
| `docs/` | | 构图规则、像素蛋糕研究、本报告 |

## 4. Agent 循环怎么工作（`agent.py`）

这是所有 agent 的核心。下面是 `LevelingAgent.run()` 的简化版：

```python
messages = [系统提示词, 用户(指令 + 照片)]
for step in range(max_steps):                      # 默认 36 步
    reply = model(messages, tools=TOOLS)           # 模型可以回复"调用工具"
    messages.append(reply)
    if not reply.tool_calls:                       # 不再调用工具 = 它认为完成了
        final = parse_json(reply.content)
        break
    for call in reply.tool_calls:
        result = dispatch(call.name, call.arguments)   # 执行 Python 工具
        messages.append(tool_message(result))           # JSON 结果交回模型
    messages.append(user(这些结果对应的图片))              # 让它"看到"叠加图、预览图
```

有六个细节值得理解，每一个都修过一个真实的失败：

1. **工具用 JSON Schema 描述**（`registry.py`）。模型只能通过名称、说明和参数认识工具，所以说明文字本身
   就是程序的一部分。"只在用户要求风格时使用"这样的规则，既写在提示词里，也写在工具说明里。
2. **图片跟在工具结果后面发送。** 工具消息只能带文字，所以叠加图、对比图、预览图放在随后的用户消息里。
   不让它看图，它选裁剪就是盲选。
3. **路径固定**（`_call_tool`）。每个工具读写哪个文件由 agent 代码决定，不由模型决定：旋转永远从原图开始，
   色彩永远基于最新的裁剪，人像永远基于色彩结果。这修好了早期两个 bug：旋转叠加，以及模型把文件写得到处都是。
4. **状态保存在代码里**，不在模型的记忆里：实际旋转角度、有效区域、裁剪框、色彩参数。最终结果报告的是像素上
   真正发生的事，而不是模型声称的事。有一次模型说裁剪"保留了兔子"，其实把兔子切成了两半。要信状态。
5. **提醒（nudge）。** 有些模型会写一篇教程（"首先打开 Photoshop……"）而不调用工具。循环发现没用工具时，
   最多发 2 次提醒。
6. **统计用量。** 一张照片完整跑 A–D 大约输入 35 万 token，用 hy4-flash 约 ¥0.09。

## 5. 四个阶段

### A. 水平矫正（`horizon.py`）
- `estimate_tilt` 用 LSD 找直线，分别拟合近水平线和近竖直线（竖线用梯形透视模型），返回每组候选的离散度和
  透视斜率，并给出一张叠加图。
- **由模型选择信哪组线**：海平线、建筑边缘可以信；山坡、树枝、窗帘不能信。这是代码做不好的判断。
- `apply_rotation` 旋转原图，并返回不含空白角的最大矩形；`check_level` 测残余倾斜。
- 用你自己的 Lightroom 修图来评分：`ground_truth.py` 用 SIFT 把 RAW 和你的导出对齐，反推出你转的角度。
  误差小于 0.5° 算通过。

### B. 构图裁剪（`composition.py`）
- `propose_crops` 按经典规则（`docs/composition_rules.md`）给大量候选框打分：三分法、眼睛在上三分线、
  头顶空间、视线方向留白、不在关节处裁切（按 8 头身比例）、地平线在三分线、按纹理决定天地比例、边缘干净。
- **显著物体**（新）：发光的灯、雕像、招牌，用频率调谐显著性找出（先用人像分割去掉人）。裁剪要么完整保留，
  要么完全不要，绝不切一半。
- 模型从对比图里选；`check_composition` 给选择打分。

### C. 全局色彩（`color.py`）
- 流水线顺序：白平衡 → 曝光 → 高光/阴影（引导滤波求局部基底，避免光晕）→ 色调曲线 → 8 色 HSL →
  自然饱和度/饱和度 → 胶片 LUT → 分离色调 → 颗粒。
- 风格：手写配方 + 8 种富士胶片模拟 LUT（HaldCLUT）。
- **样片追色**：`match_reference` 拟合滑块数值，让照片接近样片的统计特征：按档位（stop）匹配的亮度分位、
  中间调 a/b、色度、每个颜色的 HSL。匹配有阻尼和上限，所以夜景不会被强行拉成白天。
- **风格记忆**：11 种样片风格；其中 9 张是你选的，2 张来自 Wikimedia Commons。`suggest_styles` 测量
  场景线索（夜晚、人物、脸的大小、天空、红叶、花、雪、建筑、亮度）给风格排序，`apply_style` 应用其中一种。
  **只在用户要求风格时使用。**
- **人脸色温上限**（新）：每个色彩工具都会报告 `face_shift`，即主脸皮肤颜色移动了多少（用白平衡滑块单位
  表示）。上限是色温 ±20、色调 ±20。超出时，工具会在这张脸自己的像素上算出 `skin_correction`，由阶段 D
  通过皮肤蒙版应用。这样场景保留风格，皮肤仍然像皮肤。
- 检查项：高光溢出、暗部死黑、脸的亮度、肤色色相/饱和度、"蓝天保持蓝色"、中性色偏、人脸色偏。

### D. 人像（`people.py`）
- `detect_people`：YuNet 人脸、MediaPipe 478 点人脸关键点、人体姿态、6 类分割（头发、身体皮肤、脸部皮肤、
  衣服……）。蒙版用引导滤波贴合边缘。
- `preview_retouch` / `apply_retouch`：给整个人补光（`subject_light`，不是只提亮脸）、保留毛孔的频率分离
  磨皮、匀肤、用平滑形变场做轻微瘦脸瘦身，以及按区域的局部调整（包括 C 阶段给出的肤色校正）。
- 检查项：人物周围无光晕、脸没有单独提亮（比身体多亮不超过 0.8 EV）、皮肤纹理保留、瘦身轻微、背景没变形。

## 6. 评估（`evals/harness.py`）

`python -m evals.harness --agent hy --folder path\to\photos` 在每张照片上跑 agent 并打分：
- **水平**：|agent 角度 − 你的 Lightroom 角度| < 0.5°。
- **裁剪**：`check_composition`，没有硬性违规，且 ≥ 80% 的规则通过。
- **色彩**：对最终图片做 `check_color`。
- **人像**：重新测量人像检查项。

`--rescore` 用当前规则重新给保存过的结果打分，不花 API 费用。通过与否由 harness 决定，不由模型决定。
让模型给自己打分会引发 *reward hacking*：它会学会说"效果很好"。这也是白板上的研究问题：
独立的评审（critic）是否比自我评估更好？

## 7. 踩过的坑，以及每个坑教会了什么

| 问题 | 原因 | 修复 | 经验 |
|---|---|---|---|
| 旋转越转越歪 | 模型对已旋转的文件再旋转 | 永远从原图旋转；固定路径 | 状态放在代码里 |
| 某视觉模型从不调用工具 | 它在写教程 | 提醒；harness 记为失败 | 检查行为，不只检查输出 |
| 人物周围一圈光 | 蒙版边缘被模糊 | 引导滤波蒙版 | 把瑕疵变成可测量的检查 |
| 天空变紫 | 加暖白平衡 + 压高光 | 天空检查 + 提示词规则 | 把审美变成可测量的规则 |
| 只提亮脸像贴上去的 | 只对脸补光 | 给整个人补光 | 把摄影师的做法写进规则 |
| 追色把夜景洗白 | 匹配了绝对亮度 | 按档位匹配并设上限 | 统计特征 ≠ 画面内容 |
| 裁剪切掉半只兔子 | 没有"物体"的概念 | 显著物体规则，硬性失败 | 模型的说法需要验证 |
| 套风格后脸变冷/变绿 | 白平衡作用到所有人 | 人脸色偏上限 + 蒙版肤色校正 | 先保护观众最先看的地方 |
| 人脸子 agent | 复杂度高，收益小 | 删除 | agent 不是越多越好 |

## 8. 怎么运行

```powershell
.\.venv\Scripts\python.exe agent.py path\to\photo.NEF --instruction "修一下，夜景人像风格"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evals.harness --agent hy --folder path\to\photos
```
配置在 `.env`（`HY_API_KEY`、`HY_BASE_URL`、`HY_MODEL=hy4-flash`）。每次运行都会写入
`runs/<照片>_<时间>/`，里面有 `trace.json`（模型的每条消息和每次工具调用）、对比图和输出文件。
读 `trace.json` 是理解 agent 实际做了什么的最好方式。

## 9. 已知限制和下一步

- 人脸检测找不到大幅仰头的脸（DSC_6905），这时皮肤校正无法执行，agent 应改为降低风格强度。
- 你的两张样片（蓝调、街头夜景）修得很亮，风格判断会把它们当成白天。对真实的夜景 RAW 没问题，因为 RAW 很暗。
- agent 类名还叫 `LevelingAgent`（第一天的名字），现在它已经跑全部四个阶段。
- 白板上的"导出 XMP 给 Lightroom"还没做（目前输出 TIFF + 参数 JSON）。
- 想法：用独立的评审模型做主观评估；做一个从你自己的 Lightroom 修图中学习的记忆。
