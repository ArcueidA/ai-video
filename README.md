# Wan 2.2 14B Image-to-Video (720P 60fps 10s Dance Workflow)

基于 **Wan 2.2 14B (Q4_K_M GGUF)** + **LightX2V 蒸馏加速 LoRA** + **Dancing LoRA (`b3ll13-d8nc3r`)** + **Practical-RIFE 4x 补帧** 的 Google Colab 图生视频工作流，支持一键生成 **720×1280 · 60fps · 10秒** 竖屏高清跳舞视频。

## 项目结构

```text
ai-video/
├── assets/
│   └── character.jpg                       # 输入人物参考图 (576×1024)
├── notebooks/
│   └── Wan22_720p_60fps_Dance.ipynb        # 一键运行 Colab Notebook (支持自动备份 Google Drive)
├── outputs/
│   └── wan22_dancing_720p_60fps_10s.mp4    # 生成示例成片 (720×1280, 60fps, 10.0s)
├── scripts/
│   └── generate_720p_60fps.py              # 核心推理 + RIFE 4x 补帧 + ffmpeg 封装脚本
├── .gitignore
└── README.md
```

## 核心参数与配置

- **硬件要求**：Google Colab `L4 GPU`（22.5 GB VRAM + High-RAM）或 `A100 GPU`
- **基础模型**：Wan 2.2 I2V 14B (`high_noise` + `low_noise`, `Q4_K_M` GGUF)
- **加速方案**：LightX2V 蒸馏 LoRA（6~8 步出片，大幅缩短扩散采样时间）
- **运动增强**：`b3ll13-d8nc3r` 舞蹈 LoRA
- **插帧平滑**：Practical-RIFE `4x` 插帧（`16 fps -> 60 fps`）
- **成片规格**：`720 × 1280`（9:16 竖屏）、`60 fps`、`10.0s`、H.264 (`yuv420p`, CRF 16)

## 快速使用（Google Colab）

1. 打开 [Google Colab](https://colab.research.google.com/)，选择 **文件 ➔ 上传笔记本**，上传 `notebooks/Wan22_720p_60fps_Dance.ipynb`。
2. 在 **代码执行程序 ➔ 更改运行时类型** 中选择 **`L4 GPU`**（或 `A100 GPU`）。
3. 点击 **代码执行程序 ➔ 全部运行**，生成完成后视频会自动保存至你的 Google 云端硬盘并触发本地下载。
