# VEX AI VAIRC Push Back 2025–2026 YOLO11 模型

这个仓库基于 [MagikidAI/VEX_AI_VAIRC_Push_Back_2025_2026](https://github.com/MagikidAI/VEX_AI_VAIRC_Push_Back_2025_2026) 的数据集，
训练并发布了适用于 **VEX AI / VAIRC Push Back 2025–2026 赛季** 的 YOLO11 目标检测模型。

主要目标：在真实比赛场地上识别
- 红 / 蓝方块（场内 & 装载区 & 得分区）
- center_goal / long_goal
- load_station
- red/blue_park 等关键场地元素

---

## 1. 环境准备

建议使用 Conda，新建一个独立环境：

```bash
conda create -n vex-yolo python=3.10 -y
conda activate vex-yolo
```

安装带 CUDA 的 PyTorch（如果有 NVIDIA GPU）：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

安装 Ultralytics YOLO：

```bash
pip install ultralytics
```

---

## 2. 数据集说明

数据集结构（来自原仓库）：

```text
VEX_AI_VAIRC_Push_Back_2025_2026/
  ├─ train/
  │   ├─ images/
  │   └─ labels/
  ├─ valid/
  │   ├─ images/
  │   └─ labels/
  ├─ test/
  │   ├─ images/
  │   └─ labels/
  ├─ data.yaml
  └─ classes.txt
```

类别列表：

```text
0  blue
1  blue_goal
2  red
3  red_goal
4  blue_load
5  red_load
6  load_station
7  center_goal
8  long_goal
9  red_park
10 blue_park
```

> 使用时可以在后处理里将  
> `red / red_goal / red_load / load_station` 视为“红方区域/红块”，  
> `blue / blue_goal / blue_load` 视为“蓝方区域/蓝块”。

---

## 3. 训练配置

本仓库提供的模型使用 YOLO11n 训练，基础命令示例：

```bash
yolo train   model=yolo11n.pt   data=data.yaml   epochs=100   imgsz=640   batch=16   project=pushback_25_26_gpu   name=y11n_magikid_gpu
```

训练结果目录（示例）：

```text
pushback_25_26_gpu/
  └─ y11n_magikid_gpu/
      ├─ weights/
      │   ├─ best.pt
      │   └─ last.pt
      └─ results.png
```

请根据你自己的路径更新上面的目录名。

---

## 4. 使用训练好的模型进行推理

### 4.1 在测试集上预测

```bash
yolo detect predict   model=pushback_25_26_gpu/y11n_magikid_gpu/weights/best.pt   source=test/images   imgsz=640   conf=0.5
```

结果将保存在 `runs/detect/predict/`。

### 4.2 在自定义图片上预测

将图片放到 `my_test/` 目录，例如 `my_test/image.png`：

```bash
yolo detect predict   model=pushback_25_26_gpu/y11n_magikid_gpu/weights/best.pt   source=my_test/image.png   imgsz=640   conf=0.5
```

---

## 5. 导出模型（ONNX / TensorRT）

为方便部署到 Jetson 等设备，可以导出 ONNX 或 TensorRT 引擎：

```bash
yolo export   model=pushback_25_26_gpu/y11n_magikid_gpu/weights/best.pt   format=onnx

yolo export   model=pushback_25_26_gpu/y11n_magikid_gpu/weights/best.pt   format=engine
```

导出文件会保存在同一个 `weights/` 目录中，例如：

```text
best.onnx
best.engine
```

---

## 6. 对机器人代码的使用建议

- 视觉检测只负责输出框 + 类别 + 置信度；
- 在机器人侧可以按策略合并类别，例如：

  - `{"red", "red_goal", "red_load", "load_station"}` → 统一视为 “red”
  - `{"blue", "blue_goal", "blue_load"}` → 统一视为 “blue”

- 将框中心转换成相对坐标（像素 / 角度），供底盘对准、导航使用。

---

## 7. 致谢

- 数据集来源：MagikidAI 为 VEX AI VAIRC Push Back 2025–2026 制作的数据集  
- YOLO11 框架：Ultralytics

如果你在使用或部署过程中遇到问题，欢迎提 Issue 或自行 Fork 修改。
