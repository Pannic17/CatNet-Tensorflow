# CatNet-Tensorflow

CatNet 的 TensorFlow 分类训练与模型实验仓库。主流程基于 MobileNetV2，识别猫的 7 种毛色／花纹；同时包含 MobileNetV3 实验、猫脸预处理、颜色提取、TensorFlow checkpoint / SavedModel 和多个 ONNX 模型。

应用端与其他模块入口：[CatNet-Unity](https://github.com/Pannic17/CatNet-Unity)。猫脸数据准备见 [CatNet-Face-Cut](https://github.com/Pannic17/CatNet-Face-Cut)，PyTorch 对照实验见 [CatNet-Test-Pytorch](https://github.com/Pannic17/CatNet-Test-Pytorch)。

## 主要文件

| 文件／目录 | 用途 |
| --- | --- |
| `model_v2.py`、`model_v3.py` | TensorFlow MobileNetV2 / MobileNetV3 网络定义 |
| `train_mobilenet_v2.py` | 7 类 MobileNetV2 训练，保存 checkpoint 和 SavedModel |
| `train_mobilenet_v3.py`、`utils.py` | MobileNetV3 Small 训练和数据集生成 |
| `trainGPU_mobilenet_v2.py` | 保留的 GPU 训练实验，仍含花卉数据路径与 5 类配置 |
| `split_data.py` | 按类别将猫脸图片划分为 train / val |
| `predict.py` | 猫脸裁剪、颜色聚类与 TensorFlow checkpoint 预测 |
| `predict_onnx.py` | ONNX Runtime CPU 推理实验 |
| `cv_process.py` | Haar 猫脸检测、裁剪、颜色转换及主色提取 |
| `read_ckpt.py`、`trans_v3_weights.py` | 外部预训练权重转换实验 |
| `ckpt2pb.py`、`load_saved_model.py` | 历史冻结图和 SavedModel 加载实验 |
| `onnx_visualization.py` | ONNX 图结构可视化工具，依赖 pydot / Graphviz |
| `save_weights/`、`CMN_*/`、`models_tf/`、`*.onnx` | 已保存的不同实验版本模型 |
| `test/`、`class_indices.json`、`Result.xlsx` | 示例图片、标签映射和实验结果文件 |

类别顺序：`Bicolor`、`Calico`、`Colorpoint`、`Mix`、`Orange`、`Solid`、`Tabby`。这些是外观类别，而非猫品种。

## 环境

主流程使用 Python、TensorFlow、NumPy、Pillow、Matplotlib、OpenCV 和 tqdm；ONNX 预测额外使用 onnx、onnxruntime。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install tensorflow numpy pillow matplotlib opencv-python tqdm onnx onnxruntime
```

仓库未锁定依赖版本。代码使用历史 Keras checkpoint / SavedModel API（包括 `save_format='tf'`、`reset_states()`），需选择兼容环境或调整 API；以上安装命令不代表已验证的版本组合。

## 数据与训练

从仓库根目录运行脚本。数据集未包含在仓库内；自行准备每类猫脸图片：

```text
data_set/cat_data/
  cat_face/<类别名>/*.jpg
  train/<类别名>/*.jpg
  val/<类别名>/*.jpg
```

`train` 和 `val` 均应包含相同的 7 个类别目录。如果尚未划分，可执行：

```powershell
python split_data.py
```

它从 `data_set/cat_data/cat_face` 按 90% / 10% 划分，随机种子为 0，**会先删除已有 train / val 目录再重建**。

确保 `save_weights` 目录存在后，执行主训练入口：

```powershell
python train_mobilenet_v2.py
```

默认输入为 224 × 224 RGB，像素按 `(pixel / 255 - 0.5) * 2` 归一化到 `[-1, 1]`，batch size 为 16，训练 24 轮。预训练权重加载代码当前被注释，因此默认从随机初始化开始训练。训练会重写 `class_indices.json`，在验证准确率提升时保存：

- `save_weights/CMN_b16e24_onnx_v6.ckpt` 及其数据文件。
- `save_weights/CMN_b16e24_onnx_v6/` SavedModel。

MobileNetV3 入口默认 batch size 为 16、48 轮、7 类；运行前需检查其预训练权重路径。历史 GPU 入口仍是 5 类花卉配置，不能直接视作猫分类训练入口。

## 预测

1. 修改 `cv_process.py` 中硬编码的 `H:` 盘级联 XML 路径，可使用 Face-Cut 仓库的 `haarcascade_frontalcatface_extended.xml`。
2. 将 JPG 放入 `test/`，从仓库根目录运行以下任一脚本；按提示输入不含 `.jpg` 的文件名。

```powershell
python predict.py
python predict_onnx.py
```

`predict.py` 默认加载 `save_weights/CMN_b16e24_onnx_v5.ckpt`，而主训练入口保存的是 v6；使用新训练输出时需修改权重路径并核对分类头结构。`predict_onnx.py` 默认加载 `model11v6.onnx`，输入为归一化后的 `(1, 224, 224, 3)`，输出类别索引和最大值。该脚本虽然先检测猫脸，目前传给 `main()` 的是原图 `plt_img`，不是裁剪后的 `cat`。

## 接入 Unity

将验证过的 ONNX 导入主仓库使用的 Barracuda 环境，检查 RGB、NHWC、224 × 224、`[-1, 1]` 预处理、标签顺序、像素方向与 `softmax` 输出名称。仓库保存了多种实验版本，不能假定所有模型都具有相同接口。主训练脚本直接保存 TensorFlow 格式，未提供统一的 TensorFlow → ONNX 导出入口。

数据集及部分外部预训练权重未提供；模型文件名不代表经过统一兼容性验证。本文基于源码与文件结构整理，未重新训练、运行推理或验证 Unity 模型导入。
