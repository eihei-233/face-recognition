# 向量库人脸识别系统

基于 **ChromaDB** 向量数据库 + **InsightFace** 人脸检测与特征提取 + **Ollama DeepSeek-R1:8B** LLM 分析的人脸识别系统，支持中文身份标签。

---

## 目录

- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [模块说明](#模块说明)
- [使用方式](#使用方式)
  - [1. 注册人脸](#1-注册人脸)
  - [2. 启动网页端](#2-启动网页端)
  - [3. 命令行识别](#3-命令行识别)
- [运行截图示例](#运行截图示例)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 系统架构

```
输入图像
  │
  ▼
┌─────────────────────┐
│  face_detector.py   │  SCRFD (det_10g.onnx)
│  人脸检测 + 5点对齐  │  输出: bbox + 512维特征
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ face_embedding.py   │  InsightFace (w600k_r50.onnx)
│  提取512维特征向量   │  L2归一化
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   vector_db.py      │  ChromaDB (余弦相似度)
│  向量检索 Top-K      │  持久化存储
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  llm_analyzer.py    │  Ollama DeepSeek-R1:8B
│  LLM / 规则分析      │  支持回退
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   visualize.py      │  PIL 渲染
│  绘制人脸框+身份标签  │  支持中文
└─────────────────────┘
```

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| ChromaDB | 1.5.9+ | 向量数据库 |
| ONNX Runtime | 1.16+ | 模型推理 |
| OpenCV | 4.8+ | 图像处理 |
| NumPy | 1.24+ | 数值计算 |
| Pillow | 10.0+ | 中文文本渲染 |
| Streamlit | 1.57+ | 网页界面（可选） |
| Ollama | 最新版 | LLM 分析（可选） |

## 快速开始

### 步骤 1：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 2：下载模型

系统使用两类模型：

**① 人脸检测与特征提取模型（ONNX）**

`det_10g.onnx`、`w600k_r50.onnx` 等 InsightFace 模型应存放在 `models/` 目录下。若未下载，运行以下命令自动下载：

```bash
python -c "
import urllib.request, zipfile, os
url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(url, 'models/buffalo_l.zip')
with zipfile.ZipFile('models/buffalo_l.zip', 'r') as zf:
    zf.extractall('models/')
os.remove('models/buffalo_l.zip')
print('Model downloaded.')
"
```

**② LLM 分析模型（Ollama）**

`deepseek-r1:8b` 通过 Ollama 下载和管理，用于分析识别结果。安装 [Ollama](https://ollama.com/) 后运行：

```bash
ollama pull deepseek-r1:8b
```

> **注意**：LLM 仅用于结果分析，非识别必需。若 Ollama 不可用，系统自动回退到内置规则分析，不影响人脸识别核心功能。

### 步骤 3：配置数据路径

编辑 `config.py`，确认数据路径正确。系统会自动查找 `Desktop/celeba_100_identities_3reg_3test`，若数据在其他位置，手动修改 `DATA_DIR`。

### 步骤 4：注册人脸到向量库

```bash
python main.py register --reset
```

这将读取 `register/` 目录下所有身份文件夹中的图像，提取人脸特征并存入 ChromaDB。

### 步骤 5：启动网页端

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，上传图像即可进行人脸识别。

---

## 项目结构

```
d:\face recognition\
├── config.py            # 配置文件（路径、阈值、模型参数）
├── face_detector.py     # 人脸检测模块（SCRFD ONNX 推理 + 对齐）
├── face_embedding.py    # 人脸特征提取模块（InsightFace ONNX）
├── vector_db.py         # 向量数据库模块（ChromaDB 增删查）
├── llm_analyzer.py      # LLM 分析模块（Ollama + 规则回退）
├── visualize.py         # 可视化模块（PIL 绘制人脸框/标签/序号）
├── register_faces.py    # 批量注册脚本
├── recognize.py         # 识别引擎（检测→嵌入→检索→分析）
├── main.py              # 命令行入口
├── app.py               # Streamlit 网页端
├── requirements.txt     # Python 依赖
├── README.md            # 本文件
├── models/              # ONNX 模型文件
│   ├── det_10g.onnx     # 人脸检测模型（SCRFD）
│   └── w600k_r50.onnx   # 人脸识别模型（512维特征）
└── chroma_db/           # ChromaDB 持久化数据（自动生成）
```

---

## 模块说明

### `config.py` — 配置文件
统一管理所有可配置参数：
- **路径**：数据目录、模型目录、数据库目录
- **检测**：置信度阈值 `FACE_DETECTION_CONFIDENCE = 0.35`
- **识别**：相似度阈值 `SIMILARITY_THRESHOLD = 0.30`，Top-K `= 5`
- **Ollama**：服务地址、模型名称、超时时间

### `face_detector.py` — 人脸检测
封装 InsightFace `det_10g.onnx`（SCRFD 架构）：
- 输入：图像文件路径（str）或 BGR numpy 数组
- 输出：`(image, face_infos)`，其中 `face_infos` 为列表，每项包含 `bbox`、`landmarks`（5点关键点）、`aligned_face`（112×112 对齐人脸）
- 支持 distance-based bbox 解码 + NMS 后处理

### `face_embedding.py` — 特征提取
封装 InsightFace `w600k_r50.onnx`：
- 输入：112×112 对齐人脸（BGR）
- 输出：512 维 L2 归一化特征向量
- 提供 `cosine_similarity()` 静态方法

### `vector_db.py` — 向量数据库
基于 ChromaDB 的持久化向量存储：
- `add_face(identity, image_name, embedding)` — 添加单张人脸
- `search(embedding, top_k=5)` — 余弦相似度检索
- `get_stats()` — 统计信息
- `reset()` — 清空重建

### `llm_analyzer.py` — LLM 分析
调用 Ollama DeepSeek-R1:8B 分析识别结果；若 LLM 不可用，自动回退到内置规则分析（输出置信度、区分度、判定结论）。

### `visualize.py` — 可视化
使用 PIL 在图像上绘制：
- **人脸框**：绿色（匹配成功）、红色（unknown）
- **标签**：身份名 + 相似度，中文使用微软雅黑/黑体字体
- **序号**：多人脸时左上角圆圈数字

### `app.py` — 网页端
Streamlit 应用，提供 3 个页面：
- 📸 **人脸识别**：上传最多 10 张图片，一键识别，展示标注结果
- 📋 **批量注册**：上传新人脸图像，保存到 `register/` 并入库
- 📊 **数据库统计**：查看已注册身份和人数

---

## 使用方式

### 1. 注册人脸

```bash
# 注册全部身份（清库重建）
python main.py register --reset

# 仅注册前 10 个身份（快速测试）
python main.py register --first 10

# 或直接调用
python register_faces.py --reset
```

注册结果示例：
```
处理图像总数: 360
成功注册:     359
未检测到人脸: 1
数据库人脸数: 359
数据库身份数: 120
耗时:         162 秒 (450 ms/张)
```

### 2. 启动网页端

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`：

- 上传图像（最多 10 张）→ 点击 **"开始人脸识别"** → 查看标注结果
- 侧边栏显示数据库统计、Ollama 连接状态、图例

### 3. 命令行识别

```bash
# 识别单张图像
python main.py search photo.jpg --identity identity_00070

# 查看数据库统计
python main.py stats

# 清空数据库
python main.py reset
```

---

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SIMILARITY_THRESHOLD` | 0.30 | 余弦相似度阈值，≥此值判定为同一人 |
| `FACE_DETECTION_CONFIDENCE` | 0.35 | 人脸检测置信度阈值 |
| `TOP_K_RESULTS` | 5 | 返回前 K 个匹配结果 |
| `OLLAMA_MODEL` | deepseek-r1:8b | LLM 模型名称 |

### Ollama 配置

若 Windows 用户名含中文导致 Ollama 无法加载模型：

1. 设置环境变量：`setx OLLAMA_MODELS D:\ollama_models`
2. 复制模型：`xcopy %USERPROFILE%\.ollama\models D:\ollama_models /E /I`
3. 重启 Ollama（系统托盘退出 → 重新启动）

LLM 不可用时系统自动回退到内置规则分析，不影响人脸识别核心功能。

---

## 常见问题

**Q: 启动报错 "Detection model not found"**  
A: 模型文件未下载，运行快速开始中的模型下载命令。

**Q: 人脸框不显示中文**  
A: 系统会自动查找 `C:/Windows/Fonts/msyh.ttc`（微软雅黑）等中文字体，确保系统安装了中文字体。

**Q: ChromaDB 报错 "Device or resource busy"**  
A: 数据库正被另一个进程（如 Streamlit）占用。先关闭 Streamlit（`Ctrl+C`），再执行操作。

**Q: 识别准确率低**  
A: 调整 `config.py` 中的 `SIMILARITY_THRESHOLD`，降低阈值可放宽匹配条件（当前默认 0.30）。
