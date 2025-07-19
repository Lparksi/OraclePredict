# 🔮 Oracle Predict - 智能图像检测与分类系统

Oracle Predict 是一个基于深度学习的智能图像检测与分类系统，提供了完整的前后端解决方案。系统采用 ONNX 模型进行推理，支持目标检测和图像分类任务，并提供了现代化的 Web 界面进行交互。

## ✨ 特性

### 🎯 核心功能
- **目标检测**: 检测图像中的多个目标对象
- **图像分类**: 对检测到的目标进行分类
- **批量处理**: 支持多张图片同时处理
- **实时可视化**: 检测结果叠加显示在原图上
- **交互式标注**: 鼠标悬停高亮对应检测框

### 🖥️ 用户界面
- **现代化设计**: 基于 Vue 3 的响应式界面
- **拖拽上传**: 支持拖拽文件上传
- **实时预览**: 图片上传后立即预览
- **结果可视化**: Canvas 绘制检测框和标签
- **结果下载**: 可下载带标注的结果图片

### 🚀 技术特性
- **高性能推理**: 基于 ONNX Runtime 的模型推理
- **RESTful API**: 完整的 Flask REST API
- **跨平台支持**: 支持 Windows、Linux、macOS
- **响应式设计**: 适配桌面端、平板、移动端

## 🏗️ 系统架构

```
Oracle Predict
├── 后端服务 (Flask + ONNX Runtime)
│   ├── 图像预处理
│   ├── 模型推理引擎
│   ├── 结果后处理
│   └── RESTful API
├── 前端界面 (Vue 3 + TypeScript)
│   ├── 图片上传组件
│   ├── 结果可视化
│   ├── 批量处理界面
│   └── 模型信息展示
└── AI 模型 (ONNX)
    ├── 检测模型 (detection.onnx)
    └── 分类模型 (classification.onnx)
```

## 📁 项目结构

```
OraclePredict/
├── app.py                      # Flask API 服务端
├── pyproject.toml              # Python 项目配置
├── test_api.py                 # API 测试脚本
├── API_README.md               # API 使用文档
├── inferences/                 # 推理引擎模块
│   ├── engines.py              # 核心推理引擎
│   ├── configs/
│   │   └── config.toml         # 模型配置文件
│   └── models/                 # ONNX 模型文件
│       ├── detection-fp16.onnx # 检测模型 (FP16)
│       ├── detection-fp32.onnx # 检测模型 (FP32)
│       ├── classification-fp16.onnx # 分类模型 (FP16)
│       └── classification-fp32.onnx # 分类模型 (FP32)
└── client/                     # 前端应用
    └── OBC-client/             # Vue 3 项目
        ├── src/
        │   ├── App.vue         # 主应用组件
        │   ├── main.ts         # 应用入口
        │   └── style.css       # 全局样式
        ├── package.json        # 前端依赖
        ├── FRONTEND_GUIDE.md   # 前端使用指南
        └── start.bat           # Windows 启动脚本
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.13+
- **Node.js**: 16.0+
- **操作系统**: Windows / Linux / macOS

### 1. 克隆项目

```bash
git clone https://github.com/Lparksi/OraclePredict.git
cd OraclePredict
```

### 2. 后端设置

```bash
# 使用 uv 安装依赖 (推荐)
uv sync

# 或使用 pip 安装
pip install -e .
```

### 3. 启动后端服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动

### 4. 前端设置

```bash
cd client/OBC-client
npm install
npm run dev
```

前端将在 `http://localhost:5173` 启动

## 📊 类别编码

本项目使用 [OBC306 编码](https://jgw.aynu.edu.cn/home/down/detail/index.html?sysid=16) 进行目标分类。

## 🔧 配置说明

### 模型配置 (`inferences/configs/config.toml`)

```toml
# 模型路径配置
detection-model-path = "inferences/models/detection-fp32.onnx"
classification-model-path = "inferences/models/classification-fp32.onnx"

# ONNX Runtime 配置
providers = ["CPUExecutionProvider"]  # 可选: CUDAExecutionProvider

# 推理参数
precision = "fp32"          # fp16 或 fp32
conf-threshold = 0.5        # 置信度阈值
iou-threshold = 0.4         # NMS IoU 阈值
```

### API 端点

| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/predict` | 单张图片预测 |
| POST | `/predict/batch` | 批量图片预测 |
| GET | `/model/info` | 获取模型信息 |

## 💻 使用示例

### API 调用示例

**健康检查**
```bash
curl http://localhost:5000/health
```

**单张图片预测**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

**批量预测**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"images":["data:image/jpeg;base64,..."]}' \
     http://localhost:5000/predict/batch
```

### Python 调用示例

```python
import requests

# 单张图片预测
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
    print(f"检测到 {result['count']} 个对象")
```

## 🎨 前端功能

### 主要界面

1. **单张预测**
   - 拖拽或点击上传图片
   - 实时预览和结果可视化
   - 交互式边界框标注
   - 检测详情展示

2. **批量预测**
   - 多文件选择上传
   - 批量处理进度显示
   - 统计结果汇总

3. **模型信息**
   - 当前模型配置查看
   - 系统状态监控

### 可视化特性

- **多彩标注**: 使用10种颜色区分不同检测对象
- **交互高亮**: 鼠标悬停时高亮对应边界框
- **标注切换**: 可显示/隐藏标注信息
- **结果下载**: 保存带标注的结果图片

## 🧪 测试

### API 测试

```bash
python test_api.py
```

### 手动测试

1. 启动后端服务
2. 启动前端应用
3. 上传测试图片
4. 查看检测结果

## 📈 性能优化

### 推理优化
- **模型量化**: 支持 FP16/FP32 精度选择
- **批量处理**: 支持多图片并发处理
- **内存管理**: 智能内存释放机制

### 前端优化
- **图片缩放**: 大图片自动缩放显示
- **Canvas 渲染**: 高性能图像标注
- **异步加载**: 非阻塞用户界面

## 🛠️ 开发指南

### 后端开发

```bash
# 开发模式启动
python app.py

# 添加新的推理功能
# 编辑 inferences/engines.py

# 添加新的API端点
# 编辑 app.py
```

### 前端开发

```bash
cd client/OBC-client

# 开发模式
npm run dev

# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

## 📝 部署

### Docker 部署 (推荐)

```dockerfile
# 后端 Dockerfile
FROM python:3.13-slim
COPY . /app
WORKDIR /app
RUN pip install -e .
EXPOSE 5000
CMD ["python", "app.py"]
```

### 生产环境部署

**后端 (Gunicorn)**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**前端 (Nginx)**
```bash
npm run build
# 将 dist/ 目录部署到 Nginx
```

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🐛 问题报告

如果你发现了 bug 或有功能建议，请在 [Issues](https://github.com/Lparksi/OraclePredict/issues) 页面提交。

## 📞 联系方式

- **项目维护者**: Lparksi
- **项目地址**: https://github.com/Lparksi/OraclePredict
- **问题反馈**: [GitHub Issues](https://github.com/Lparksi/OraclePredict/issues)

## 🙏 致谢

- ONNX Runtime 团队提供的高性能推理框架
- Vue.js 团队提供的优秀前端框架
- Flask 团队提供的轻量级 Web 框架
- 所有贡献者的支持和反馈

---

⭐ 如果这个项目对你有帮助，请给个 Star！