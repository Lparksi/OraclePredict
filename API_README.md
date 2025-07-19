# Oracle Predict API 使用说明

这是一个基于 Flask 的图像推理 API 服务，用于处理图像的目标检测和分类任务。

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -e .
```

### 2. 启动服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动。

### 3. 测试 API

```bash
python test_api.py
```

## API 接口文档

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。

**响应示例:**
```json
{
  "status": "healthy",
  "message": "API服务运行正常"
}
```

### 2. 单张图片推理

**POST** `/predict`

对单张图片进行推理。

**请求方式1: 文件上传**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

**请求方式2: Base64 JSON**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "推理完成",
  "results": [
    {
      "bbox": [100, 150, 300, 400],
      "class": 1,
      "confidence": 1.0
    }
  ],
  "count": 1
}
```

### 3. 批量图片推理

**POST** `/predict/batch`

对多张图片进行批量推理。

**请求示例:**
```json
{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
  ]
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "批量推理完成",
  "batch_results": [
    {
      "index": 0,
      "success": true,
      "results": [
        {
          "bbox": [100, 150, 300, 400],
          "class": 1,
          "confidence": 1.0
        }
      ],
      "count": 1
    }
  ],
  "total_processed": 1
}
```

### 4. 获取模型信息

**GET** `/model/info`

获取当前模型的配置信息。

**响应示例:**
```json
{
  "success": true,
  "model_info": {
    "detection_model": "inferences/models/detection-fp32.onnx",
    "classification_model": "inferences/models/classification-fp32.onnx",
    "providers": ["CPUExecutionProvider"],
    "precision": "fp32",
    "conf_threshold": 0.5,
    "iou_threshold": 0.4
  }
}
```

## 支持的图片格式

- JPEG/JPG
- PNG
- BMP
- WEBP
- 其他 PIL 支持的格式

## 输入要求

- 图片可以是任意尺寸，会自动进行预处理
- 支持 RGB 和 RGBA 格式
- Base64 编码需要包含数据头 (如 `data:image/jpeg;base64,`)

## 输出说明

### bbox (边界框)
包含4个整数的数组: `[x1, y1, x2, y2]`
- `x1, y1`: 左上角坐标
- `x2, y2`: 右下角坐标

### class (类别)
整数值，表示检测到的对象类别ID

### confidence (置信度)
浮点数，表示检测结果的置信度 (当前固定为1.0)

## 错误处理

所有错误响应都遵循以下格式:
```json
{
  "error": "错误类型",
  "message": "详细错误信息"
}
```

常见错误:
- `400`: 请求参数错误
- `500`: 服务器内部错误 (通常是模型推理失败)

## 配置说明

模型配置文件位于 `inferences/configs/config.toml`，可以调整以下参数:
- `conf-threshold`: 置信度阈值
- `iou-threshold`: IoU阈值
- `precision`: 模型精度 (fp16/fp32)
- `providers`: ONNX Runtime 执行提供器

## 开发调试

启动开发模式:
```bash
python app.py
```

服务会在调试模式下运行，自动重载代码变更。

## 生产部署

推荐使用 Gunicorn 进行生产部署:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 注意事项

1. 确保模型文件存在于 `inferences/models/` 目录
2. 大图片会占用较多内存和处理时间
3. 批量推理时建议限制同时处理的图片数量
4. 生产环境建议添加认证和限流机制
