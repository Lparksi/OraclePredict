# YOLO 分类模型 Flask API

## 简介

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 分类模型，提供了一个 Flask Web API 服务。用户可通过 HTTP POST 请求提交图片路径，获取图片的 Top-5 分类预测结果。

---

## 环境依赖

- Python 3.8+
- Flask
- torch
- ultralytics
- pillow

安装依赖（建议使用虚拟环境）：

```bash
pip install flask torch ultralytics pillow
```
或使用 `uv` 包管理器
```bash
uv sync
```

---

## 文件结构

```
.
├── predict.py                # 主API脚本
├── best.pt                   # 你的YOLO分类模型权重文件
├── class_indices.json        # 类别索引映射文件
├── ID_to_chinese.json        # 类别中文名映射文件
```

---

## 启动服务

在命令行中运行：

```bash
python predict.py
```

默认服务监听在 `0.0.0.0:5000`。

---

## API 使用方法

### 1. 请求方式

- URL: `http://<服务器IP>:5000/predict`
- 方法: `POST`
- Content-Type: `application/x-www-form-urlencoded`
- 参数:  
  - `file_path`：图片的绝对路径（服务器本地路径）

### 2. 示例请求

```bash
curl -X POST http://127.0.0.1:5000/predict -d "file_path=E:\Dataset\HUST-OBC\output\test\0045\H_0045_60C27_15.png"
```

### 3. 返回结果

成功时返回：

```json
{
    "success": true,
    "image_path": "E:\\Dataset\\HUST-OBC\\output\\test\\0045\\H_0045_60C27_15.png",
    "predictions": [
        {
            "id": "0",
            "confidence": 0.98765,
            "chinese_char": "类别中文名"
        },
        ...
    ],
    "inference_time": 0.123
}
```

失败时返回：

```json
{
    "success": false,
    "error": "错误信息",
    "image_path": "xxx"
}
```

---

## 注意事项

1. **图片路径必须为服务器本地的绝对路径。**
2. `best.pt` 必须为 YOLO 分类模型权重（非检测权重）。
3. `class_indices.json` 和 `ID_to_chinese.json` 需与训练时类别顺序一致。
4. 若无预测结果，`predictions` 返回空列表。

---

## 常见问题

- **模型未检测到目标怎么办？**  
  检查图片内容、模型权重和类别映射文件是否匹配。

- **如何更换模型？**  
  替换 `best.pt` 文件，并确保类别映射文件同步更新。

---



---