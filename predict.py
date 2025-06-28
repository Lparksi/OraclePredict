import os
from flask import Flask, request, jsonify
import json
from PIL import Image
import time
from ultralytics import YOLO
import torch

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 使用os.path.join来确保跨平台路径兼容性
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载映射文件
def load_json_file(filename):
    file_path = os.path.join(BASE_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception(f"映射文件 {filename} 不存在")
    except json.JSONDecodeError:
        raise Exception(f"映射文件 {filename} 格式错误")

try:
    class_indices = load_json_file('class_indices.json')
    id_to_chinese = load_json_file('ID_to_chinese.json')
except Exception as e:
    print(f"初始化错误: {str(e)}")
    exit(1)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path, device):
    """加载YOLO模型"""
    start_time = time.time()
    try:
        model_path = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在")
        model = YOLO(model_path)
        print(f"YOLO模型加载完成。设备={device}，加载时间={time.time() - start_time:.2f}秒")
        return model
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")

def get_transform():
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(image_path, model, class_indices, device):
    """YOLO分类模型预测，返回top5"""
    try:
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("文件路径必须是非空字符串")
        if not os.path.exists(image_path):
            raise FileNotFoundError("文件不存在")
        if not allowed_file(image_path):
            raise ValueError("不支持的文件格式")

        # YOLO分类模型推理
        results = model(image_path)
        probs = results[0].probs  # 分类概率
        if probs is None:
            return []

        top5_ids = probs.top5
        top5_confs = probs.top5conf

        predictions = []
        for class_id, confidence in zip(top5_ids, top5_confs):
            predictions.append({
                'id': class_indices.get(str(class_id), str(class_id)),
                'confidence': round(float(confidence), 5),
                'chinese_char': id_to_chinese.get(class_indices.get(str(class_id), str(class_id)), "未知")
            })
        return predictions
    except FileNotFoundError as e:
        raise Exception(f"文件错误: {str(e)}")
    except ValueError as e:
        raise Exception(f"输入错误: {str(e)}")
    except Exception as e:
        raise Exception(f"预测过程出错: {str(e)}")

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "best.pt"  # 替换为你的YOLO权重文件
try:
    model = load_model(model_path, device=device)
except Exception as e:
    print(f"模型初始化失败: {str(e)}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    # 检查表单数据
    if 'file_path' not in request.form:
        return jsonify({
            'success': False,
            'error': '请提供文件路径（file_path）'
        }), 400

    file_path = request.form['file_path']
    
    try:
        start_time = time.time()
        results = predict_single_image(file_path, model, class_indices, device)
        
        return jsonify({
            'success': True,
            'image_path': file_path,
            'predictions': results,
            'inference_time': round(time.time() - start_time, 3)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'image_path': file_path
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")