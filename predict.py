
import os
import logging
from flask import Flask, request, jsonify
import json
from PIL import Image
import time
from ultralytics import YOLO
import torch
from flasgger import Swagger
from torchvision import transforms

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
swagger = Swagger(app)

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
    logger.info("映射文件加载成功")
except Exception as e:
    logger.error(f"初始化错误: {str(e)}")
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
        logger.info(f"YOLO模型加载完成。设备={device}，加载时间={time.time() - start_time:.2f}秒")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
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
        logger.info(f"开始预测: {image_path}")
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
            logger.warning(f"未检测到分类概率: {image_path}")
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
        logger.info(f"预测完成: {image_path} -> {predictions}")
        return predictions
    except FileNotFoundError as e:
        logger.error(f"文件错误: {str(e)}")
        raise Exception(f"文件错误: {str(e)}")
    except ValueError as e:
        logger.error(f"输入错误: {str(e)}")
        raise Exception(f"输入错误: {str(e)}")
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
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
    """
    图片分类预测
    ---
    tags:
      - 预测接口
    parameters:
      - name: file_path
        in: formData
        type: string
        required: true
        description: 图片文件路径
    responses:
      200:
        description: 预测结果
        schema:
          type: object
          properties:
            success:
              type: boolean
            image_path:
              type: string
            predictions:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: string
                  confidence:
                    type: number
                  chinese_char:
                    type: string
            inference_time:
              type: number
      400:
        description: 参数错误
      500:
        description: 服务器错误
    """
    # 检查表单数据
    if 'file_path' not in request.form:
        logger.warning("请求缺少 file_path 参数")
        return jsonify({
            'success': False,
            'error': '请提供文件路径（file_path）'
        }), 400

    file_path = request.form['file_path']
    logger.info(f"收到预测请求: {file_path}")
    try:
        start_time = time.time()
        results = predict_single_image(file_path, model, class_indices, device)
        resp = {
            'success': True,
            'image_path': file_path,
            'predictions': results,
            'inference_time': round(time.time() - start_time, 3)
        }
        logger.info(f"预测响应: {resp}")
        return jsonify(resp)
    except Exception as e:
        logger.error(f"预测异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'image_path': file_path
        }), 500


# 新增：支持直接上传图片的接口
@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """
    图片上传并分类预测
    ---
    tags:
      - 预测接口
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: 待预测图片文件
    responses:
      200:
        description: 预测结果
        schema:
          type: object
          properties:
            success:
              type: boolean
            image_path:
              type: string
            predictions:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: string
                  confidence:
                    type: number
                  chinese_char:
                    type: string
            inference_time:
              type: number
      400:
        description: 参数错误
      500:
        description: 服务器错误
    """
    if 'file' not in request.files:
        logger.warning("请求缺少 file 参数")
        return jsonify({'success': False, 'error': '请上传图片文件（file）'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("未选择文件")
        return jsonify({'success': False, 'error': '未选择文件'}), 400
    if not allowed_file(file.filename):
        logger.warning(f"不支持的文件格式: {file.filename}")
        return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

    temp_path = os.path.join(BASE_DIR, f"temp_{int(time.time()*1000)}_{file.filename}")
    file.save(temp_path)
    logger.info(f"收到上传图片: {temp_path}")
    try:
        start_time = time.time()
        results = predict_single_image(temp_path, model, class_indices, device)
        resp = {
            'success': True,
            'image_path': temp_path,
            'predictions': results,
            'inference_time': round(time.time() - start_time, 3)
        }
        logger.info(f"上传预测响应: {resp}")
        return jsonify(resp)
    except Exception as e:
        logger.error(f"上传预测异常: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'image_path': temp_path
        }), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"已删除临时文件: {temp_path}")
            except Exception as ex:
                logger.warning(f"删除临时文件失败: {temp_path}, {str(ex)}")

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")