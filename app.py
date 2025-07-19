from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import sys
import os

# 将inferences目录添加到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'inferences'))
from engines import inference

app = Flask(__name__)
CORS(app)  # 允许跨域请求


def decode_image(image_data):
    """解码base64图片数据或处理上传的文件"""
    try:
        if isinstance(image_data, str):
            # 处理base64编码的图片
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # 解码base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # 转换为OpenCV格式
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_cv = image_np
                
            return image_cv
        else:
            # 处理文件上传
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_cv = image_np
                
            return image_cv
    except Exception as e:
        raise ValueError(f"图片解码失败: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'message': 'API服务运行正常'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """图片推理接口"""
    try:
        # 检查请求中是否包含图片数据
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({
                'error': '缺少图片数据',
                'message': '请在请求中包含图片文件或base64编码的图片数据'
            }), 400
        
        # 处理文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'error': '未选择文件',
                    'message': '请选择要上传的图片文件'
                }), 400
            
            # 读取文件数据
            image_data = file.read()
            image = decode_image(image_data)
        
        # 处理JSON中的base64数据
        elif 'image' in request.json:
            image_data = request.json['image']
            image = decode_image(image_data)
        
        # 执行推理
        results = []
        for detection in inference(image):
            results.append({
                'bbox': detection['bbox'],  # [x1, y1, x2, y2]
                'class': detection['classes'],
                'confidence': 1.0  # 由于原代码没有返回置信度，这里设置为1.0
            })
        
        return jsonify({
            'success': True,
            'message': '推理完成',
            'results': results,
            'count': len(results)
        })
    
    except ValueError as ve:
        return jsonify({
            'error': '输入数据错误',
            'message': str(ve)
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': '推理失败',
            'message': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """批量图片推理接口"""
    try:
        if 'images' not in request.json:
            return jsonify({
                'error': '缺少图片数据',
                'message': '请在请求中包含images数组'
            }), 400
        
        images_data = request.json['images']
        if not isinstance(images_data, list):
            return jsonify({
                'error': '数据格式错误',
                'message': 'images应该是一个数组'
            }), 400
        
        batch_results = []
        
        for i, image_data in enumerate(images_data):
            try:
                image = decode_image(image_data)
                
                results = []
                for detection in inference(image):
                    results.append({
                        'bbox': detection['bbox'],
                        'class': detection['classes'],
                        'confidence': 1.0
                    })
                
                batch_results.append({
                    'index': i,
                    'success': True,
                    'results': results,
                    'count': len(results)
                })
            
            except Exception as e:
                batch_results.append({
                    'index': i,
                    'success': False,
                    'error': str(e),
                    'results': [],
                    'count': 0
                })
        
        return jsonify({
            'success': True,
            'message': '批量推理完成',
            'batch_results': batch_results,
            'total_processed': len(images_data)
        })
    
    except Exception as e:
        return jsonify({
            'error': '批量推理失败',
            'message': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    try:
        # 读取配置信息
        import toml
        configs = toml.load('inferences/configs/config.toml')
        
        return jsonify({
            'success': True,
            'model_info': {
                'detection_model': configs['detection-model-path'],
                'classification_model': configs['classification-model-path'],
                'providers': configs['providers'],
                'precision': configs['precision'],
                'conf_threshold': configs['conf-threshold'],
                'iou_threshold': configs['iou-threshold']
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': '获取模型信息失败',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # 检查模型文件是否存在
    detection_model_path = 'inferences/models/detection-fp32.onnx'
    classification_model_path = 'inferences/models/classification-fp32.onnx'
    
    if not os.path.exists(detection_model_path):
        print(f"警告: 检测模型文件不存在: {detection_model_path}")
    
    if not os.path.exists(classification_model_path):
        print(f"警告: 分类模型文件不存在: {classification_model_path}")
    
    print("启动Flask API服务...")
    print("API文档:")
    print("  GET  /health - 健康检查")
    print("  POST /predict - 单张图片推理")
    print("  POST /predict/batch - 批量图片推理")
    print("  GET  /model/info - 获取模型信息")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
