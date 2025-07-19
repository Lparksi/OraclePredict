import requests
import base64
import json
from PIL import Image
import io
import numpy as np

def test_api():
    """测试API接口的简单脚本"""
    
    # API基础URL
    base_url = "http://localhost:5000"
    
    print("=== Oracle Predict API 测试 ===\n")
    
    # 1. 测试健康检查
    print("1. 测试健康检查接口...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        print()
    except Exception as e:
        print(f"健康检查失败: {e}")
        return
    
    # 2. 测试模型信息
    print("2. 测试模型信息接口...")
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print()
    except Exception as e:
        print(f"获取模型信息失败: {e}")
    
    # 3. 创建测试图片
    print("3. 创建测试图片...")
    test_image = Image.new('RGB', (640, 480), color='red')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # 转换为base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    print(f"创建了 {test_image.size} 的测试图片")
    print()
    
    # 4. 测试单张图片推理 (base64)
    print("4. 测试单张图片推理 (base64)...")
    try:
        payload = {"image": f"data:image/png;base64,{img_base64}"}
        response = requests.post(f"{base_url}/predict", json=payload)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print()
    except Exception as e:
        print(f"单张图片推理失败: {e}")
    
    # 5. 测试文件上传
    print("5. 测试文件上传推理...")
    try:
        img_buffer.seek(0)
        files = {'image': ('test.png', img_buffer, 'image/png')}
        response = requests.post(f"{base_url}/predict", files=files)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print()
    except Exception as e:
        print(f"文件上传推理失败: {e}")
    
    # 6. 测试批量推理
    print("6. 测试批量推理...")
    try:
        # 创建多张测试图片
        images = []
        for i in range(2):
            test_img = Image.new('RGB', (320, 240), color=['red', 'blue'][i])
            img_buf = io.BytesIO()
            test_img.save(img_buf, format='PNG')
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.getvalue()).decode()
            images.append(f"data:image/png;base64,{img_b64}")
        
        payload = {"images": images}
        response = requests.post(f"{base_url}/predict/batch", json=payload)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print()
    except Exception as e:
        print(f"批量推理失败: {e}")

if __name__ == "__main__":
    print("请确保API服务已启动 (python app.py)")
    print("按回车键开始测试...")
    input()
    test_api()
