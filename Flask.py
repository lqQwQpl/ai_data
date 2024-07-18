from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# 使用 Agg 后端以避免 GUI 警告
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

# 加载 .keras 模型
model = load_model(r'D:\py_project\Ai_data\D2_cnn_model.h5', compile=False)

# 文件上传配置
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

Image.MAX_IMAGE_PIXELS = None
def load_image(file_paths, target_size=(256, 256)):
    images = []
    for file_path in file_paths:
        try:
            img = load_img(file_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            continue
            
    if not images:
        raise ValueError("No valid images loaded.")
        
    images = np.array(images)
    return images

def generate_default_chart():
    # 生成默认图表
    plt.figure(figsize=(10, 5))
    plt.bar(['Class 1', 'Class 2', 'Class 3'], [5, 3, 2])  # 示例数据
    plt.title('Default Prediction Chart')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    return base64.b64encode(img_buf.getvalue()).decode('utf-8')

@app.route('/api/get-default-chart', methods=['GET'])
def get_default_chart():
    img_base64 = generate_default_chart()
    return jsonify({'chart': f'data:image/png;base64,{img_base64}'})

@app.route('/api/analyze-file', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('file')

    temp_folder = 'temp_uploads'
    os.makedirs(temp_folder, exist_ok=True)

    file_paths = []
    try:
        # 保存上传的文件
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_folder, filename)
            file.save(file_path)
            file_paths.append(file_path)

        # 使用 load_image 函数加载和处理图像数据
        images = load_image(file_paths)

        if images.size == 0:
            return jsonify({'error': 'No valid images found'}), 400

        # 使用模型进行预测
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)

        # 生成预测结果图表
        plt.figure(figsize=(10, 5))
        plt.hist(predicted_classes, bins=np.arange(len(set(predicted_classes)) + 1) - 0.5, edgecolor='black')
        plt.title('Predicted Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        
        # 将图表保存为图像
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')

        return jsonify({
            'predicted_classes': predicted_classes.tolist(),
            'chart': f'data:image/png;base64,{img_base64}'  # 返回图表的base64编码
        })
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
    finally:
        # 清理临时文件夹
        for file_path in file_paths:
            os.remove(file_path)
        os.rmdir(temp_folder)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # 默认情况下，Flask会在localhost和5000端口上运行
    app.run(host='127.0.0.1', port=5000, debug=True)