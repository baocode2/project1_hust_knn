from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import cv2
import pickle

app = Flask(__name__)

# --- TẢI MODEL ---
print("Đang tải KNN model...")
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

knn_model = model_data['best_model']
scaler = model_data['scaler']
pca = model_data['pca']

print("Model đã sẵn sàng!")

def get_character(class_index):
    """Chuyển index thành ký tự (0-9, A-Z)"""
    return str(class_index) if class_index < 10 else chr(ord('A') + class_index - 10)

def preprocess_image(image):
    """Xử lý ảnh đầu vào: Grayscale -> Invert -> Crop -> Resize 20x20 -> Pad 28x28 -> Flatten"""
    # 1. Xử lý màu và nền
    if len(image.shape) == 3:
        if image.shape[2] == 4: # Xử lý kênh Alpha
            background = np.ones_like(image[:,:,:3]) * 255
            alpha = image[:,:,3:4] / 255.0
            image = image[:,:,:3] * alpha + background * (1 - alpha)
            image = image.astype(np.uint8)
        # CẢI TIẾN: Thay vì Grayscale thông thường (dễ mất màu vàng/sáng trên nền trắng)
        # Ta dùng "Darkest Channel" (kênh tối nhất) để bắt được mực màu
        if np.mean(image) > 127: # Nếu ảnh tổng thể là sáng (nền trắng)
            image = np.min(image, axis=2) # Lấy kênh màu đậm nhất (Vàng RGB(255,255,0) -> Min=0 (Đen))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        image = image.astype(np.uint8)

    # 2. Đảo màu thành nền đen chữ trắng giống MNIST
    if np.mean(image) > 127:
        image = 255 - image

    # Áp dụng Otsu's Thresholding để loại bỏ nhiễu xám (quan trọng cho ảnh chụp/upload)
    # Nó sẽ tự động tìm ngưỡng để tách hẳn Chữ (Trắng) và Nền (Đen)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Cắt vùng chứa chữ (Crop Bounding Box)
    coords = cv2.findNonZero(image)
    if coords is None: return np.zeros(784, dtype='float32')
    x, y, w, h = cv2.boundingRect(coords)
    image = image[y:y+h, x:x+w]

    # 4. Resize về cạnh lớn nhất là 20
    scale = 20 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Đặt vào tâm khung 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    start_x, start_y = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = image

    # 6. Chuẩn hóa và duỗi thẳng ảnh
    return (canvas.astype('float32') / 255.0).flatten()

def get_probabilities(image_vector):
    """Trả về mảng xác suất từ KNN"""
    image_vector = image_vector.reshape(1, -1)
    image_scaled = scaler.transform(image_vector)
    image_pca = pca.transform(image_scaled)
    return knn_model.predict_proba(image_pca)[0]

def process_prediction(image_array):
    """Hàm xử lý chung cho cả file và base64"""
    # 1. Tiền xử lý
    processed_image = preprocess_image(image_array)
    
    # 2. Lấy xác suất
    probabilities = get_probabilities(processed_image)
    
    # 3. Lấy Top 5 index có xác suất cao nhất
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    
    # 4. Lấy Top 1 (Winner) từ vị trí đầu tiên của Top 5
    top_1_index = top_5_indices[0]
    
    # 5. Tạo danh sách kết quả chi tiết
    top_5_predictions = []
    for idx in top_5_indices:
        top_5_predictions.append({
            'class': int(idx),
            'character': get_character(int(idx)),
            'confidence': float(probabilities[idx])
        })

    return {
        'success': True,
        'predicted_class': int(top_1_index),           
        'predicted_character': get_character(int(top_1_index)),
        'confidence': float(probabilities[top_1_index]),
        'top_5': top_5_predictions
    }

@app.route('/')
def index():
    return render_template('index_knn.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data: return jsonify({'error': 'No image'}), 400
        
        image_bytes = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        return jsonify(process_prediction(np.array(image)))
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({'error': 'Empty file'}), 400
        
        image = Image.open(file.stream)
        return jsonify(process_prediction(np.array(image)))
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Server running at http://localhost:3000")
    app.run(debug=True, host='0.0.0.0', port=3000)