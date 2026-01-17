import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from emnist import extract_training_samples, extract_test_samples
import time

print("="*70)
print("NHẬN DIỆN CHỮ VIẾT TAY BẰNG K-NEAREST NEIGHBORS (KNN)")
print("="*70)

# 1. TẢI DỮ LIỆU
print("\n 1. Tải dữ liệu từ EMNIST...")
x_train_letters, y_train_letters = extract_training_samples('letters')
x_test_letters, y_test_letters = extract_test_samples('letters')

x_train_digits, y_train_digits = extract_training_samples('digits')
x_test_digits, y_test_digits = extract_test_samples('digits')

# Gộp dữ liệu
x_train = np.concatenate([x_train_digits, x_train_letters], axis=0)
x_test = np.concatenate([x_test_digits, x_test_letters], axis=0)
y_train = np.concatenate([y_train_digits, y_train_letters + 10], axis=0)
y_test = np.concatenate([y_test_digits, y_test_letters + 10], axis=0)

print(f"✓ Train samples: {len(x_train)}")
print(f"✓ Test samples: {len(x_test)}")
print(f"✓ Classes: 36 (0-9 và A-Z)")

# 2. LẤY SUBSET ĐỂ TRAIN NHANH
# KNN không cần "training" nhưng cần lưu toàn bộ data
# Để demo nhanh, ta lấy subset
USE_SUBSET = False  # Đổi thành False để dùng full dataset

if USE_SUBSET:
    print("\n 100,000 samples")
    
    # Lấy 100,000 mẫu train
    indices = np.random.choice(len(x_train), 100000, replace=False)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # Lấy 50,000 mẫu test
    indices_test = np.random.choice(len(x_test), 50000, replace=False)
    x_test = x_test[indices_test]
    y_test = y_test[indices_test]
    
    print(f"✓ Subset train: {len(x_train)}")
    print(f"✓ Subset test: {len(x_test)}")

# 3. TIỀN XỬ LÝ - FLATTEN
print("\n 2. Tiền xử lý dữ liệu...")

# Flatten ảnh 28x28 thành vector 784 chiều
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

# Chuẩn hóa về [0, 1]
x_train_flat = x_train_flat.astype('float32') / 255.0
x_test_flat = x_test_flat.astype('float32') / 255.0

print(f"✓ Shape sau khi flatten: {x_train_flat.shape}")

# 4. FEATURE SCALING
print("\n 3. Chuẩn hóa features (StandardScaler)...")
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)
print("✓ Đã chuẩn hóa features về mean = 0, std = 1")

# 5. DIMENSIONALITY REDUCTION - PCA

# Giảm về 100 tối ưu cho KNN
print("\n 4. Giảm chiều dữ liệu bằng PCA...")

pca = PCA(n_components=100, random_state=42)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"✓ PCA giữ lại {explained_variance:.2f}% thông tin")
print(f"✓ Shape sau PCA: {x_train_pca.shape}")

# 6. TRAINING KNN MODEL
print("\n 5. Đang train K-Nearest Neighbors (KNN)...")
print("-"*70)

print("\n🔧 K-Nearest Neighbors (KNN) Algorithm:")

start_time = time.time()

# Tạo KNN model
knn_model = KNeighborsClassifier(
    n_neighbors=10,      # K = 10
    weights='distance', # Trọng số theo khoảng cách
    algorithm='auto',   # Tự động chọn thuật toán tối ưu
    n_jobs=-1          # Dùng tất cả CPU cores
)

# "Train" KNN (thực chất chỉ lưu data)
knn_model.fit(x_train_pca, y_train)

train_time = time.time() - start_time

print(f"\n✓ KNN 'training' hoàn tất!")
print(f"✓ Time: {train_time:.2f}s ")

# 7. ĐÁNH GIÁ MODEL
print("\n 6. Quá trình đánh giá thuật toán ")

start_predict = time.time()
y_pred = knn_model.predict(x_test_pca)
predict_time = time.time() - start_predict

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("KẾT QUẢ KNN MODEL:")
print("="*70)
print(f"✓ Accuracy: {accuracy*100:.2f}%")
print(f"✓ Training time: {train_time:.2f}s")
print(f"✓ Prediction time: {predict_time:.2f}s ({len(x_test)} samples)")
print(f"✓ Avg time per sample: {predict_time/len(x_test)*1000:.2f}ms")

# 8. CHI TIẾT ĐÁNH GIÁ
print(f"\n 7. Đánh giá chi tiết KNN model:")
print("-"*70)

def get_character(idx):
    return str(idx) if idx < 10 else chr(ord('A') + idx - 10)

target_names = [get_character(i) for i in range(36)]

print("\nClassification Report (10 classes đầu - Số 0-9):")
print(classification_report(y_test, y_pred, target_names=target_names, 
                          labels=list(range(10)), zero_division=0))

# 9. LƯU MODEL
print("\n 8. Đang lưu KNN model và preprocessing objects...")

model_data = {
    'best_model': knn_model,
    'best_model_name': 'K-Nearest Neighbors (K=10)',
    'scaler': scaler,
    'pca': pca,
    'all_models': {'K-Nearest Neighbors (K=10)': knn_model},
    'results': {
        'K-Nearest Neighbors (K=10)': {
            'accuracy': accuracy,
            'train_time': train_time
        }
    }
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Đã lưu: model.pkl")
print(f"   - Model: K-Nearest Neighbors (K=10)")
print(f"   - Accuracy: {accuracy*100:.2f}%")
print(f"   - Scaler: StandardScaler")
print(f"   - PCA: 100 components")

# 10. VẼ BIỂU ĐỒ
print("\n 9. Vẽ biểu đồ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ 1: Accuracy per class (10 classes đầu)
ax1 = axes[0]
class_accuracies = []
for i in range(10):
    mask = y_test == i
    if np.sum(mask) > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        class_accuracies.append(acc * 100)
    else:
        class_accuracies.append(0)

bars = ax1.bar([get_character(i) for i in range(10)], class_accuracies, color='#3498db')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Class (0-9)', fontsize=12, fontweight='bold')
ax1.set_title('KNN Accuracy per Class (Digits)', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Biểu đồ 2: Confusion Matrix (5x5 đầu)
ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred, labels=list(range(5)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=[get_character(i) for i in range(5)])
disp.plot(ax=ax2, cmap='Blues', colorbar=True)
ax2.set_title('Confusion Matrix - KNN\n(Classes 0-4)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('knn_results.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu biểu đồ: knn_results.png")

# 11. VẼ PCA COMPONENTS
print("\nĐang vẽ PCA components...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Top 10 PCA Components (Features learned)', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    component = pca.components_[i].reshape(28, 28)
    ax.imshow(component, cmap='coolwarm')
    ax.set_title(f'Component {i+1}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('pca_components.png', dpi=150, bbox_inches='tight')
print("✓ Đã lưu: pca_components.png")

# 12. KIỂM TRA DỰ ĐOÁN MẪU
print("\n" + "="*70)
print("KIỂM TRA 15 DỰ ĐOÁN MẪU:")
print("="*70)

sample_predictions = knn_model.predict(x_test_pca[:15])
for i in range(15):
    pred_char = get_character(sample_predictions[i])
    true_char = get_character(y_test[i])
    status = "✓" if pred_char == true_char else "✗"
    print(f"{status} Mẫu {i+1:2d}: Dự đoán = '{pred_char}', Thực tế = '{true_char}'")

# 13. VẼ SAMPLE PREDICTIONS
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('Sample Predictions - KNN Model', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap='gray')
    ax.axis('off')
    
    pred_char = get_character(sample_predictions[i])
    true_char = get_character(y_test[i])
    color = 'green' if pred_char == true_char else 'red'
    
    ax.set_title(f"Pred: {pred_char} | True: {true_char}", 
                color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('knn_sample_predictions.png', dpi=150, bbox_inches='tight')
print("\n✓ Đã lưu: knn_sample_predictions.png")

print("\n" + "="*70)
print("✓ HOÀN TẤT!")
print("="*70)

print("Truy cập: http://localhost:3000")
print("\nVẽ chữ số hoặc chữ cái để test!")
print("="*70)