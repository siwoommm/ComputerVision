import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt


# 1. CIFAR-10 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 일부만 사용 (메모리 절약)
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# 3. 정답 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. 리사이즈
def resize_batch(images, size=(224, 224)):
    resized = np.zeros((images.shape[0], size[0], size[1], 3), dtype=np.float32)
    for i in range(images.shape[0]):
        resized[i] = tf.image.resize(images[i], size).numpy()
    return resized

x_train_resized = resize_batch(x_train)
x_test_resized = resize_batch(x_test)

# 5. VGG용 전처리
x_train_resized = preprocess_input(x_train_resized)
x_test_resized = preprocess_input(x_test_resized)

# 6. VGG16 로드 (top 제거, 고정)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 7. 새 분류기 연결
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 8. 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 9. 훈련
model.fit(x_train_resized, y_train,
          epochs=10,
          batch_size=32,
          validation_split=0.1)

# 10. 평가
loss, acc = model.evaluate(x_test_resized, y_test)
print(f"\n테스트 정확도: {acc:.4f}")

# CIFAR-10 클래스 라벨 목록
cifar10_labels = ['airplane','automobile','bird','cat','deer',
                  'dog','frog','horse','ship','truck']

# 11. 예측 수행
preds = model.predict(x_test_resized[:5])
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test[:5], axis=1)

# 12. 시각화
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {cifar10_labels[pred_labels[i]]}\nTrue: {cifar10_labels[true_labels[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

