import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. CNN 모델 설계
model = Sequential([
    # Conv Block 1
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Conv Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Conv Block 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # FC Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 4. 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 훈련
model.fit(x_train, y_train,
          epochs=10,
          batch_size=64,
          validation_split=0.1)

# 6. 평가
loss, acc = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {acc:.4f}")

# 7. 테스트 이미지 예측
# 예: 테스트 이미지 0~4번 출력 및 예측 결과 확인
preds = model.predict(x_test[:5])
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test[:5], axis=1)

# 8. 시각화
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {cifar10_labels[pred_labels[i]]}\nTrue: {cifar10_labels[true_labels[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
