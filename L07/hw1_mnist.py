import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
x_train = x_train / 255.0  # 픽셀 값 정규화 (0~1)
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  # 원-핫 인코딩
y_test = to_categorical(y_test, 10)

# 3. 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),       
    Dense(128, activation='relu'),       
    Dense(10, activation='softmax')      
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 6. 테스트 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\n테스트 정확도: {test_acc:.4f}')
