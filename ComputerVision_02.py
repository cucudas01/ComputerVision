import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt

# 1. GPU 장치 확인
print("사용 가능한 장치:", tf.config.list_physical_devices())

# 2. CIFAR-10 데이터셋 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. 모델 구축 (AlexNet 기반 성능 향상 버전)
model = models.Sequential([
    # [방법 1] 데이터 증강 (수평 뒤집기만 추가하여 연산량 최소화)
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),

    # Layer 1
    layers.Conv2D(96, (3, 3), activation='relu'),
    layers.BatchNormalization(), # [방법 2] 배치 정규화 추가
    layers.MaxPooling2D((2, 2)),

    # Layer 2
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Layer 3, 4, 5
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # FC Layers
    layers.Flatten(),
    
    # [방법 3] FC 레이어 최적화 (4096 대신 512 사용 - 속도 향상 핵심)
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 학습 (속도를 위해 10 에폭 유지)
print("\n--- 최적화된 GPU 학습 시작 ---")
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels),
                    batch_size=128)

# 6. 최종 성능 확인 및 그래프 출력
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n최종 테스트 정확도: {test_acc*100:.2f}%')

# 그래프 출력
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Improved AlexNet Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Improved AlexNet Loss')
plt.legend()
plt.show()