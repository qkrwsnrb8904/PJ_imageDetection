import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# 데이터 경로 설정
data_dir = './Image_data'  # 클래스별 폴더가 있는 디렉토리
test_image_path = 'can.jpg'  # 외부 테스트 이미지 경로

# 데이터 증강 및 로드
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 데이터의 20%를 검증용으로 사용
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # 학습용 데이터
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # 검증용 데이터
)

# 사전 학습된 모델 로드 (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False)

# 모델 수정 (쓰레기 분류용)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 사전 학습된 레이어 동결
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# 외부 이미지 분류 함수
def classify_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    
    predicted_class = class_indices[np.argmax(prediction)]
    print(f"Predicted class: {predicted_class}")

# 외부 이미지 테스트
classify_image(test_image_path)
