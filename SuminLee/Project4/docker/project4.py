#모델 생성만 있는 파일을 도커에 포함하였습니다.
#VGG net

import tensorflow as tf
from keras.layers import BatchNormalization, Dropout, Activation

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


INPUT_SHAPE = (224, 224, 3) # 입력 데이터가 들어가는 포멧 설정
OUTPUT_SHAPE = 4 # 출력 데이터가 나오는 포멧 설정
BATCH_SIZE = 64 # 한 번에 처리할 데이터량 설정
EPOCHS = 70 # 신경망을 학습할 횟수
VERBOSE = 1 # 학습 진행 상황 출력 모드 설정 


model = Sequential([
  Conv2D(64, 3, strides=1, padding='same', input_shape=INPUT_SHAPE),
  BatchNormalization(),
  Activation(activation='relu'),
  Conv2D(64, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  MaxPooling2D((2,2), strides=2),

  Conv2D(128, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  Conv2D(128, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  MaxPooling2D((2,2), strides=2),

  Conv2D(256, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  Conv2D(256, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  Conv2D(256, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  MaxPooling2D((2,2), strides=2),

  Conv2D(512, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  Conv2D(512, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  Conv2D(512, 3, strides=1, padding='same'),
  BatchNormalization(),
  Activation(activation='relu'),
  MaxPooling2D((2,2), strides=2),
  MaxPooling2D((2,2), strides=2),

  Flatten(),
  Dense(4096, activation='relu'),
  Dropout(0.5),
  Dense(4096, activation='relu'),
  Dropout(0.5),
  Dense(4, activation='softmax')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy', 
    #categorical_crossentropy: 다중 분류 손실함수. 출력값이 원 핫 인코딩 된 결과로 나온다.
    #각 샘플이 정확히 하나의 클래스에 속하는 경우 사용한다.
    metrics=['accuracy']
)