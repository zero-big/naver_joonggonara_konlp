# -----------------------  Step 0. 준비
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import sys

# -----------------------  Step 1. 데이터 모델링
#데이터 불러오기
X_train, X_test, Y_train, Y_test = np.load(
    'models/modeling_data_max_30_wordsize_5939.npy', allow_pickle=True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, X_test.shape)

#모델만들기
model = Sequential()
model.add(Embedding(5939, 300, input_length=30))   #총 단어들의 갯수와 제일 긴 문장이 가진 단어의 수를 넣음
# #자연어 일 경우엔 위 방법으로 인베딩해주면 됨
# 12426차원이 생기는데 차원이 너무 멀어 데이터 간의 유사성을 찾기 어려우므로, 300차원으로 바꿔서 밀집도를 높여 유사성을 찾음
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))   #문장의 앞뒤관계만 확인하므로 1차원
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
#모델저장
model.save('./models/joonggonara_learning.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()