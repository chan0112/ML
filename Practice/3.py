import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # mnist는 손 글씨 숫자 데이터를 가진 모듈.

(x_train, y_train),(x_test,y_test) = mnist.load_data() # load_data 함수 호출하여 손 글씨 숫자 데이터를 읽어옴
print("x_train=%s y_train=%s x_test=%s y_test=%s "%(x_train.shape,y_train.shape,x_test.shape,y_test.shape))

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28):
    for x in range(28):
        print(" %4s "%x_train[0][y][x],end=' ') # 각 픽셀 값 출력
    print()

plt.figure(figsize=(10,10)) # 그림 만들준비 및 그림의 인치 단위 크기 조정.
for i in range(25):
    plt.subplot(5,5,i+1) # 그릠 창 분할 후 하위 그림을 그림. 5,5 는 행과 열의 개수 i+1은 그림의 위치
    plt.xticks([]) # x축 눈금 설정, 이때 빈 리스트를 주어 눈금 표시 x
    plt.yticks([]) # y축 눈금 설정, 이때 빈 리스트를 주어 눈금 표시 x
    plt.imshow(x_train[i], cmap=plt.cm.binary) # color map, binary로 그림 이진화해서 표현
    plt.xlabel(y_train[i]) # x라벨 값.

plt.show()

x_train, x_test = x_train /255.0, x_test / 255.0 # 모든 픽셀의 숫자를 255.0으로 나누어 각 필섹 값을 0.0~1.0 사이의 실수로 바꿈
x_train, x_test = x_train.reshape(60000,784) , x_test.reshape(10000,784)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)), # 784개의 입력노드 생성
    tf.keras.layers.Dense(128, activation='relu'), # 128개의 단위 인공 신경 생성 이떄 활성화 함수는 relu
    tf.keras.layers.Dense(10, activation='softmax') # 10개의 단위 인공 신경 생성 이떄 활성화 함수는 softmax
])  # 인공 신경망 생성, 128 + 10개의 인공 신경으로 구성

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # 인공 신경망 구성, 손실+최적화 함수를 이용.

model.fit(x_train,y_train,epochs=5) # fit 함수에 x_train, y_train 데이터를 입력하여 5번 학습 시작

model.evaluate(x_test,y_test) # 학습이 끝난 인공신경망에 x_test값을 주어 학습 결과 평가

p_test = model.predict(x_test) # predict함수 호출, x_test값을 주어 결과를 예측해보며 인공 신경만 시험. 이떄 값은 p_test로 받는다.
print('p_test[0] : ',p_test[0]) # 예측값 출력

print('p_test[0] : ', np.argmax(p_test[0]), ' y_test[0] : ',y_test[0])

x_test = x_test.reshape(10000,28,28)

cnt_wrong = 0 # 잘못 예측된 그림 개수
p_wrong = [] # 잘못 예측된 그림 번호
for i in range(10000):
    if np.argmax(p_test[i]) != y_test[i]:
        p_wrong.append(i)
        cnt_wrong += 1

print(' cnt_wrong : ', cnt_wrong)
print(' predicted wrong 10 : ', p_wrong[:10])

plt.figure(figsize=(10,10))
for i in range(25):  # 틀린거 보여주기
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
    plt.xlabel(" %s : p%s y%s " %(p_wrong[i],np.argmax(p_test[p_wrong[i]]), y_test[p_wrong[i]]))

plt.show()

""" fashion_mnist를 이용하여 패션 관련 데이터 셋 가능.  """