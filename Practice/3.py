import tensorflow as tf
import matplotlib.pyplot as plt

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
    plt.xlable(y_train[i]) # x라벨 값.

plt.show()

x_train, x_test = x_train /255.0, x_test / 255.0
x_train, x_test = x_train.reshape(60000,784) , x_test.reshape(10000,784)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuacy'])

model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test,y_test)