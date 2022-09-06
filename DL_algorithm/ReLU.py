"""
ReLU함수는 딥러링에서 가장 인기 있는 활성화 함수로 특히 합성곱 신경망(CNN)에서 많이 사용된다.
계산이 빠르고 신층 신경망에서도 신호 전달이 잘 된다.
--일반 적으로 은닉 층에 ReLU, 출력 층에 시그모이드 or 소프트맥스 적용.
ReLU함수 순전파 & 역전파
y = (x>0)*x, xb = (y>0)*1*yb
"""
i1, i2 = .05, .10
t1, t2 = .01, .99

w1, w3 = .15, .25
w2, w4 = .20, .30
b1, b2 = .35, .35

w5, w7 = .40, .50
w6, w8 = .45, .55
b3, b4 = .60, .60

for epoch in range(2000):

    print(' epoch = %d'%epoch)

    h1 = i1*w1 + i2*w2 + 1*b1
    h2 = i1*w3 + i2*w4 + 1*b2
    h1 = (h1>0)*h1
    h2 = (h2>0)*h2

    o1 = h1*w5 + h2*w6 + 1*b3
    o2 = h1*w7 + h2*w8 + 1*b4
    o1 = (o1>0)*o1
    o2 = (o2>0)*o2

    print(' o1, o2 = %6.3f, %6.3f '%(o1,o2))

    E = (o1-t1)**2/2 + (o2-t2)**2/2
    if E<0.0000001:
        break

    o1b, o2b = o1 - t1, o2 - t2
    o1b, o2b = (o1>0)*1*o1b, (o2>0)*1*o2b

    h1b, h2b = o1b*w5+o2b*w7, o1b*w6+o2b*w8
    h1b, h2b = (h1>0)*1*h1b, (h2>0)*1*h2b

    w1b, w3b = i1*h1b, i1*h2b
    w2b, w4b = i2*h1b, i2*h2b
    b1b, b2b = 1*h1b, 1*h2b
    w5b, w7b = h1*o1b, h1*o2b
    w6b, w8b = h2*o1b, h2*o2b
    b3b, b4b = 1*o1b, 1*o2b

    lr = 0.01
    w1, w3 = w1 - lr*w1b, w3 - lr*w3b
    w2, w4 = w2 - lr*w2b, w4 - lr*w4b
    b1, b2 = b1 - lr*b1b, b2 - lr*b2b
    w5, w7 = w5 - lr*w5b, w7 - lr*w7b
    w6, w8 = w6 - lr*w6b, w8 - lr*w8b
    b3, b4 = b3 - lr*b3b, b4 - lr*b4b