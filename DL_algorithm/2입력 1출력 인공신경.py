x1, x2 = 2,3
t = 27
w1 = 3
w2 = 4
b = 1

for epoch in range(2000):
    print('epoch = %d '%epoch)

    y = x1*w1 + x2*w2 + 1*b #순전파
    print(' y = %6.3f '%y)

    E = (y-t)**2/2
    print(' E = %.7f' %E)
    if E<0.0000001:
        break

    yb = y-t # 역전파 오차
    x1b,x2b = yb*w1, yb*w2 # 입력 역전파
    w1b = yb*x1  # 가중치, 편향 역전파
    w2b = yb*x2  # 가중치, 편향 역전파
    bb = yb*1    # 가중치, 편향 역전파
    print(' x1b, x2b = %6.3f, %6.3f '%(x1b,x2b))
    print(' w1b, w2b, bb = %6.3f, %6.3f, %6.3f '%(w1b,w2b,bb))

    lr = 0.01
    w1 = w1-lr*w1b # 인공 신경망 학습
    w2 = w2-lr*w2b # 인공 신경망 학습
    b = b-lr*bb # 인공 신경망 학습
    print(' w1, w2, b = %6.3f, %6.3f, %6.3f'%(w1,w2,b))