import numpy as np

np.set_printoptions(formatter={' float_kind ':lambda x: " {0:6.3f}".format(x)})

X = np.array([[2,3,4]])
T = np.array([[27,-30,179]])
W = np.array([[3,5,8],[4,6,9],[5,7,10]])
B = np.array([[1,2,3]])

for epoch in range(1000):

    print(' epoch = %d'%epoch)

    Y = X @ W + B # X 행렬의 가로줄 항목, W 행렬의 세로줄 항목이 순서대로 곱해진 후 모두 더해져서 임시 행렬의 항목을 구성.
    print(' Y =',Y)

    E = np.sum((Y-T)**2/2)
    print(' E = %.7f'%E)
    if E < 0.0000001:
        break

    Yb = Y-T
    Xb = Yb @ W.T
    Wb = X.T @ Yb
    Bb = 1*Yb
    print(' Xb =\n', Xb)
    print(' Wb =\n', Wb)
    print(' Bb =\n', Bb)

    lr = 0.01
    W = W - lr*Wb
    B = B -lr*Bb
    print(' W=\n',W)
    print(' B=\n',B)