x = 2
t = 10
w = 3
b = 1

for epoch in range(2):

    print(' epoch = %d' %epoch)

    y = x*w + 1*b
    print(' y = %6.3f' %y)

    yb = y-t
    xb = yb*w
    wb = yb*x
    bb = yb*1

    print(' xb = %6.3f, wb = %6.3f, bb = %6.3f'%(xb,wb,bb))

    lr = 0.01
    w = w - lr*b
    b = b - lr*bb
    print( ' x = %6.3f, w = %6.3f , b = %6.3f'%(x,w,b))