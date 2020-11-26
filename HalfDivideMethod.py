def half_divide(func, segment, eps):
    a=segment[0]
    b=segment[1]
    x1 = (a + b-eps) / 2
    x2=(a+b+eps)/2
    k=0
    while (b-a)>eps:
        if func(x1)<=func(x2):
            b=x2
            x1 = (a + b - eps) / 2
            x2 = (a + b + eps) / 2
        else:
            a = x1
            x1 = (a + b - eps) / 2
            x2 = (a + b + eps) / 2
        k= k + 1
        if k==50:
            break
    return (a+b)/2
