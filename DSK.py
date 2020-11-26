def DSK_algorithm(func, x, h):
    if func(x) < func(x + h):
        h = -1*h
    hi=h
    xi=x
    i=0
    while func(xi) > func(xi + hi):
        xi = xi + hi;
        hi = 2 * hi;
        i= i + 1
    xn=[]
    xn.append(xi + hi);
    xn.append(xi);
    xn.append(xi - (hi / 2));
    xn.append(xi + hi - hi / 2);
    a=0
    b=0
    if func(xn[0]) < func(xn[2]):
        a = min(xn[0], xn[1])
        b = max(xn[1], xn[0])
    if func(xn[0]) > func(xn[2]):
        a = min(xn[2], xn[3])
        b = max(xn[2], xn[3])
    segment=[]
    segment.append(a)
    segment.append(b)
    return segment