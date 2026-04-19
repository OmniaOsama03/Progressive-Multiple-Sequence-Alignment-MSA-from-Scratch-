from numpy import zeros, arange, argmax

def dp(v, w, delta, gap=0, local=False):
    n = len(v)
    m = len(w)
    s = zeros((n+1, m+1), dtype=int)
    b = zeros((n+1, m+1), dtype=int)
    b[0, 1:] = 1
    if not local:
        s[0] = arange(m+1) * gap
        s[:,0] = arange(n+1) * gap
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [s[i-1,j] + gap,
                       s[i,j-1] + gap,
                       s[i-1,j-1] + delta(v[i-1], w[j-1])]
            if local: options.append(0)
            bestOption = argmax(options)
            s[i,j] = options[bestOption]
            b[i,j] = bestOption
    return s, b
