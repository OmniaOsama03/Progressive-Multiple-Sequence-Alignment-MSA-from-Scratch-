from numpy import zeros, arange, argmax

def dp(v, w, delta, gap=0, local=False):

    n = len(v) #columns in profile v
    m = len(w) #columns in profile w

    s = zeros((n+1, m+1), dtype=float) #scoring matrix (float since value can be fractional with profiles)
    b = zeros((n+1, m+1), dtype=int) #backtracking matrix

    b[0, 1:] = 1 #top row is initialized to 1 bc those cells came from the left

    if not local:
        s[0] = arange(m+1) * gap #row 1: gaps into v
        s[:,0] = arange(n+1) * gap #column 1: gaps into w

    for i in range(1, n+1):
        for j in range(1, m+1):

            options = [s[i-1,j] + gap,
                       s[i,j-1] + gap,
                       s[i-1,j-1] + delta(v[i-1], w[j-1])]
            
            if local: 
                options.append(0)

            bestOption = argmax(options)
            s[i,j] = options[bestOption]
            b[i,j] = bestOption
    return s, b
