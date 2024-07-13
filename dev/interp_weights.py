h_float = 1.1
w_float = 1.
total = 0
for ix in range(2):
    for jx in range(2):
        h_int = int(h_float+ix)
        w_int = int(w_float+jx)
        a = max(0,1-abs(h_int-h_float))
        b = max(0,1-abs(w_int-w_float))
        total += a*b
        print(a,b)
print(total)
