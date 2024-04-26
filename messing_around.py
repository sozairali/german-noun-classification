chr = ['<S>'] + list(str('foo')) + ['<E>']

print(chr)

chs = []
for i in range(0, 3):
    
    print(i, chr[i:])
    chs.append(chr[i:])
    print(chs)


chs = list(zip(*chs))
print(chs)

print(type(chs[0]))