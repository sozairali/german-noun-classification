'''

def create_ngram(word, n):
    for w in word:
        chs = ['<S>'] + list(str(word)) + ['<E>']
        for i in range(n):
            chs1 = yield chs[n:]
            chs = zip[chs, chs1]
            
'''
