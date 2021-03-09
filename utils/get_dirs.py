"""
util script to get the dirs file, notice that it is already given
"""

import scipy.io

mat = scipy.io.loadmat('CelltypeClassification.mat')
temp = {}

f = open('dirs.txt', mode = 'w')

for file in mat['sPV'][0][0][0]:
    if file[0][0] not in temp:
        temp[file[0][0]] = 1
        f.write('Data\\' + file[0][0] + '\n')
    else:
        temp[file[0][0]] += 1

print(temp)
        

f.close()
