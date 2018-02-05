from numpy import genfromtxt
from numpy import savetxt

data = genfromtxt('WSe2.csv', delimiter = ',')
# print(data)
index = [[[i*4 + 3, i*4 + 1, i*4] for i in range(8)] for j in range(len(data) - 1)]
# print(index)
d = []
t = []
max = 0
min = 1
for idx in enumerate(index):
	# print(idx)
	i = idx[0] + 1
	for k, j in enumerate(idx[1]):
		a = -data[i][j[0]] / 35 - 1 # Vtg
		b = data[i][j[1]]           # Vds (should NOT shift, Pi-NN has not bias for tanh!!)
		c = abs(data[i][j[2]] * 0.4E7)
		if k == 7:
			print('t : ' + str(k))
			t.append([a, b, c])
		else: 
			print('d : ' + str(k))
			d.append([a, b, c])
		max = c if c > max else max
		min = c if c < min else min

print(max)
print(min)
savetxt('data/wse2_train.csv', d, delimiter=',') 
savetxt('data/wse2_test.csv', t, delimiter=',')

