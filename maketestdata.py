import re

import numpy as np

filename = 'data/test10000.csv'
with open(filename) as f:
	lines = f.readlines()
print('readfile done')
res = []
ccccc = 0
testf = open('data/testmain.csv', 'w')
answerf = open('data/testmaingroundtruth.csv', 'w')
answerf.write('"TRIP_ID","LATITUDE","LONGITUDE"\n')
for line in lines:
	#print(len(lines), lines[:10])
	line = re.findall(r'".*?"', line.strip())
	#print(len(line), line)
	l = eval(eval(line[8]))
	if (len(l) == 0):
		continue
	tail = l[-1]

	minlen = 0.2
	maxlen = 0.8

	nowlen = np.random.rand() * (maxlen - minlen) + minlen
	nowlen = int(len(l) * nowlen) + 1
	#print(len(l), nowlen)

	if (len(l) > 1):
		l = l[:nowlen]
	s = '"['
	for i in l:
		if len(s) != 2:
			s += ','
		s += '[' + str(i[0]) + ',' + str(i[1]) + ']'
	s += ']"'
	line[8] = s
	testf.write(','.join(line) + '\n')
	answerf.write(line[0] + ',' + str(tail[0]) + ',' + str(tail[1]) + '\n')
