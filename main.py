import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import klda
from scipy import exp
import torch
import torchvision
import random
import pickle
rand = random.randint

folder = 'mnist/'
trainimgfilename = 'train-images.idx3-ubyte'
trainlabelfilename = 'train-labels.idx1-ubyte'
testimgfilename = 't10k-images.idx3-ubyte'
testlabelfilename = 't10k-labels.idx1-ubyte'

def showimage(arr, x, y):
	plt.imshow(arr, cmap='binary')
	plt.show()
	return
	outputchar = " .-+=UODGEHKBXWM";
	for i in range(x):
		for j in range(y):
			print(outputchar[int(arr[i][j] / 16)], end = '')
		print()

def read_images(filename):
	binfile = open(filename, 'rb')
	buf = binfile.read()
	index = 0
	magic, img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
	print('read images', magic, img_num, numRows, numColums)
	index += struct.calcsize('>IIII')
	img_list = np.zeros((img_num, 28 * 28))
	for i in range(img_num):
		im = struct.unpack_from('>784B', buf, index)
		index += struct.calcsize('>784B')
		im = np.array(im)
		im = im.reshape(1, 28 * 28)
		img_list[ i , : ] = im
		
		im = im.reshape(28, 28)
		
		#showimage(im, 28, 28)
		
	return img_num, img_list

def read_labels(filename):
	binfile = open(filename, 'rb')
	buf = binfile.read()
	index = 0
	magic, label_num = struct.unpack_from('>II', buf, index)
	print('read labels', magic, label_num)
	index += struct.calcsize('>II')
	
	label_list = []
	for i in range(label_num):
		label_item = int(struct.unpack_from('>B', buf, index)[0])
		label_list.append(int(label_item))
		index += struct.calcsize('>B')
	
	return label_num, label_list

def RBFKernel(x, y, gamma = 1e-7):
	#sum = 0
	#for i in range(len(x)):
	#	sum += (x[i] - y[i]) ** 2
	return exp( - np.sum(np.square(x - y)) * gamma)

def kernel(x, y):
	return np.dot(x, y)
	
def ridge(train_x, train_y, test_x, test_y, kernel = None, ori0 = [0], ori1 = [1]):
	ridge = Ridge(alpha=float('{}'.format(0.5))).fit(train_x, train_y)
	result = []
	for x in ridge.predict(test_x):
		if x < 0.5:
			result.append(0)
		else:
			result.append(1)
	return result

def KLDAtest(train_x, train_y, test_x, test_y, kernel, ori0 = [0], ori1 = [1]):
	test = [[], []]
	for i in range(len(test_x)):
		if test_y[i] < 2:
			test[test_y[i]].append(test_x[i])
	output = []
	for gamma in [1e-8, 3e-8, 5e-8, 7e-8, 9e-8, 1e-7, 3e-7, 5e-7, 7e-7, 9e-7, 1e-6]:
	#gamma = 1e-7
		print(gamma)
		accr = 0
		tot = 0
		alpha = klda.klda(train_x, train_y, gamma)
		trainres = [0, 0]
		trainnum = [0, 0]
		for i in range(len(train_x)):
			tmp = np.zeros((len(train_x),))
			for j in range(len(train_x)):
				tmp += kernel(train_x[j], train_x[i], gamma)
			trainres[train_y[i]] += np.dot(alpha.T, tmp)
			trainnum[train_y[i]] += 1.0
		trainres[0] /= trainnum[0]
		trainres[1] /= trainnum[1]
		decision = (trainres[0] + trainres[1]) / 2
		if (trainres[1] > decision):
			flag1 = 1
		else:
			flag1 = 0
		result = 0
		for i in range(len(test_x)):
			if test_y[i] < 2:
				tmp = np.zeros((len(train_x),))
				for j in range(len(train_x)):
					tmp += kernel(train_x[j], test_x[i], gamma)
				y = np.dot(alpha.T, tmp) - decision
				if y > 0:
					y = flag1
				else:
					y = 1 - flag1
				if y == test_y[i]:
					result += 1
		output.append([gamma, result])
	print(output)	
	input()

def KLDA(train_x, train_y, test_x, test_y, kernel, ori0 = [0], ori1 = [1]):
	M = [[], []]
	gamma = 3e-7
	alpha = klda.klda(train_x, train_y, gamma)
	trainres = [0, 0]
	trainnum = [0, 0]
	for i in range(len(train_x)):
		tmp = np.zeros((len(train_x),))
		for j in range(len(train_x)):
			tmp += kernel(train_x[j], train_x[i], gamma)
		trainres[train_y[i]] += np.dot(alpha.T, tmp)
		trainnum[train_y[i]] += 1.0
	trainres[0] /= trainnum[0]
	trainres[1] /= trainnum[1]
	decision = (trainres[0] + trainres[1]) / 2
	if (trainres[1] > decision):
		flag1 = 1
	else:
		flag1 = 0
	result = []
	accu = 0
	all = 0
	for i in range(len(test_x)):
		tmp = np.zeros((len(train_x),))
		for j in range(len(train_x)):
			tmp += kernel(train_x[j], test_x[i], gamma)
		y = np.dot(alpha.T, tmp) - decision
		if y > 0:
			y = flag1
		else:
			y = 1 - flag1
		if y == 0 and test_y[i] in ori0 or y == 1 and test_y[i] in ori1:
			accu += 1
		if test_y[i] in ori0 or test_y[i] in ori1:
			all += 1
		result.append(y)
	#print(result)
	#print(accu, all)
	return result
	
def KPerceptron(train_x, train_y, test_x, test_y, kernel, ori0 = 0, ori1 = 1):
	n = [0] * len(train_x)
	for i in range(len(train_y)):
		if train_y[i] == 0:
			train_y[i] = -1
	pre = []
	for i in train_x:
		tmp = []
		for j in train_x:
			tmp.append(kernel(i, j))
		pre.append(tmp)
	pre = np.array(pre)
	mintime = 1000000 // len(train_x)
	maxtime = 10000000 // len(train_x)
	for T in range(maxtime):
		if T + 1 % mintime == 0:
			accu = 0
			tot = 0
			for i in range(len(test_y)):
				if test_y[i] in ori0 or test_y[i] in ori1:
					tot += 1
					if (result[i] == 0 and test_y[i] in ori0) or (result[i] == 1 and test_y[i] in ori1):
						accu += 1
			#print(accu / tot, len(train_x))
			if accu > 0.95:
				break
		pick = rand(0, len(train_x) - 1)
		y = 0
		for i in range(len(train_x)):
			y += n[i] * train_y[i] * pre[pick][i]
		if (y == 0 or (y > 0) != (train_y[pick] > 0)):
			#print(pick, y, train_y[pick])
			#input()
			n[pick] += 1
	result = []
	for pick in range(len(test_x)):
		y = 0
		for i in range(len(train_x)):
			y += n[i] * train_y[i] * kernel(test_x[pick], train_x[i])
		if y > 0:
			result.append(1)
		elif y < 0:
			result.append(0)
		else:
			result.append(rand(0, 1))
	
	accu = 0
	tot = 0
	for i in range(len(test_y)):
		if test_y[i] in ori0 or test_y[i] in ori1:
			tot += 1
			if (result[i] == 0 and test_y[i] in ori0) or (result[i] == 1 and test_y[i] in ori1):
				accu += 1
	#print(accu / tot, len(train_x))
	
	return result

def LR(train_x, train_y, test_x, test_y):
	reg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
	reg.fit(train_x, train_y)
	result = reg.predict(test_x)
	return accuracy_score(result, test_y), result
	
def SVM(train_x, train_y, test_x, test_y):
	a = train_x[0]
	train_x = [[int(y // 128) for y in x] for x in train_x]
	#print(list(zip(train_x[0], a)))
	svc = SVC()
	svc.fit(train_x, train_y)
	test_x = [[y // 128 for y in x] for x in test_x]
	result = svc.predict(test_x)
	#print(svc.predict([train_x[0]]), train_y[0])
	return accuracy_score(result, test_y), result
	
def MLP1(train_x, train_y, test_x, test_y, loss):
	train_x = torch.FloatTensor(train_x)
	train_y = torch.LongTensor(train_y)
	test_x = torch.FloatTensor(test_x)
	class NN(torch.nn.Module):
		def __init__(self, input, mid, output):
			super(NN, self).__init__()
			self.fc1 = torch.nn.Linear(input, mid)
			self.relu = torch.nn.ReLU()
			self.fc2 = torch.nn.Linear(mid, output)
		def forward(self, x):
			tmp = self.fc1(x)
			tmp = self.relu(tmp)
			tmp = self.fc2(tmp)
			return tmp
	model = NN(28 * 28, 80, 10)
	
	class data(torch.utils.data.Dataset):
		def __init__(self, x, y):
			self.x = x
			self.y = y
		def __getitem__(self, index):
			return self.x[index], self.y[index]
		def __len__(self):
			return self.y.size(0)
	train = data(train_x, train_y)
	bs = 100
	tx, ty = train[0]
	train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = bs, shuffle = True)
	
	opt = torch.optim.Adam(model.parameters(), lr = 0.0001)
	for epoch in range(10000):
		#print('epoch', epoch)
		time = 0
		for input, label in train_loader:
			opt.zero_grad()
			pred = model(input)
			l = loss(pred, label)
			l.backward()
			opt.step()
			time += 1
			#if (time % 49 == 0):
			#	print(time, l.data.item())
	
	res = model(test_x)
	result = []
	for i in range(len(res)):
		max = 0
		for j in range(10):
			if res[i][j] > res[i][max]:
				max = j
		result.append(max)
	
	return accuracy_score(result, test_y), result
	
def MLP2(train_x, train_y, test_x, test_y, loss):
	train_x = torch.FloatTensor(train_x)
	tmp = []
	for i in train_y:
		tt = [0] * 10
		tt[i] = 1
		tmp.append(tt)
	train_y = tmp
	train_y = torch.FloatTensor(train_y)
	test_x = torch.FloatTensor(test_x)
	class NN(torch.nn.Module):
		def __init__(self, input, mid, output):
			super(NN, self).__init__()
			self.fc1 = torch.nn.Linear(input, mid)
			self.relu = torch.nn.ReLU()
			self.fc2 = torch.nn.Linear(mid, output)
			self.softmax = torch.nn.Softmax()
		def forward(self, x):
			tmp = self.fc1(x)
			tmp = self.relu(tmp)
			tmp = self.fc2(tmp)
			tmp = self.softmax(tmp)
			return tmp
	model = NN(28 * 28, 80, 10)
	
	class data(torch.utils.data.Dataset):
		def __init__(self, x, y):
			self.x = x
			self.y = y
		def __getitem__(self, index):
			return self.x[index], self.y[index]
		def __len__(self):
			return self.y.size(0)
	train = data(train_x, train_y)
	bs = 100
	tx, ty = train[0]
	train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = bs, shuffle = True)
	
	opt = torch.optim.Adam(model.parameters(), lr = 0.0001)
	for epoch in range(10000):
		#print('epoch', epoch)
		time = 0
		for input, label in train_loader:
			opt.zero_grad()
			pred = model(input)
			l = loss(pred, label)
			l.backward()
			opt.step()
			time += 1
			#if (time % 100 == 0):
			#	print(time, l.data.item())
	
	res = model(test_x)
	result = []
	for i in range(len(res)):
		max = 0
		for j in range(10):
			if res[i][j] > res[i][max]:
				max = j
		result.append(max)
	
	return accuracy_score(result, test_y), result
	
def CNN(train_x, train_y, test_x, test_y, loss):
	train_x = torch.FloatTensor(train_x)
	train_x.resize_((len(train_x), 1, 28, 28))
	train_y = torch.LongTensor(train_y)
	test_x = torch.FloatTensor(test_x)
	test_x.resize_((len(test_x), 1, 28, 28))
	class NN(torch.nn.Module):
		def __init__(self):
			super(NN, self).__init__()
			self.conv1 = torch.nn.Conv2d(1, 6, 5, padding = 2)
			self.conv2 = torch.nn.Conv2d(6, 16, 5)
			self.fc1 = torch.nn.Linear(400, 120)
			self.fc2 = torch.nn.Linear(120, 84)
			self.fc3 = torch.nn.Linear(84, 10)
			self.relu = torch.nn.ReLU()
			self.softmax = torch.nn.Softmax()
		def forward(self, x):
			tmp = torch.nn.functional.max_pool2d(self.conv1(x), (2, 2))
			tmp = torch.nn.functional.max_pool2d(self.conv2(tmp), (2, 2))
			tmp = tmp.view((-1, 400))
			tmp = self.fc3(self.relu(self.fc2(self.relu(self.fc1(tmp)))))
			tmp = self.softmax(tmp)
			return tmp
	model = NN()
	
	class data(torch.utils.data.Dataset):
		def __init__(self, x, y):
			self.x = x
			self.y = y
		def __getitem__(self, index):
			return self.x[index], self.y[index]
		def __len__(self):
			return self.y.size(0)
	train = data(train_x, train_y)
	bs = 100
	tx, ty = train[0]
	train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = bs, shuffle = True)
	
	opt = torch.optim.Adam(model.parameters(), lr = 0.0001)
	for epoch in range(1000):
		print('epoch', epoch)
		time = 0
		for input, label in train_loader:
			opt.zero_grad()
			pred = model(input)
			l = loss(pred, label)
			l.backward()
			opt.step()
			time += 1
			if (time % 100 == 0):
				print(time, l.data.item())
	
	res = model(test_x)
	result = []
	for i in range(len(res)):
		max = 0
		for j in range(10):
			if res[i][j] > res[i][max]:
				max = j
		result.append(max)
	
	return accuracy_score(result, test_y), result
	
def onevsone(classnum, classifier, traindata, trainlabel, testdata, testlabel, kernel = None):
	train = [[] for i in range(10)]
	for i in range(len(traindata)):
		train[trainlabel[i]].append(traindata[i])
	vote = [[0] * classnum for i in range(len(testdata))]
	for i in range(classnum):
		print('doing', i)
		for j in range(classnum):
			if i != j:
				tmpdata = train[i] + train[j]
				tmplabel = [0] * len(train[i]) + [1] * len(train[j])
				if kernel == None:
					tmpres = classifier(tmpdata, tmplabel, testdata, testlabel, [i], [j])
				else:
					tmpres = classifier(tmpdata, tmplabel, testdata, testlabel, kernel, [i], [j])
				for k in range(len(tmpres)):
					if tmpres[k] == 0:
						vote[k][i] += 1
					else:
						vote[k][j] += 1
	res = []
	#print(vote)
	for i in range(len(testdata)):
		tmp = []
		nowmax = -1
		for j in range(classnum):
			if vote[i][j] > nowmax:
				nowmax = vote[i][j];
				tmp = [j]
			elif vote[i][j] == nowmax:
				tmp.append(j)
		res.append(tmp)
	accu = 0.0
	for i in range(len(testlabel)):
		if testlabel[i] in res[i]:
			accu += 1.0 / len(res[i])
	accu /= len(testlabel)
	return accu, res

def onevsall(classnum, classifier, traindata, trainlabel, testdata, testlabel, kernel = None):
	res = []
	for i in range(len(testdata)):
		res.append([])
	for i in range(classnum):
		print('doing', i)
		tmplabel = []
		for j in range(len(trainlabel)):
			if trainlabel[j] == i:
				tmplabel.append(0)
			else:
				tmplabel.append(1)
		jlist = []
		for j in range(classnum):
			if j != i:
				jlist.append(j)
		if kernel == None:
			tmpres = classifier(traindata, tmplabel, testdata, testlabel, [i], jlist)
		else:
			tmpres = classifier(traindata, tmplabel, testdata, testlabel, kernel, [i], jlist)
		for k in range(len(tmpres)):
			if tmpres[k] == 0:
				res[k].append(i)
	accu = 0.0
	for i in range(len(testlabel)):
		if testlabel[i] in res[i]:
			accu += 1.0 / len(res[i])
	accu /= len(testlabel)
	#print(accu)
	return accu, res
'''
#read data
trainnum, trainimg = read_images(folder + trainimgfilename)
tmp, trainlabel = read_labels(folder + trainlabelfilename)
testnum, testimg = read_images(folder + testimgfilename)
tmp, testlabel = read_labels(folder + testlabelfilename)

#decrease training set
trainnum = 1000
trainimg = trainimg[:trainnum]
trainlabel = trainlabel[:trainnum]

testnum = 1000
testimg = testimg[:testnum]
testlabel = testlabel[:testnum]
'''
'''
ridgeaccu, ridgeres = onevsone(10, ridge, trainimg, trainlabel, testimg, testlabel)
print('ridge:', ridgeaccu)
KLDAaccu, KLDAres = onevsone(10, KLDA, trainimg, trainlabel, testimg, testlabel, RBFKernel)
print('kernel LDA:', KLDAaccu)
KPaccu, KPres = onevsall(10, KPerceptron, trainimg, trainlabel, testimg, testlabel, RBFKernel)
print('kernel perceptron:', KPaccu)
LRaccu, LRres = LR(trainimg, trainlabel, testimg, testlabel)
print('logistic regression:', LRaccu)
SVMaccu, SVMres = SVM(trainimg, trainlabel, testimg, testlabel)
print('SVM:', SVMaccu)
MLP1accu, MLP1res = MLP1(trainimg, trainlabel, testimg, testlabel, torch.nn.CrossEntropyLoss())
print('MLP with CrossEntrophyLoss:', MLP1accu)
MLP2accu, MLP2res = MLP2(trainimg, trainlabel, testimg, testlabel, torch.nn.KLDivLoss())
print('MLP with Softmax + KLDivLoss:', MLP2accu)
CNNaccu, CNNres = CNN(trainimg, trainlabel, testimg, testlabel, torch.nn.CrossEntropyLoss())
print('CNN:', CNNaccu)
'''

#IMDB data & GRU

with open('imdb/imdb.pkl', 'rb') as f:
	[imdbinx, imdbiny] = pickle.load(f)
with open('imdb/imdb.dict.pkl', 'rb') as f:
	dict = pickle.load(f)
with open('imdb/result.pkl', 'rb') as f:
	onehot2vec = pickle.load(f)
numlist = list(range(len(imdbinx)))
random.shuffle(numlist)
imdbx = []
imdby = []
for i in numlist:
	imdbx.append(imdbinx[i])
	imdby.append(imdbiny[i])
trainnum = len(imdbx) // 10 * 9
trainx = imdbx[:trainnum]
trainy = imdby[:trainnum]
testx = imdbx[trainnum:]
testy = imdby[trainnum:]

def cuda(tensor):
    """A cuda wrapper
    """
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor
		
def sequence_mask(sequence_length, max_length=None):
    """
    e.g., sequence_length = "5,7,8", max_length=None
    it will return
    tensor([[ 1,  1,  1,  1,  1,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  1]], dtype=torch.float32)

    :param sequence_length: a torch tensor
    :param max_length: if not given, it will be set to the maximum of `sequence_length`
    :return: a tensor with dimension  [*sequence_length.size(), max_length]
    """
    if len(sequence_length.size()) > 1:
        ori_shape = list(sequence_length.size())
        sequence_length = sequence_length.view(-1) # [N, ?]
        reshape_back = True
    else:
        reshape_back = False

    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long() # [max_length]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length) # [batch, max_len], repeats on each column
    seq_range_expand = torch.autograd.Variable(seq_range_expand).to(sequence_length.device)
    #if sequence_length.is_cuda:
    #    seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand) # [batch, max_len], repeats on each row

    ret = (seq_range_expand < seq_length_expand).float() # [batch, max_len]

    if reshape_back:
        ret = ret.view(ori_shape + [max_length])

    return ret

class GRU(torch.nn.Module):
	def __init__(self, dictsize, hidden = 300):
		super(GRU,self).__init__()
		self.hiddennum = hidden
		self.gru = torch.nn.GRU(dictsize, hidden)
		self.linear = torch.nn.Linear(hidden, 1)
	def forward(self,inputs, lengths): # [T, N, 300]
		res, _ = self.gru(inputs)# [T, N, hidden]
		#print(self.hidden.size())
		mask = sequence_mask(lengths).transpose(0, 1).unsqueeze(2).expand_as(res)
		res = res * mask
		res = torch.sum(res, dim = 0) # [N, hidden]
		res = res / lengths.float().unsqueeze(1).expand_as(res)
		#print(res.size(), lengths.unsqueeze(1))
		return self.linear(res)# [N, hidden] -> [N]

class Data(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __getitem__(self, index):
		resx = [onehot2vec[x] for x in self.x[index]]
		#print(index, self.y[index])
		return torch.FloatTensor(resx), torch.FloatTensor([self.y[index]])
	def __len__(self):
		return len(self.y)
		
	def collate_fn(self, data):
		X, y = list(zip(*data))
		#import pdb
		#pdb.set_trace()
		#print(X, y)
		#print(y)
		
		# add padding
		batch_size = len(X)
		lens = [len(x) for x in X]
		max_len = max(lens)
		padded_X = torch.zeros([batch_size, max_len, 300]).float()
		for idx, x in enumerate(X):
			padded_X[idx, :len(x)] = x
		
		X = cuda(padded_X.transpose(0, 1)) # [T, N, 300]
		Y = cuda(torch.stack(y, dim = 0)) # [N, 1]
		lens = cuda(torch.tensor(lens).long()) # [N]
		#print(lens)
		#os.system("pause")
		#print('---', X.size(), Y.size(), lens.size())
		return X, Y, lens
		
class MSELoss(torch.nn.Module):
	def __init__(self):
		super(MSELoss, self).__init__()
	def forward(self, x, y, lengths):
		#print(x, y, lengths)
		a = x - y
		a = a * a
		a = torch.mean(a)
		return a
		
batchsize = 128
dictsize = len(dict)
train_dataset = Data(trainx, trainy)
test_dataset = Data(testx, testy)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batchsize, collate_fn = train_dataset.collate_fn)#, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batchsize, collate_fn = test_dataset.collate_fn)#, shuffle = True)

model = cuda(GRU(300))
loss = MSELoss()
oldloss = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(10):
	print('epoch', epoch)
	time = 0
	model.train()
	for input, label, lengths in train_loader:
		opt.zero_grad()
		pred = model(input, lengths)
		#print(input[0][0], label[0][0], lengths, pred[0][0])
		#os.system('pause')
		#import pdb
		#pdb.set_trace()
		l = loss(pred, label, lengths)
		if (time % (len(trainx) // 10 // batchsize) == 0):
			print(time, l.data.item())
		#l.backward(retain_graph=True)
		l.backward()
		opt.step()
		getdata = []
		maxlen = 0
		time += 1
	res = 0
	model.eval()
	for input, label, lengths in test_loader:
		pred = model(input, lengths)
		#print(pred[-1][0], label[-1][0])
		for i in range(len(lengths)):
			if (label[i] >= 0.5) == (pred[i] >= 0.5):
				res += 1
		#res += torch.sum(l).data.item()
		#l = loss(pred, label, len(input))
		#res += l.data.item()# ** 0.5
	print(res / len(testx))