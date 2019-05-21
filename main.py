import torch
import torchvision
import re
import time
import os
import math
import json
import base64
import pickle
import numpy as np
import sys


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

class newid:
	def __init__(self):
		self.idmap = {}
	def get(self, num):
		if (num not in self.idmap.keys()):
			self.idmap[num] = len(self.idmap)
		return self.idmap[num]
	def __len__(self):
		return len(self.idmap)

class regular:
	def __init__(self, min = 1e100, max = - 1e100):
		self.min = min
		self.max = max
	def input(self, num):
		if num < self.min:
			self.min = num
		if num > self.max:
			self.max = num
	def scale(self, num):
		delta = (num - 1) * (self.max - self.min) / 2
		self.max += delta
		self.min -= delta
	def changeminmax(self, min, max):
		self.min = min;
		self.max = max;
	def get(self, num):
		return (num - self.min) / (self.max - self.min)
		
	def put(self, num):
		return (self.max - self.min) * num + self.min

new0 = newid()
new2 = newid()
new3 = newid()
new4 = newid()
regx = regular()
regy = regular()

def readdata(filename):
	with open(filename) as f:
		lines = f.readlines()
	print('readfile done', time.clock())
	res = []
	ccccc = 0
	for line in lines:
		ccccc += 1
		if ccccc % 1000 == 0:
			print(ccccc, time.clock())
		line = re.findall(r'".*?"', line.strip())
		NA = None
		A = 'A'
		B = 'B'
		C = 'C'
		line = [eval(x) for x in line]
		if line[2] == '':
			line[2] = '-1'
		if line[3] == '':
			line[3] = '-1'
		for i in [0, 2, 3, 4, 5, 7, 8]:
			line[i] = eval(line[i])
		#print(line)
		res.append(line)
	for i in range(len(res)):
		res[i][0] = new0.get(res[i][0])
	for i in range(len(res)):
		if res[i][2] != -1:
			res[i][2] = new2.get(res[i][2])
	for i in range(len(res)):
		if res[i][3] != -1:
			res[i][3] = new3.get(res[i][3])
	for i in range(len(res)):
		res[i][4] = new4.get(res[i][4])
	for i in range(len(res)):
		for j in res[i][-1]:
			regx.input(j[0])
			regy.input(j[1])
	#regx.scale(1.1)
	#regy.scale(1.1)
	print(time.clock())
	return res

def regulazation(res):
	for i in range(len(res)):
		for j in range(len(res[i][-1])):
			res[i][-1][j][0] = regx.get(res[i][-1][j][0])
			res[i][-1][j][1] = regy.get(res[i][-1][j][1])
	
def meanHaversineDistance(a, b):
	#print(a, b)
	lat1 = regx.put(a[0])
	lon1 = regy.put(a[1])
	lat2 = regx.put(b[0])
	lon2 = regy.put(b[1])
	#print(lat1, lon1, lat2, lon2)
	REarth = 6371
	pi = 3.14159265358979323846
	lat = abs(lat1-lat2)*pi/180
	lon = abs(lon1-lon2)*pi/180
	lat1 = lat1*pi/180
	lat2 = lat2*pi/180
	a = math.sin(lat/2)*math.sin(lat/2)+math.cos(lat1)*math.cos(lat2)*math.sin(lon/2)*math.sin(lon/2)
	d = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
	d = REarth*d
	return d
	#print(regx.min, regx.max, regy.min, regy.max, d)
	#return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
	
time.clock()

parameters = json.loads(open('parameters.json').read())
if 'pointnum' not in parameters.keys():
	parameters['pointnum'] = 5
#parameters['traindata'] = 'train100000.csv'
#parameters['testnumber'] = -1

if parameters['seed'] != 0:
	torch.cuda.manual_seed_all(parameters['seed'])
	torch.manual_seed(parameters['seed'])

inputfilestr = parameters['datafolder'] + parameters['traindata'] + parameters['testdata'] + parameters['testgroundtruth'] + str(parameters['testnumber'])
inputbase64 = base64.b64encode(inputfilestr.encode('utf-8')).decode('utf-8')
parameters['traindata'] = parameters['datafolder'] + parameters['traindata']
parameters['testdata'] = parameters['datafolder'] + parameters['testdata']
parameters['testgroundtruth'] = parameters['datafolder'] + parameters['testgroundtruth']
if os.path.exists(parameters['datafolder'] + 'quick/' + inputbase64) and os.path.exists(parameters['datafolder'] + 'quick/raw' + inputbase64):
	with open(parameters['datafolder'] + 'quick/' + inputbase64, 'rb') as f:
		[traindata, trainlabel, testdata, testlabel, new0, new2, new3, new4, regx, regy] = pickle.load(f)
	with open(parameters['datafolder'] + 'quick/raw' + inputbase64, 'rb') as f:
		[dataraw, testraw] = pickle.load(f)
	print('load from saved data')
else:
	dataraw = readdata(parameters['traindata'])
	testraw = readdata(parameters['testdata'])
	regx.changeminmax(-8.699124600000001, -8.53519679180361)
	regy.changeminmax(41.0994612, 41.2110623998884)
	regulazation(dataraw)
	regulazation(testraw)
	with open(parameters['testgroundtruth']) as f:
		testlabelraw = [x.split(',')[1:] for x in f.readlines()[1:]]
	#print(testraw, testlabelraw)
	maxlength = 4000
	data = []
	label = []
	for i in dataraw:
		if i[7]:
			continue
		i = i[-1]
		if len(i) == 0:
			continue
		#for T in range(len(i), maxlength):
		#	i.append([-1, -1])
		data.append(torch.FloatTensor(i))
		#i.append([0, 0])
		#ll = [i[-1]] * len(i)
		#label.append(torch.FloatTensor(ll))
		label.append(torch.FloatTensor(i[-1]))
		
	testdata = []
	testlabel = []
	testnumber = parameters['testnumber']
	if testnumber == -1:
		testnumber = len(testraw)
	for i in range(len(testraw[:testnumber])):
		ii = testraw[i][-1]
		if len(ii) == 0:
			continue
		testdata.append(torch.FloatTensor(ii))
		testlabel.append(torch.FloatTensor([regx.get(float(testlabelraw[i][0])), regy.get(float(testlabelraw[i][1]))]))
	#print(testdata, testlabel)
	#os.system("pause")

	traindata = data
	trainlabel = label
	#trainsize = (len(data) // 10) * 9
	#traindata = data[:trainsize]
	#trainlabel = label[:trainsize]
	#testdata = data[trainsize:]
	#testlabel = label[trainsize:]
	with open(parameters['datafolder'] + 'quick/' + inputbase64, 'wb') as f:
		pickle.dump([traindata, trainlabel, testdata, testlabel, new0, new2, new3, new4, regx, regy], f)
	dataraw = [x[:-1] for x in dataraw]
	testraw = [x[:-1] for x in testraw]
	#print(dataraw[0], testraw[0], len(dataraw), len(testraw))
	#os.system('pause')
	with open(parameters['datafolder'] + 'quick/raw' + inputbase64, 'wb') as f:
		pickle.dump([dataraw, testraw], f)
#print(dataraw[0], testraw[0])
print(time.clock())

if len(sys.argv) > 1:
	if sys.argv[1] == 'c':
		parameters['continue'] = True
	if len(sys.argv) > 2:
		if float(sys.argv[2]) > 0:
			parameters['learningrate'] = [[0, float(sys.argv[2])]]
	if len(sys.argv) > 3:
		modelname = sys.argv[3]
		if modelname == 'rnn':
			parameters['model'] = 'RNNSimple'
		elif modelname == 'lstm':
			parameters['model'] = 'LSTMSimple'
		elif modelname == 'gru':
			parameters['model'] = 'GRUSimple'
		elif modelname == 'grumeta':
			parameters['model'] = 'GRUMeta'
		elif modelname == 'grubimeta':
			parameters['model'] = 'GRUBiMeta'
		elif modelname == 'mlp':
			parameters['model'] = 'MLPSimple'
		elif modelname == 'mlpmeta':
			parameters['model'] = 'MLPMeta'
		else:
			print('model name error')
			exit()
	
class RNNSimple(torch.nn.Module):
	def __init__(self, hidden = parameters['rnnhidden']):
		super(RNNSimple,self).__init__()
		self.hiddennum = hidden
		self.lstm = torch.nn.RNN(2, hidden)
		self.linear = torch.nn.Linear(hidden, 2)
		#self.hidden = torch.randn(1, 1, hidden)
	def forward(self, inputs, call, stand, taxi, hr, week):
		res, _ = self.lstm(inputs)#, self.hidden)
		#print(self.hidden.size())
		return self.linear(res)
		
class LSTMSimple(torch.nn.Module):
	def __init__(self, hidden = parameters['rnnhidden']):
		super(LSTMSimple,self).__init__()
		self.hiddennum = hidden
		self.lstm = torch.nn.LSTM(2, hidden)
		self.linear = torch.nn.Linear(hidden, 2)
		#self.hidden = torch.randn(1, 1, hidden)
	def forward(self, inputs, call, stand, taxi, hr, week):
		res, _ = self.lstm(inputs)#, self.hidden)
		#print(self.hidden.size())
		return self.linear(res)

class GRUSimple(torch.nn.Module):
	def __init__(self, hidden = parameters['rnnhidden']):
		super(GRUSimple,self).__init__()
		self.hiddennum = hidden
		self.gru = torch.nn.GRU(2, hidden)
		self.linear = torch.nn.Linear(hidden, 2)
		#self.hidden = torch.randn(1, 1, hidden)
	def forward(self, inputs, call, stand, taxi, hr, week):
		res, _ = self.gru(inputs)#, self.hidden)
		#print(self.hidden.size())
		return self.linear(res)
		
class GRUMeta(torch.nn.Module):
	def __init__(self, hidden = parameters['rnnhidden'], enum = parameters['embedding']):
		super(GRUMeta,self).__init__()
		self.hiddennum = hidden
		self.embeddingnum = enum
		self.gru = torch.nn.GRU(2, hidden)
		self.linear = torch.nn.Linear(hidden + enum * 5, 2)
		self.callemb = torch.nn.Embedding(len(new2) + 1, enum)
		self.standemb = torch.nn.Embedding(len(new3) + 1, enum)
		self.taxiemb = torch.nn.Embedding(len(new4), enum)
		self.hremb = torch.nn.Embedding(48, enum)
		self.weekemb = torch.nn.Embedding(7, enum)
		#self.hidden = torch.randn(1, 1, hidden)
	def forward(self, inputs, call, stand, taxi, hr, week):
		#print(inputs.size(), inputs.size(), call.size(), stand.size(), taxi.size(), hr.size(), week.size())
		res, _ = self.gru(inputs)#, self.hidden)
		call = self.callemb(call + 1)
		stand = self.standemb(stand + 1)
		taxi = self.taxiemb(taxi)
		hr = self.hremb(hr)
		week = self.weekemb(week) #[N, EMB]
		meta = torch.cat([call, stand, taxi, hr, week], 1) # [N, EMB * 5]
		meta = meta.unsqueeze(0).expand(res.size(0), meta.size(0), meta.size(1)).float() # [T, N, EMB * 5]
		#print(res.size(), meta.size())
		res = torch.cat([res, meta], 2)
		#print(self.hidden.size())
		return self.linear(res)
		
class GRUBiMeta(torch.nn.Module):
	def __init__(self, hidden = parameters['rnnhidden'], enum = parameters['embedding']):
		super(GRUBiMeta,self).__init__()
		self.hiddennum = hidden
		self.embeddingnum = enum
		self.gru = torch.nn.GRU(2, hidden, bidirectional = True)
		self.linear = torch.nn.Linear(hidden + enum * 5, 2)
		self.callemb = torch.nn.Embedding(len(new2) + 1, enum)
		self.standemb = torch.nn.Embedding(len(new3) + 1, enum)
		self.taxiemb = torch.nn.Embedding(len(new4), enum)
		self.hremb = torch.nn.Embedding(48, enum)
		self.weekemb = torch.nn.Embedding(7, enum)
		#self.hidden = torch.randn(1, 1, hidden)
	def forward(self, inputs, call, stand, taxi, hr, week):
		#print(inputs.size(), inputs.size(), call.size(), stand.size(), taxi.size(), hr.size(), week.size())
		res, _ = self.gru(inputs)#, self.hidden)
		res = (res[:, :, :self.hiddennum] + res[:, :, self.hiddennum:]) / 2
		#print(res.size())
		call = self.callemb(call + 1)
		stand = self.standemb(stand + 1)
		taxi = self.taxiemb(taxi)
		hr = self.hremb(hr)
		week = self.weekemb(week) #[N, EMB]
		meta = torch.cat([call, stand, taxi, hr, week], 1) # [N, EMB * 5]
		meta = meta.unsqueeze(0).expand(res.size(0), meta.size(0), meta.size(1)).float() # [T, N, EMB * 5]
		#print(res.size(), meta.size())
		res = torch.cat([res, meta], 2)
		#print(self.hidden.size())
		return self.linear(res)
		
class MLPSimple(torch.nn.Module):
	def __init__(self, pointnum = parameters['pointnum']):
		super(MLPSimple,self).__init__()
		self.pointnum = pointnum
		self.linear = torch.nn.Linear(pointnum * 2 * 2, 2)
	def forward(self, inputs, call, stand, taxi, hr, week):
		return self.linear(inputs)
		
class MLPMeta(torch.nn.Module):
	def __init__(self, enum = parameters['embedding'], pointnum = parameters['pointnum']):
		super(MLPMeta,self).__init__()
		self.embeddingnum = enum
		self.pointnum = pointnum
		self.callemb = torch.nn.Embedding(len(new2) + 1, enum)
		self.standemb = torch.nn.Embedding(len(new3) + 1, enum)
		self.taxiemb = torch.nn.Embedding(len(new4), enum)
		self.hremb = torch.nn.Embedding(48, enum)
		self.weekemb = torch.nn.Embedding(7, enum)
		self.linear = torch.nn.Linear(pointnum * 2 * 2 + 5 * enum, 2)
	def forward(self, inputs, call, stand, taxi, hr, week):
		#print(inputs.size(), inputs.size(), call.size(), stand.size(), taxi.size(), hr.size(), week.size())
		call = self.callemb(call + 1)
		stand = self.standemb(stand + 1)
		taxi = self.taxiemb(taxi)
		hr = self.hremb(hr)
		week = self.weekemb(week) #[N, EMB]
		res = torch.cat([inputs, call, stand, taxi, hr, week], 1) # [N, X + EMB * 5]
		return self.linear(res) # [N, 2]

class RNNData(torch.utils.data.Dataset):
	def __init__(self, x, y, meta):
		self.x = x
		self.y = y
		self.meta = [x[2:6] for x in meta]
	def __getitem__(self, index):
		return self.x[index], self.y[index], self.meta[index][0], self.meta[index][1], self.meta[index][2], self.meta[index][3]
	def __len__(self):
		return len(self.y)
		
	def collate_fn(self, data):
		X, y, call, stand, taxi, tt = list(zip(*data))
		
		# add padding
		batch_size = len(X)
		lens = [len(x) for x in X]
		max_len = max(lens)
		padded_X = torch.zeros([batch_size, max_len, 2]).float()
		for idx, x in enumerate(X):
			padded_X[idx, :len(x)] = x
		
		X = cuda(padded_X.transpose(0, 1)) # [T, N, 2]
		Y = cuda(torch.stack(y, dim = 0).unsqueeze(0).expand_as(X)) # [T, N, 2]
		lens = cuda(torch.tensor(lens).long()) # [N]
		call = cuda(torch.tensor(call).long())
		stand = cuda(torch.tensor(stand).long())
		taxi = cuda(torch.tensor(taxi).long())
		hr = []
		week = []
		for i in tt:
			[hr1, min1, week1] = time.strftime('%H,%M,%w', time.localtime(i)).split(',')
			hr.append(int(hr1) * 2 + int(min1) // 30)
			week.append(int(week1))
		hr = cuda(torch.tensor(hr))
		week = cuda(torch.tensor(week))
		return X, Y, lens, call, stand, taxi, hr, week
		
class MLPData(torch.utils.data.Dataset):
	def __init__(self, x, y, meta):
		self.x_raw = x
		self.y = y
		pointnum = parameters['pointnum']
		start = torch.stack([x[0].squeeze(0).expand(pointnum, 2) for x in self.x_raw]) # [N, pnum, 2]
		end = torch.stack([x[-1].squeeze(0).expand(pointnum, 2) for x in self.x_raw]) # [N, pnum, 2]
		for idx, x in enumerate(self.x_raw):
			xlen = min(pointnum, len(x))
			start[idx, -xlen:] = x[:xlen]
			end[idx, :xlen] = x[-xlen:]
		start.resize_(start.size(0), pointnum * 2) # [N, pnum * 2]
		end.resize_(start.size(0), pointnum * 2) # [N, pnum * 2]
		self.x = torch.cat([start, end], dim = 1)
		self.meta = [x[2:6] for x in meta]
	def __getitem__(self, index):
		return self.x[index], self.y[index], self.meta[index][0], self.meta[index][1], self.meta[index][2], self.meta[index][3]
	def __len__(self):
		return len(self.y)
	def collate_fn(delf, data):
		X, y, call, stand, taxi, tt = list(zip(*data))
		'''
		# add padding
		batch_size = len(X)
		pointnum = parameters['pointnum']
		lens = [len(x) for x in X]
		max_len = max(lens)
		start = torch.stack([x[0].squeeze(0).expand(pointnum, 2) for x in X]) # [N, pnum, 2]
		end = torch.stack([x[-1].squeeze(0).expand(pointnum, 2) for x in X]) # [N, pnum, 2]
		for idx, x in enumerate(X):
			xlen = min(pointnum, len(x))
			start[idx, -xlen:] = x[:xlen]
			end[idx, :xlen] = x[-xlen:]
		
		start.resize_(start.size(0), pointnum * 2) # [N, pnum * 2]
		end.resize_(start.size(0), pointnum * 2) # [N, pnum * 2]
		'''
		X = cuda(torch.stack(X, dim = 0)) # [N, pnum * 4]
		Y = cuda(torch.stack(y, dim = 0)) # [N, 2]
		call = cuda(torch.tensor(call).long())
		stand = cuda(torch.tensor(stand).long())
		taxi = cuda(torch.tensor(taxi).long())
		hr = []
		week = []
		for i in tt:
			[hr1, min1, week1] = time.strftime('%H,%M,%w', time.localtime(i)).split(',')
			hr.append(int(hr1) * 2 + int(min1) // 30)
			week.append(int(week1))
		hr = cuda(torch.tensor(hr))
		week = cuda(torch.tensor(week))
		return X, Y, call, stand, taxi, hr, week
		
class MSELoss(torch.nn.Module):
	def __init__(self):
		super(MSELoss, self).__init__()
	def forward(self, x, y, lengths):
		#import pdb
		#pdb.set_trace()
		#print(x.size(), y.size(), sequence_mask(lengths).size())
		a = x - y
		a = a * a * sequence_mask(lengths).transpose(0, 1).unsqueeze(2).expand_as(a)
		a = torch.mean(torch.sum(a, (0, 2)) / 2 / lengths.float())
		#print(x,y,a, lengths)
		#os.system('pause')
		#if a[0] > 1000 or not(a[0] == a[0]):
		#	import pdb
		#	pdb.set_trace()
		return a
		
class LinearMSELoss(torch.nn.Module):
	def __init__(self):
		super(LinearMSELoss, self).__init__()
	def forward(self, x, y):
		a = x - y
		a = a * a
		a = torch.mean(a)
		return a
	
lrvec = []
parameters['learningrate'].append([parameters['epochnumber'], 0])
for i in range(1, len(parameters['learningrate'])):
	for j in range(len(lrvec), parameters['learningrate'][i][0]):
		lrvec.append(parameters['learningrate'][i - 1][1])
#print(lrvec)
def rnnlikemodel():
	batchsize = parameters['batchsize']
	train_dataset = RNNData(traindata, trainlabel, dataraw)
	test_dataset = RNNData(testdata, testlabel, testraw)
	train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batchsize, collate_fn = train_dataset.collate_fn, shuffle = True)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 100, collate_fn = test_dataset.collate_fn)#, shuffle = True)
	model = cuda(globals()[parameters['model']]())
	print('model name:', parameters['model'])
	loss = MSELoss()
	#learningrate = 0.1
	#parameters['epochnumber'] = 30
	lastres = 0

	if 'continue' in parameters.keys() and parameters['continue']:
		if os.path.exists('model.pkl'):
			model.load_state_dict(torch.load('model.pkl'))
		print('load model')

	for epoch in range(parameters['epochnumber']):
		learningrate = lrvec[epoch]
		eruntime = time.clock()
		opt = torch.optim.Adam(model.parameters(), lr = learningrate)
		print('epoch', epoch, 'lr', learningrate)
		btime = 0
		model.train()
		#getdata = []
		#maxlen = 0
		for input, label, lengths, call, stand, taxi, hr, week in train_loader:
			opt.zero_grad()
			pred = model(input, call, stand, taxi, hr, week)
			l = loss(pred, label, lengths)
			if (btime % (len(traindata) // 10 // batchsize) == 0):
				print(btime, l.data.item())
			#l.backward(retain_graph=True)
			l.backward()
			opt.step()
			getdata = []
			maxlen = 0
			btime += 1
		model.eval()
		torch.save(model.state_dict(), 'model.pkl')
		res = 0
		for input, label, lengths, call, stand, taxi, hr, week in test_loader:
			#print(pred.size(), label.size(), lengths, call.size(), stand.size(), taxi.size(), hr.size(), week.size())
			pred = model(input, call, stand, taxi, hr, week)
			#print(pred[-1][0], label[-1][0])
			for i in range(len(label[0])):
				res += meanHaversineDistance(pred[lengths[i] - 1][i], label[0][i])
			#l = loss(pred, label, len(input))
			#res += l.data.item()# ** 0.5
		lastres = res / len(testdata)
		print(lastres, 'use time:', time.clock() - eruntime)
		
def linearmodel():
	batchsize = parameters['batchsize']
	train_dataset = MLPData(traindata, trainlabel, dataraw)
	test_dataset = MLPData(testdata, testlabel, testraw)
	train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batchsize, collate_fn = train_dataset.collate_fn, shuffle = True)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 100, collate_fn = test_dataset.collate_fn)#, shuffle = True)
	model = cuda(globals()[parameters['model']]())
	print('model name:', parameters['model'])
	loss = LinearMSELoss()
	#learningrate = 0.1
	#parameters['epochnumber'] = 30
	lastres = 0

	if 'continue' in parameters.keys() and parameters['continue']:
		if os.path.exists('model.pkl'):
			model.load_state_dict(torch.load('model.pkl'))
		print('load model')

	for epoch in range(parameters['epochnumber']):
		learningrate = lrvec[epoch]
		eruntime = time.clock()
		opt = torch.optim.Adam(model.parameters(), lr = learningrate)
		print('epoch', epoch, 'lr', learningrate)
		btime = 0
		model.train()
		#getdata = []
		#maxlen = 0
		for input, label, call, stand, taxi, hr, week in train_loader:
			opt.zero_grad()
			pred = model(input, call, stand, taxi, hr, week)
			l = loss(pred, label)
			if (btime % (len(traindata) // 10 // batchsize) == 0):
				print(btime, l.data.item())
			#l.backward(retain_graph=True)
			l.backward()
			opt.step()
			getdata = []
			maxlen = 0
			btime += 1
		model.eval()
		torch.save(model.state_dict(), 'model.pkl')
		res = 0
		for input, label, call, stand, taxi, hr, week in test_loader:
			#print(pred.size(), label.size(), lengths, call.size(), stand.size(), taxi.size(), hr.size(), week.size())
			pred = model(input, call, stand, taxi, hr, week)
			#print(pred[-1][0], label[-1][0])
			for i in range(len(label)):
				res += meanHaversineDistance(pred[i], label[i])
			#l = loss(pred, label, len(input))
			#res += l.data.item()# ** 0.5
		lastres = res / len(testdata)
		print(lastres, 'use time:', time.clock() - eruntime)
		
if parameters['model'][:3] == 'MLP':
	linearmodel()
else:
	rnnlikemodel()