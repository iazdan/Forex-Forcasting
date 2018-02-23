import numpy as np
from sklearn.preprocessing import scale, StandardScaler
class data_preparation():
	def __init__(self, seq_length=30):
		print 'The data is being created...'
		self.T = seq_length

		tr_set, tr_labels, val_set , val_labels, ts_set, ts_labels = self.read_split_data()
		self.tr_set = tr_set
		self.tr_labels = tr_labels
		self.val_set = val_set
		self.val_labels = val_labels
		self.ts_set = ts_set
		self.ts_labels = ts_labels

	def create_labeled_seq(self, dataset, gbpusd):
		n, m = np.shape(dataset)
		seq = np.zeros((n-self.T,m*self.T), dtype=np.float64)
		labels = np.zeros(n-self.T)
		
		for i in range(n-self.T):
			seq[i,:] = dataset[i:i+self.T,:].ravel()
			labels[i] = gbpusd[self.T+i,-1]
		return seq, labels

	def read_split_data(self):
		directory = '/home/yazdan/Desktop/Fall_2016/ML/ANN/Forexdata/'
		light = np.loadtxt(directory+'light.csv', delimiter=',', dtype=np.float64)
		gold = np.loadtxt(directory+'gold.csv', delimiter=',', dtype=np.float64)
		usd = np.loadtxt(directory+'usd.csv', delimiter=',', dtype=np.float64)
		ftse100 = np.loadtxt(directory+'ftse100.csv', delimiter=',', dtype=np.float64)
		#data = np.array([np.concatenate((ftse100[i], light[i], gold[i], usd[i]))  for i in range(len(gold))])
		#data=usd
		#5 columns
		#usd=scale(usd)
		u=usd[:,4]-usd[:,1]
		#usd[:,1]=u
		#usd[:,2]=u
		#usd[:,3]=u
		#usd[:,4]=u
		
		#4 columns
		'''u=usd[:,0]-usd[:,3]
		usd[:,0]=u
		usd[:,1]=u
		usd[:,2]=u
		usd[:,3]=u'''

		usd=np.column_stack((usd[:,0],u))
		data=np.column_stack((usd[:,0],u))#usd
		
		
		
		n_tr = int(0.6 * len(usd))  # ... percent of the data is used for trainign
		n_val = int(0.2 * len(usd))
		n_te = len(usd) - (n_tr + n_val)
		
		train_set, train_labels = self.create_labeled_seq(data[:n_tr,:], usd[:n_tr,:])
		valid_set , valid_labels= self.create_labeled_seq(data[n_tr:(n_tr+n_val),:], usd[n_tr:(n_tr+n_val),:])
		test_set, test_labels =  self.create_labeled_seq(data[(n_tr+n_val):,:], usd[(n_tr+n_val):,:])
	
		
	
		#normalization	using standarde normal distribution
		'''tr_mean, tr_std = np.mean(train_set, axis=0), np.std(train_set, axis=0)
		labels_mean, labels_std = np.mean(train_labels, axis=0), np.std(train_labels, axis=0)
		train_set, train_labels = 1.0 * (train_set - tr_mean)/tr_std, 1.0 * (train_labels -labels_mean)/labels_std

		valid_set , valid_labels= 1.0 * (valid_set - tr_mean)/tr_std, 1.0 * (valid_labels -labels_mean)/labels_std
		test_set, test_labels =  1.0 * (test_set - tr_mean)/tr_std, 1.0 * (test_labels -labels_mean)/labels_std'''
		

		'''#Normalization within interval [0,1]
		
		tr_max, tr_min = np.max(train_set, axis=0), np.min(train_set, axis=0)
		labels_max, labels_min = np.max(train_labels, axis=0), np.min(train_labels, axis=0)
		train_set, train_labels = 1.0 * (train_set - tr_min)/(tr_max - tr_min), 1.0 * (train_labels -labels_min)/(labels_max - labels_min)

		valid_set, valid_labels = 1.0 * (valid_set - tr_min)/(tr_max - tr_min), 1.0 * (valid_labels -labels_min)/(labels_max - labels_min)
		test_set, test_labels = 1.0 * (test_set - tr_min)/(tr_max - tr_min), 1.0 * (test_labels -labels_min)/(labels_max - labels_min)'''
		

		#Normalization within interval [-1,1]
		
		'''tr_max, tr_min = np.max(train_set, axis=0), np.min(train_set, axis=0)
		labels_max, labels_min = np.max(train_labels, axis=0), np.min(train_labels, axis=0)
		train_set, train_labels = 2.0 * (train_set - tr_min)/(tr_max - tr_min) - 1, 2.0 * (train_labels -labels_min)/(labels_max - labels_min) - 1

		valid_set, valid_labels = 2.0 * (valid_set - tr_min)/(tr_max - tr_min) - 1, 2.0 * (valid_labels -labels_min)/(labels_max - labels_min) - 1
		test_set, test_labels = 2.0 * (test_set - tr_min)/(tr_max - tr_min) - 1, 2.0 * (test_labels -labels_min)/(labels_max - labels_min) - 1'''

		del data
		return train_set, train_labels, valid_set , valid_labels, test_set, test_labels

		
				
		
		
	
	
