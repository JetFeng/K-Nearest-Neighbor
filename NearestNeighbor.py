#coding=utf-8
import numpy as np
import os

class NearestNeighbor():
    def __init__(self):
        pass

    def train(self,x,y):
        # x 是 N*D 的向量，每一行代表一个实例，y是一个N*1的向量，每一行代表实例类别
        self.Xtr = x
        self.ytr = y

    def predict(self,x):
        num_test = x.shape[0]
        Ypred = np.zeros(num_test,dtype = self.ytr.dtype)

        for i in xrange(num_test):
            #曼哈顿距离
            distances = np.sum(self.Xtr - x[i,:],axis = 1) #numpy广播机制，数组的每一行减去后一个一维数组
            #欧拉距离
            #distances = np.sqrt(np.sum(np.square(self.Xtr -x[i,:]),axis = 1))
            min_index = np.argmin(distances)  #argmin返回最小值的数据所在索引
            Ypred[i] = self.ytr[min_index]
            print i
        return Ypred

def load_CIFAR10(dir):
    import cPickle
    train_batch_files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    test_batch_files = ['test_batch']

    Xtr = None
    Ytr = None
    for i in train_batch_files:
        file = os.path.join(dir,i)
        fo = open(file,'rb')
        dict = cPickle.load(fo)
        if(Xtr is None and Ytr is None):
            Xtr = np.array(dict['data'])
            Ytr = np.array(dict['labels'])
        else:
            Xtr = np.vstack((Xtr, np.array(dict['data'])))
            Ytr = np.hstack((Ytr, np.array(dict['labels'])))

    Ytr = Ytr.reshape((Ytr.shape[0],1))


    Xte = None
    Yte = None
    for i in test_batch_files:
        file = os.path.join(dir,i)
        fo = open(file,'rb')
        dict = cPickle.load(fo)
        if(Xte is None and Yte is None):
            Xte = np.array(dict['data'])
            Yte = np.array(dict['labels'])
        else:
            Xte = np.vstack((Xte, np.array(dict['data'])))
            Yte = np.hstack((Yte, np.array(dict['labels'])))

    Yte = Yte.reshape((Yte.shape[0],1))
    return Xtr,Ytr,Xte,Yte

#载入CIFAR10数据集
Xtr,Ytr,Xte,Yte = load_CIFAR10('data\cifar-10-batches-py')

Xtr_rows = Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0],32*32*3)

nn = NearestNeighbor() #创建NN分类器
nn.train(Xtr_rows,Ytr)
Yte_predict = nn.predict(Xte_rows)

print 'accuracy: %f'%(np.mean(Yte_predict == Yte))

