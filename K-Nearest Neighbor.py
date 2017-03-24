import numpy as np
import os
from collections import Counter


class KNearestNeighbors():
    def __init__(self):
        pass

    def train(self,x,y):
        self.Xtr = x
        self.Ytr = y

    def predict(self,x,k):
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in xrange(num_test):
            # 曼哈顿距离
            distances = np.sum(self.Xtr - x[i, :], axis=1)  # numpy广播机制，数组的每一行减去后一个一维数组
            # 欧拉距离
            # distances = np.sqrt(np.sum(np.square(self.Xtr -x[i,:]),axis = 1))
            min_index = 0
            pred = []
            for i in range(k):
                 min_index = np.argmin(distances >= min_index)  # argmin返回最小值的数据所在索引
                 min_label = self.Ytr[min_index]
                 pred.append(min_label)

            Ypred[i] = Counter(pred).most_common(1)  #统计出现次数最多的类别作为最终预测类别

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

Xval_rows = Xtr_rows[:1000,:]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:,:]
Ytr = Ytr[1000:]

validation_accuracies = []
for k in [1,3,5,10,20,50,100]:
    knn = KNearestNeighbors()
    knn.train(Xtr_rows,Ytr)
    Yval_predict = knn.predict(Xval_rows,k = k)
    acc = np.mean(Yval_predict ==Yval)
    print 'accuracy: %f' % (acc,)
    validation_accuracies.append((k,acc))