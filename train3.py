import numpy as np
import random
# import vec2train
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import pickle
import matplotlib.pyplot as plt
import data_create

#8次元を2次元に圧縮するオートエンコーダー
#クラスごとに色付けてグラフに描画、ラベリングがわかっている


#AE(8,2)
class MyAE(Chain):
    def __init__(self):
        super(MyAE,self).__init__(
            #ネットワーク
            l1=L.Linear(8,2),
            l2=L.Linear(2,8),
        )

    def __call__(self,x):
        #誤差
        out=self.fwd(x)
        return F.mean_squared_error(out,x)

    def fwd(self,x):
        #接続
        mid=F.sigmoid(self.l1(x))
        result=self.l2(mid)
        return result

data = 100
#2個ずつマイナス、ベクトルデータ生成、4パターン
x=[]
for a in range(int(data/4)):
   x.append(data_create.minus12_create())
for b in range(int(data/4)):
    x.append(data_create.minus34_create())
for c in range(int(data/4)):
    x.append(data_create.minus56_create())
for d in range(int(data/4)):
    x.append(data_create.minus78_create())

#初期化、設定
model=MyAE()
optimizer=optimizers.SGD()
optimizer.setup(model)

x=Variable(np.array(x,dtype=np.float32))

#学習
for a in range(10000):
    model.cleargrads()
    loss=model(x)
    loss.backward()
    optimizer.update()

#2次元ベクトル
y=F.sigmoid(model.l1(x))
ans=y.data


#グラフに描画
x_data_1=[]
y_data_1=[]
x_data_2=[]
y_data_2=[]
x_data_3=[]
y_data_3=[]
x_data_4=[]
y_data_4=[]

for a in range(25):
    #25ずつ4パターン
    x_data_1.append(ans[a][0])
    y_data_1.append(ans[a][1])
    x_data_3.append(ans[a+25][0])
    y_data_3.append(ans[a+25][1])
    x_data_4.append(ans[a+75][0])
    y_data_4.append(ans[a+75][1])
    x_data_2.append(ans[a+50][0])
    y_data_2.append(ans[a+50][1])
print(ans)

plt.scatter(x_data_1,y_data_1,marker="+")
plt.scatter(x_data_2,y_data_2,marker=".")
plt.scatter(x_data_3,y_data_3,marker="_")
plt.scatter(x_data_4,y_data_4,marker="^")

plt.show()

#データ保存
f=open("ae88.pkl","wb")
pickle.dump(ans,f)
f.close()
