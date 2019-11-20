import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#クラスタリングして、グラフに描画する
#2層のAE(16,6)(6,2)による2次元ベクトル
#当然だがラベリングはわからないものとする


#データ読み込み
#train5.pyで変換して用意した2次元ベクトル
f=open("ae8888.pkl","rb")
x=pickle.load(f)
f.close()


# クラスタリング、4つ
km=KMeans(n_clusters=4,init='random',n_init=1,max_iter=100)

#グラフ描画
cluster=km.fit_predict(x)
plt.scatter(x[cluster==0,0],x[cluster==0,1],marker='.')
plt.scatter(x[cluster==1,0],x[cluster==1,1],marker='.')
plt.scatter(x[cluster==2,0],x[cluster==2,1],marker='.')
plt.scatter(x[cluster==3,0],x[cluster==3,1],marker='.')
print(cluster)
plt.show()
