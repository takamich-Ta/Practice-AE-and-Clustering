import random
import pickle

#数字の羅列、ベクトル生成
#様々なデータセット作る


#1-10の数をランダム生成、8次元配列
def random8_create():
    vec = []
    for a in range(8):
        vec.append(random.randint(1, 10))
    return vec

#1-10の数をランダム生成、16次元配列
def random16_create():
    vec = []
    for a in range(16):
        vec.append(random.randint(1, 10))
    return vec

#8次元配列生成して、3,4だけマイナス
def minus34_create():
    a=random8_create()
    a[2]=a[2]*-1
    a[3]=a[3]*-1
    return a

#8次元配列生成して、1,2だけマイナス
def minus12_create():
    a=random8_create()
    a[0]=a[0]*-1
    a[1]=a[1]*-1
    return a

#8次元配列生成して、5,6だけマイナス
def minus56_create():
    a=random8_create()
    a[4]=a[4]*-1
    a[5]=a[5]*-1
    return a

#8次元配列生成して、7,8だけマイナス
def minus78_create():
    a=random8_create()
    a[6]=a[6]*-1
    a[7]=a[7]*-1
    return a

#16次元ベクトル、1234マイナス
def minus1234_create():
    a=random16_create()
    a[0]=a[0]*-1
    a[1] = a[1] * -1
    a[2] = a[2] * -1
    a[3]=a[3]*-1
    return a

# 16次元ベクトル、5678マイナス
def minus5678_create():
    a=random16_create()
    a[4]=a[4]*-1
    a[5] = a[5] * -1
    a[6] = a[6] * -1
    a[7]=a[7]*-1
    return a

#16次元、9101112マイナス
def minus9101112_create():
    a=random16_create()
    a[8]=a[8]*-1
    a[9] = a[9] * -1
    a[10] = a[10] * -1
    a[11]=a[11]*-1
    return a

#16次元、13141516マイナス
def minus13141516_create():
    a=random16_create()
    a[12]=a[12]*-1
    a[13] = a[13] * -1
    a[14] = a[14] * -1
    a[15]=a[15]*-1
    return a
