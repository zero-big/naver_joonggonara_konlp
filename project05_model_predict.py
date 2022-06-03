# -----------------------  Step 0. 준비
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Kkma
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

# -----------------------  Step 1. 데이터 정제화
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 33)
df = pd.read_csv('     ')
# print(df.head())
# df.info()

X = df['제목']
Y = df['분류']
# print(X)
with open('models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

labeled_Y = encoder.transform(Y)  # 기존의 픽클로 라벨링을 하게되니 fit을 하면 안됨.
# print(labeled_Y[:3]) # 카테고리 인덱싱 숫자를 확인.(인덱싱 될때에는 카테고리 이름의 영어 오름차순)
label = encoder.classes_  # classes에 카테고리를 리스트화 하여 저장을 함.
# print(label) #나중에 데이터 분석할때 0, 0, 0, 1, 0, 0,이면 3번 인덱싱인 정치(politics)로 데이터를 확인하는 것을 보기 위해 필요함

onehot_Y = to_categorical(labeled_Y)

#형태소를 분리하는 패키지는 여러가지가 있음.(Kkma,komoran)
okt = Okt()  #한글자료들을 패키지화 할때 사용함. #okt가 자바로 만들어져서 자바를 실행할수 있는 것이 필요함


for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
#한글 단어의 형태소 별로 딕셔너리로 변환하여 가지게 함. 나중에 그 딕셔너리와 비교하여 정보를 도출해냄
#stem 옵션을 주면 동사를 원형으로 바꿔줌.


# 자연어 처리에 필요 없는 불용어는 제거해야 함(조사 등)
stopwords = pd.read_csv('models/stopwords.csv', index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:  # 한글자도 불용어 처리를 하기 위해 길이가 1초과로 명령을 해야함.
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

with open('test1/joonggo_token.pickle', 'rb') as f:
    token = pickle.load(f)


tokened_X = token.texts_to_sequences(X)
#----------------------------------------
#새로 크롤링한 기사가 기존 토큰보다 글자수가 길면 안됨으로 글자수를 맞춰줌
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 32:
        tokened_X[i] = tokened_X[i][:32]/

X_pad = pad_sequences(tokened_X, 32)

# -----------------------  Step 2. 테스트 데이터 검증
model = load_model('models/joonggonara_learning.h5')
preds = model.predict(X_pad)
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts

# 검증데이터 OX화
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, '분류'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'

print(df['OX'].value_counts()) #OX갯수
print(df['OX'].value_counts()/len(df))  #OX확률

# ox데이터 저장
df2=[]
for i in range(len(df)):
    if df['OX'][i] == 'X':     #df['category'][i] =df.loc[i, 'category']
       df1 = df.iloc[i]
       df2.append(df1)# print(type(df1))
df2 = pd.DataFrame(df2)
df2.to_csv("./wrong_data.csv", index=False)
