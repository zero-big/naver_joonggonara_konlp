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
import sys
# -----------------------  Step 1. 데이터 정제화
# df1 = pd.read_csv('crawling_data/data_concat.csv')
# df1["제목"] = df1["제목"].str.replace(pat=r'[^가-힣a-zA-Z ]', repl=r'', regex=True, )
# df1.to_csv("./crawling_data/new_data/clear_data_45.csv", index=False)

pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv("crawling_data/data_concat.csv")
df["제목"] = df["제목"].str.replace(pat=r'[^가-힣a-zA-Z ]', repl=r'', regex=True, )
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
print(df.head())
df.info()
#
X = df['제목']  #입력
Y = df['분류']  #타겟 9개
print(X.isna().sum())

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_  # classes에 카테고리를 리스트화 하여 저장을 함.
print(label) #나중에 데이터 분석할때 0, 0, 0, 1, 0, 0,이면 3번 인덱싱인 정치(politics)로 데이터를 확인하는 것을 보기 위해 필요함
with open('models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)   # 계속 지정이 될 수 있도록 모델 폴더 안에 엔커도 피클로 저장을 함.

onehot_Y = to_categorical(labeled_Y)
print('onehot',onehot_Y)

#형태소를 분리하는 패키지는 여러가지가 있음.(Kkma,komoran)
okt = Okt()  #한글자료들을 패키지화 할때 사용함. #okt가 자바로 만들어져서 자바를 실행할수 있는 것이 필요함


for i in range(len(X)): X[i] = okt.morphs(X[i], stem=True)
# okt_morph_X = okt.morphs(X[i], stem=True) #한글 단어의 형태소 별로 딕셔너리로 변환하여 가지게 함. 나중에 그 딕셔너리와 비교하여 정보를 도출해냄
# stem 옵션을 주면 동사를 원형으로 바꿔줌.

# 자연어 처리에 필요 없는 불용어는 제거해야 함(조사 등)
stopwords = pd.read_csv('./models/stopwords.csv', index_col=0)

for i in range(len(X)):
    words = []
    # X 속 모든 리스트 요소의 j번째 요소
    for j in range(len(X[i])):
        # 한 글자 이상인 형태소만 추출
        if len(X[i][j]) > 1:
            # 불용어 제거
            if X[i][j] not in list(stopwords['stopword']):
                words.append(X[i][j])
    X[i] = ' '.join(words)

#토큰화
token = Tokenizer()  #텐서플로우에 있는 패키지 # 같은 단어는 같은 숫자로 라벨링을 해줌.
token.fit_on_texts(X)  #  범위 안의 단어들로 딕셔너리를 만들어줌
tokened_X = token.texts_to_sequences(X)
print(tokened_X)
print(token.word_index)  # 딕셔너리로 바꾼 단어들의 인덱스를 확인하는 방법
wordsize = len(token.word_index)+ 1 #나중에 데이터 양을 넣어야 함으로 총 단어수를 알기 위한 방법// 0을 추가했으므로 +1해주어야함.
print(wordsize)

with open('./models/joonggonara_titles_token.pickle', 'wb') as f:
    pickle.dump(token, f)
#자료로 사용하기 위해선 딕셔너리(리스트)의 데이터수가 같아야 하는데 다들 각자 다르므로 가장 긴 딕셔너리에 맞춰서 데이터 '0'을 넣어줌
#데이터 분석을 하는동안 뒤쪽 데이터를 더 중요하게 받아드리므로 '0'은 제일 앞에 넣어주어야 함.
#제일 긴 문장 찾기
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)
#맥스의 길이만큼 앞에 0으로 채워넣기, 텐서플로우 패키지 안에 있음.
X_pad = pad_sequences(tokened_X, max)
print(X_pad)


X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size=0.25)

print(X_train.shape, Y_train.shape)
print(X_test.shape, X_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./models/modeling_data_max_{}_wordsize_{}'.format(max, wordsize), xy)


