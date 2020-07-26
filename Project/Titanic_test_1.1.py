# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:22:46 2020

@author: SungHyun
"""

# =============================================================================
# Start Date    : 20-7-18
# End Date      : ing
# Writer        : Joe SungHyun - dyddnjswh
# Discription   : First Data Science Project
#                 1.1 Titanic Data Pridiction Survivor
# Reference     : 1. https://kaggle-kr.tistory.com/17#1_1 - Main
#                 2. https://hamait.tistory.com/342 - 문자열 처리
#                 3. https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling - 캐글 참조
#                 4. http://scikit-learn.org/stable/supervised_learning.html#supervised-learning - sklearn document
# =============================================================================

# 추가 필요사항: 좀 더 다양한 DATA 전처리 학습 및 분석, 다양한 모델 공부 및 적절 모델 사용, 선택한 모델의 parameter 튜닝 시도
# 모델 예측부터는 코드의 의미 확인 필요. accuracy 등등.
# 모델 결과를 보고도 feature selection이나 feature 제거를 하는 등의 과정이 있을 수도 있데
# 좀 더 참신한 feature engineering, 머신 러닝 모델 hyperparameter tunning, ensembling 등, 무궁무진합니다..
# 꾸준히 커널공부를 하시면 실력이 꾸준히 늘겁니다.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)

# import missingno as msno

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline


#파일 불러오기.
df_train = pd.read_csv('../Kaggle_Titanic/Data/train.csv')
df_test  = pd.read_csv('../Kaggle_Titanic/Data/test.csv')

#파일 확인
df_train.head()

#각 feature의 통계치 반환
df_train.describe()
df_test.describe()

# =============================================================================
# 여기까지 Null 데이터와 각 열의 입력된 개수?가 다른게 있음을 확인 가능.
# =============================================================================

#Null Data chect
print('train null check')
for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)

print('\ntest null check')

for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)
    
#Target Label 확인 >> 여기서는 Survive
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()

# =============================================================================
# 나름 Target label이 균등하게 분포가 되어 있음. 후처리 괜찮을듯?
# =============================================================================

# =============================================================================
# 각 Feature 확인.
# =============================================================================

#Pclass
#crosstab 표가 그려기지고, margins = True을 넣으면 합이 나오네? .style.background_gradient(cmap = 'summer_r')은 스타일
# pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True).style.background_gradient(cmap = 'summer_r')
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True)

#Pclass 별 group 평균은 Survived 기준으로 // ascending은 오름 내림 차순 정렬.
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()

#seaborn 활용해서 시각화
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()

# =============================================================================
# Pclass 영향이 큰 것을 확인 할 수 있음. 따라서 Feature에 활용.
# =============================================================================

#Sex
f, ax = plt.subplots(1, 2, figsize=(18,8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index = True).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('Sex: Suvrvied vs Dead')
plt.show()

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True)

# =============================================================================
# Sex 역시 영향이 큰 것을 확인 할 수 있음. 따라서 Feature에 활용.
# =============================================================================

#Sex & Pclass
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
sns.factorplot('Sex', 'Survived', col='Pclass', data=df_train, size=6, aspect=1.5)

# =============================================================================
# 서로 상관관계는 없어보이고 앞서 각자 봤던 대로 나오는 것 같네
# =============================================================================

#Age
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))
print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))
print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
df_train['Age'][df_train['Survived'] == 1].plot(kind='kde')
#sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()

plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])

#연령대 별로 나누어서 확인을 해보는 것도 괜찮을 것 같은데.
#일단 나이별로 확인을 해봅시다. 데이터가 충분한지도 확인이 필요한데..

cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))
    
plt.figure(figsize=(7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()

# =============================================================================
# 나이도 역시 영향이 있네. 어릴수록 살 확률이 높네. 근데 이건 Pclass와 약간 부딪히는 건가..? 나이 많은 Pclass가 높고 높으며 생존율이 높은데...?
# =============================================================================

#Sex & Pclass & Age
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()

# =============================================================================
# 여자와 어린아이를 먼저 살렸다.
# =============================================================================

#Embarked
f,ax=plt.subplots(2, 2, figsize=(20,15))
sns.countplot('Embarked', data=df_train, ax=ax[0,0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# =============================================================================
# 크게 영향은 없어 보인다. 
# =============================================================================

#Family
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

print("Maximum size of Family: ", df_train['FamilySize'].max())
print("Minimum size of Family: ", df_train['FamilySize'].min())

f,ax=plt.subplots(1, 3, figsize=(40,10))
sns.countplot('FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# =============================================================================
# 너무 많아도 너무 적어도 좋지 않네.
# =============================================================================

#Fare
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')

#Fare는 비대칭성이 심하네 >> 이러면 outlier에 굉장히 민감할 수 있어. 따라서 log를 취함??
# 아래 줄은 뒤늦게 발견하였습니다. 13번째 강의에 언급되니, 일단 따라치시고 넘어가면 됩니다.
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
plt.show()
#Cabin은 너무 Nan이 많아서 제외

#Ticket
# =============================================================================
# 연습을 해보라는데??
# =============================================================================


# =============================================================================
# Feature Engineering >> train / test에 동일하게 적용을 해줘야함.
# =============================================================================

# 1. Fill Null >> 어떻게 채우냐에 따라 성능에 차이가 있음.
#Null Data chect
print('train null check')
for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)

print('\ntest null check')

for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)
    
# Age Null 채우기

print('Age Null 개수: {:.2f}\n'.format(df_train['Age'].isnull().sum()))

#앞에 붙는 단어를 통해서 나이 채우기 
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
    
df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

pd.crosstab(df_train['Initial'], df_train['Sex']).T

#이름들 치환하기
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_train.groupby('Initial').mean()
df_train.groupby('Initial')['Survived'].mean().plot.bar()
# 역시 Miss Mrs가 생존확률이 높네

#Null을 채우는 방법은 여러개가 있데. 머신러닝 알고리즘을 사용할 수도 있고 여러 방법이 있지만 여기서는 statictcs방법을 써서 사용할 예정.
print('Age Null 채우기 및 확인')

Init_Kind = ['Mr', 'Mrs','Master','Miss','Other']

for strTmp in Init_Kind:
    df_train.loc[(df_train.Age.isnull())&(df_train.Initial== strTmp),'Age'] = round(df_train.loc[(df_train.Initial == strTmp), 'Age'].mean())
    df_test.loc[(df_test.Age.isnull())&(df_test.Initial== strTmp),'Age'] = round(df_test.loc[(df_test.Initial == strTmp), 'Age'].mean())

print('Train Age Null 개수: {:.2f}\n'.format(df_train['Age'].isnull().sum()))
print('Test Age Null 개수: {:.2f}\n'.format(df_train['Age'].isnull().sum()))

# =============================================================================
# Age Null 처리 완료.
# =============================================================================

#Embarked 처리 S가 가장 많았으므로 S로 모두 처리

print('처리 전: Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
df_train['Embarked'].fillna('S', inplace=True)
print('처리 후: Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')



# # Age를 카테고리화. 자칫 data loss를 읽을 수 있음. 일단은 빼놓음. 두 가지 방법 소개.
# # 1. 그냥 때려넣기
# df_train['Age_cat'] = 0
# df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0
# df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1
# df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2
# df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3
# df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4
# df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5
# df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6
# df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7

# df_test['Age_cat'] = 0
# df_test.loc[df_test['Age'] < 10, 'Age_cat'] = 0
# df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1
# df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2
# df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3
# df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4
# df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5
# df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6
# df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7

# # 2. 함수화
# def category_age(x):
#     if x < 10:
#         return 0
#     elif x < 20:
#         return 1
#     elif x < 30:
#         return 2
#     elif x < 40:
#         return 3
#     elif x < 50:
#         return 4
#     elif x < 60:
#         return 5
#     elif x < 70:
#         return 6
#     else:
#         return 7    
    
# df_train['Age_cat_2'] = df_train['Age'].apply(category_age)

# # 1번과 2번 확인
# print('1번 방법, 2번 방법 둘다 같은 결과를 내면 True 줘야함 -> ', (df_train['Age_cat'] == df_train['Age_cat_2']).all())

# # 확인했으니 중복되는 것 제거
# df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)
# df_test.drop(['Age'], axis=1, inplace=True)




# String으로 된 것을 수치화 시키기 >> Name, Embarked, Sex

#수치에 무슨 값이 있는 확인
df_train['Initial'].unique()
df_train['Initial'].value_counts()

df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_train['Embarked'].unique()
df_train['Embarked'].value_counts()

df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train['Embarked'].isnull().any()

df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})


# =============================================================================
# Pearson correlation을 통한 상관관계 확인 -1, 1사이 0에 가까울 수록 관계없음. 수식은 1번 reference 참조 
# =============================================================================

#HeatMap으로 표시
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Initial']] 

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

del heatmap_data

# =============================================================================
# 앞서 하나씩 봤을 때와 같이 Sex 와 Pclass가 Survived와 상관관계가 높다. Embarked도 값이 있는데 맞을까?? 다 S로 채워넣어서..
# 또 서로 상관관계가 큰 feature들은 없어 보인다.
# =============================================================================

# # =============================================================================
# # Data PreProcessing 
# # =============================================================================

# # 1. One-hot encoding >> 수치화 시켰어도 성등을 높이기 위해서 시행해 줄 수 있음.
# # >> get_dummies or sklearn 로 Labelencoder + OneHotencoder 이용

# df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
# df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')

# df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
# df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

# df_train.head()

# # =============================================================================
# # 필요없는 데이터들 삭제
# # =============================================================================
# df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
# df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
# df_train.head()

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'FamilySize', 'Initial'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'FamilySize', 'Initial'], axis=1, inplace=True)
df_train.head()


# =============================================================================
# 머신러닝 모델 생성하기. >> sklearn 활용
# =============================================================================
#importing all the required ML packages
from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 
from sklearn import metrics # 모델의 평가를 위해서 씁니다
from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.

# =============================================================================
# 지금 타이타닉 문제는 target class(survived)가 있으며, target class 는 0, 1로 이루어져 있으므로(binary) binary classfication 문제입니다.
# =============================================================================

# 1. 학습에 쓰일 데이터와, target label(Survived)를 분리
X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values

# 보통 train, test 만 언급되지만, 실제 좋은 모델을 만들기 위해서 우리는 valid set을 따로 만들어 모델 평가를 해봅니다.

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)

# =============================================================================
# 본 튜토리얼에서는 랜덤포레스트 모델 사용.
# 랜덤포레스트는 결정트리기반 모델이며, 여러 결정 트리들을 앙상블한 모델임. 
# 각 머신러닝 알고리즘에는 여러 파라미터들이 있습니다. 
# 랜덤포레스트분류기도 n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf 등 여러 파라미터들이 존재합니다. 
# 이것들이 어떻게 세팅되냐에 따라 같은 데이터셋이라 하더라도 모델의 성능이 달라집니다.
# 파라미터 튜닝은 시간, 경험, 알고리즘에 대한 이해 등이 필요합니다. 결국 많이 써봐야 모델도 잘 세울 수 있는 것이죠.
# 그래서 캐글을 추천합니다. 여러 데이터셋을 가지고 모델을 이리저리 써봐야 튜닝하는 감이 생길테니까요!
# 일단 지금은 튜토리얼이니 파라미터 튜닝은 잠시 제쳐두기로 하고, 기본 default 세팅으로 진행하겠습니다.
# 모델 객체를 만들고, fit 메소드로 학습시킵니다.
# 그런 후 valid set input 을 넣어주어 예측값(X_vld sample(탑승객)의 생존여부)를 얻습니다.
# =============================================================================

# # =============================================================================
# # Model generation 
# # =============================================================================
# model = RandomForestClassifier()
# model.fit(X_tr, y_tr)
# prediction = model.predict(X_vld)

# print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))

# # =============================================================================
# # 영향 받은 featureimportances 확인 
# # =============================================================================

# # 학습된 모델은 기본적으로 featureimportances 를 가지고 있어서 쉽게 그 수치를 얻을 수 있습니다. pandas series 활용

# from pandas import Series

# feature_importance = model.feature_importances_
# Series_feat_imp = Series(feature_importance, index=df_test.columns)

# plt.figure(figsize=(8, 8))
# Series_feat_imp.sort_values(ascending=True).plot.barh()
# plt.xlabel('Feature importance')
# plt.ylabel('Feature')
# plt.show()

# # Keras 활용
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout
# from keras.optimizers import Adam, SGD

# nn_model = Sequential()
# nn_model.add(Dense(32,activation='relu',input_shape=(4,))) #14
# nn_model.add(Dropout(0.2))
# nn_model.add(Dense(64,activation='relu'))
# nn_model.add(Dropout(0.2))
# nn_model.add(Dense(32,activation='relu'))
# nn_model.add(Dropout(0.2))
# nn_model.add(Dense(1,activation='sigmoid'))

# Loss = 'binary_crossentropy'
# nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])
# nn_model.summary()

# history = nn_model.fit(X_tr,y_tr,
#                     batch_size=64,
#                     epochs=500,
#                     validation_data=(X_vld, y_vld),
#                     verbose=1)

# hists = [history]
# hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
# hist_df.index = np.arange(1, len(hist_df)+1)
# fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
# axs[0].plot(hist_df.val_accuracy, lw=5, label='Validation Accuracy')
# axs[0].plot(hist_df.accuracy, lw=5, label='Training Accuracy')
# axs[0].set_ylabel('Accuracy')
# axs[0].set_xlabel('Epoch')
# axs[0].grid()
# axs[0].legend(loc=0)
# axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')
# axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')
# axs[1].set_ylabel('MLogLoss')
# axs[1].set_xlabel('Epoch')
# axs[1].grid()
# axs[1].legend(loc=0)
# fig.savefig('hist.png', dpi=300)
# plt.show();

# # Pytorch 활용 - CPU 활용
# import torch
# import torch.nn as nn
# import torch.optim as optim

# class Model(nn.Module):

#         def __init__(self):
#             super(Model, self).__init__()

#             # Inputs = 5, Outputs = 3, Hidden = 30
#             self.linear_1 = nn.Linear(4, 32)
#             self.linear_2 = nn.Linear(32, 64)
#             self.linear_3 = nn.Linear(64, 32)
#             self.out = nn.Linear(32, 1)
           
#         def forward(self, x):
#             x = self.linear_1(x)
#             x = nn.ReLU()(x)
#             x = nn.Dropout()(x)
#             x = self.linear_2(x)
#             x = nn.ReLU()(x)
#             x = nn.Dropout()(x)
#             x = self.linear_3(x)
#             x = nn.ReLU()(x)
#             x = nn.Dropout()(x)
#             x = self.out(x)
#             y = nn.Sigmoid()(x)
#             return y

# def accuracy(pred, y):
#     preds = pred.round()
#     return (preds == y).float().mean()
    
# model = Model()

# # criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()


# optimizer = optim.Adam(model.parameters(), lr=0.001)

# X_tr = torch.from_numpy(X_tr)
# X_vld = torch.from_numpy(X_vld)
# y_tr = torch.from_numpy(y_tr)
# y_vld = torch.from_numpy(y_vld)

# for t in range(500):
#     y_pred = model(X_tr.float())

#     loss = criterion(y_pred.float(), y_tr.float())
    
#     optimizer.zero_grad()
    
#     loss.backward()
    
#     optimizer.step()
    
#     # model.eval()
#     # y_hat = model(X_vld.float())
#     # model.train()
    
#     acc = accuracy(y_pred, y_tr)
#     print("{:2} epoch - cross entropy : {:4.2}, accuracy : {:4.2}".format(
#         t + 1, loss.detach().item(), acc.detach().item()))
    
#     # print(loss, y_vld.data[0], y_hat.data[0,0])    
    
#     # prediction = y_pred.data.max(1)[1]   # first column has actual prob.
#     # accuracy = prediction.eq(X_vld.data).sum()/1*100
#     # train_accu.append(accuracy)
#     # print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(t, loss.data[0], accuracy))

# Pytorch 활용 - CUDA 사용
import torch
import torch.nn as nn
import torch.optim as optim

# class Model(nn.Module):

#         def __init__(self):
#             super(Model, self).__init__()

#             # Inputs = 5, Outputs = 3, Hidden = 30
#             self.linear_1 = nn.Linear(4, 32)
#             self.linear_2 = nn.Linear(32, 64)
#             self.linear_3 = nn.Linear(64, 32)
#             self.out = nn.Linear(32, 1)
           
#         def forward(self, x):
#             x = self.linear_1(x)
#             x = nn.ReLU()(x)
#             x = nn.Dropout()(x)
#             x = self.linear_2(x)
#             x = nn.ReLU()(x)
#             x = nn.Dropout()(x)
#             x = self.linear_3(x)
#             x = nn.ReLU()(x)
#             x = nn.Dropout()(x)
#             x = self.out(x)
#             y = nn.Sigmoid()(x)
#             return y

def accuracy(pred, y):
    preds = pred.round()
    return (preds == y).float().mean()

model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(32, 1),
    nn.Sigmoid(),
).cuda()

torch.cuda.is_available()

cuda = torch.device('cuda')

# model = Model()

# if torch.cuda.is_available():
#     model = model.cuda()

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tr = torch.from_numpy(X_tr)
X_vld = torch.from_numpy(X_vld)
y_tr = torch.from_numpy(y_tr)
y_vld = torch.from_numpy(y_vld)
X_test = torch.from_numpy(X_test)

if torch.cuda.is_available():
    X_tr = X_tr.cuda()
    X_vld = X_vld.cuda()
    y_tr = y_tr.cuda()
    y_vld = y_vld.cuda()
    X_test = X_test.cuda()
epochs = 500;

for t in range(epochs):
    y_pred = model(X_tr.float())

    loss = criterion(y_pred.float(), y_tr.float())
    
    # optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()    
    model.eval()
    y_hat = model(X_vld.float())
    model.train()
    acc = accuracy(y_pred, y_tr)
    print("{:2} epoch - cross entropy : {:4.2}, accuracy : {:4.2}".format(
        t + 1, loss.detach().item(), acc.detach().item()))
    
    # prediction = y_pred.data.max(1)[1]   # first column has actual prob.
    # accuracy = prediction.eq(X_vld.data).sum()/1*100
    # train_accu.append(accuracy)
    # print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(t, loss.data[0], accuracy))
 
# # =============================================================================
# # 제출 준비.Submission 양식에 맞게?
# # =============================================================================

# submission = pd.read_csv('../Kaggle_Titanic/Data/sample_submission.csv')
# submission.head()

# model.eval()
# prediction = model(X_test.float())
# submission['Survived'] = prediction.cpu().detach().numpy()

# submission.to_csv('../Kaggle_Titanic/Data/GPU_submission.csv', index=False)    
    
 
# =============================================================================
# 사실 현재 feature importance 는 지금 모델에서의 importance 를 나타냅니다. 
# 만약 다른 모델을 사용하게 된다면 feature importance 가 다르게 나올 수 있습니다.
# 이 feature importance 를 보고 실제로 Fare 가 중요한 feature 일 수 있다고 판단을 내릴 수는 있지만, 
# 이것은 결국 모델에 귀속되는 하나의 결론이므로 통계적으로 좀 더 살펴보긴 해야합니다.
# featuure importance 를 가지고 좀 더 정확도가 높은 모델을 얻기 위해 feature selection 을 할 수도 있고, 
# 좀 더 빠른 모델을 위해 feature 제거를 할 수 있습니다.
# =============================================================================

# =============================================================================
# 제출 준비.Submission 양식에 맞게?
# =============================================================================

# submission = pd.read_csv('../Kaggle_Titanic/Data/sample_submission.csv')
# submission.head()

# prediction = model.predict(X_test)
# submission['Survived'] = prediction

# submission.to_csv('../Kaggle_Titanic/Data/my_first_submission.csv', index=False)