import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

books = pd.read_csv(r'C:\Users\User\Desktop\tugas no3\Ujian_MachineLearning_JCDS04-master\Dataset_2\books.csv')
ratings = pd.read_csv(r'C:\Users\User\Desktop\tugas no3\Ujian_MachineLearning_JCDS04-master\Dataset_2\ratings.csv')
# books.columns.values()
# books
# books.corr()
books.loc[books['original_title'].isnull(), 'original_title'] = books.loc[books['original_title'].isnull(), 'title'].values

def mergeCol(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])
books['Features']=books.apply(mergeCol,axis='columns')

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
#     ngram_range=(1,1), 
    tokenizer = lambda i: i.split(' '),
#     analyzer = 'word' 
    )

matrixFeature = model.fit_transform(books['Features'])
features = model.get_feature_names()
totalFeatures=len(features)
print(totalFeatures)

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
score=cosine_similarity(matrixFeature)
# print(score)

# Find the index value of ratedbook (ranked 1-5) 
ratedbook1=books[books['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
ratedbook2=books[books['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
ratedbook3=books[books['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
ratedbook4=books[books['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
ratedbook=[ratedbook1,ratedbook2,ratedbook3,ratedbook4]
# print(ratedbook)

# Score list based on rank 1 to 4
listScore1=list(enumerate(score[ratedbook1]))
listScore2=list(enumerate(score[ratedbook2]))
listScore3=list(enumerate(score[ratedbook3]))
listScore4=list(enumerate(score[ratedbook4]))

listScore = []
for i in listScore1:
    listScore.append((i[0],0.25*(listScore1[i[0]][1]+listScore2[i[0]][1]+listScore3[i[0]][1]+listScore4[i[0]][1])))

sortListScore=sorted(
    listScore,
    key=lambda j:j[1],
    reverse=True)
# print(sortListScore[:5][0])

# Recomendation 
similarBooks=[]
for i in sortListScore:
    if i[1]>0:
        similarBooks.append(i)


for i in range(0,5):
    if similarBooks[i][0] not in ratedbook:
        print('-',books['original_title'].iloc[similarBooks[i][0]])
    else:
        i+=5
        print('-',books['original_title'].iloc[similarBooks[i][0]])
    