
# coding: utf-8

# # Anime Recommendation System
# Jingwen Yan, Jingyi Sun, Yi Zhu

# In[9]:


import numpy as np
import pandas as pd
rating = pd.read_csv('rating.csv',na_values = ['?',""])
anime=pd.read_csv('anime.csv',na_values = ['?',""])


# In[10]:


rating.head()


# In[11]:


anime.head()


# In[12]:


n_users=rating['user_id'].nunique()


# In[13]:


n_animes=anime['anime_id'].nunique()


# In[14]:


rating['user_id'].nunique()


# In[15]:


anime['anime_id'].nunique()


# In[16]:


rating['anime_id'].nunique()


# In[17]:


rating['rating'].describe()


# In[18]:


print('Num of Users: '+ str(n_users))
print('Num of Movies: '+str(n_animes))


# In[19]:


df = pd.merge(rating,anime,on='anime_id')
data=df[df['rating_x']!=-1]


# In[20]:


data.head()


# In[21]:


data['anime_id'] = data['anime_id'].astype('category')
data['anime_id'] =data['anime_id'].cat.codes

data['genre'] = data['genre'].astype('category')
data['genre'] = data['genre'].cat.codes

data['type'] = data['type'].astype('category')
data['type'] = data['type'].cat.codes

data['name'] = data['name'].astype('category')
data['name'] = data['name'].cat.codes

data['episodes'] = data['episodes'].astype('category')
data['episodes'] = data['episodes'].cat.codes


# In[22]:


data=data[data['user_id']<=10000]


# In[23]:


data['anime_id'].nunique


# In[24]:


user_anime = pd.crosstab(data['user_id'], data['name'])
user_anime.head(10)


# # Plot

# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.boxplot(x='type',y='rating',data=anime)
plt.title('anime-type VS rating')
plt.show()


# In[73]:


TV_anime=anime[anime['type']=='TV']
TV_anime['genre'].value_counts().sort_values(ascending=True).tail(20).plot.barh(figsize=(8,8))
plt.title('genres of TV-Animes')
plt.xlabel('frequency')
plt.ylabel('genres')
plt.show()


# # Word Cloud

# In[76]:


from wordcloud import WordCloud


# In[78]:


import pandas as pd
import os

data1 = pd.read_csv('type.csv', encoding='utf-8')
with open('type.txt','a+', encoding='utf-8') as f:
    for line in anime.values:
        f.write((str(line[0])+'\n'))


# In[69]:


filename = "type.txt"
with open(filename) as f:
 mytext = f.read()


# In[71]:


wordcloud = WordCloud(background_color="white",width=2000, height=1600, margin=2).generate(mytext)
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # PCA

# In[63]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(user_anime)
pca_samples = pca.transform(user_anime)


# In[64]:


ps = pd.DataFrame(pca_samples)
ps.head()


# In[65]:


tocluster = pd.DataFrame(ps[[0,1,2]])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (6, 4)
plt.style.use('ggplot')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")
plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tocluster[0], tocluster[2], tocluster[1])

plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()


# # K_means

# In[23]:


from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
    


# # Memory-based CF

# In[22]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.25)

n_users_train=train_data['user_id'].max()
n_animes_train=train_data['anime_id'].max()

n_users_test=test_data['user_id'].max()
n_animes_test=test_data['anime_id'].max()

train_data_matrix = np.zeros((n_users_train, n_animes_train))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    
test_data_matrix = np.zeros((n_users_test, n_animes_test))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    
from sklearn.metrics.pairwise import pairwise_distances
user_similarity_train = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity_train = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity_test = pairwise_distances(test_data_matrix, metric='cosine')
item_similarity_test = pairwise_distances(test_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

item_prediction_train= predict(train_data_matrix, item_similarity_train, type='item')
user_prediction_train = predict(train_data_matrix, user_similarity_train, type='user')

item_prediction_test= predict(test_data_matrix, item_similarity_test, type='item')
user_prediction_test = predict(test_data_matrix, user_similarity_test, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print('Train User-based CF RMSE: ' + str(rmse(user_prediction_train, train_data_matrix)))
print('Train Item-based CF RMSE: ' + str(rmse(item_prediction_train, train_data_matrix)))

print('Test User-based CF RMSE: ' + str(rmse(user_prediction_test, test_data_matrix)))
print('Test Item-based CF RMSE: ' + str(rmse(item_prediction_test, test_data_matrix)))


# # SVD

# In[25]:


import surprise
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate


# In[26]:


reader = surprise.Reader(rating_scale = (1, 10))
anime_data = surprise.Dataset.load_from_df(data.drop(columns = ['name',
                                                                'genre',
                                                                'type',
                                                                'episodes',
                                                                'rating_y',
                                                                'members']), reader)


# In[27]:


svd_algo = surprise.SVD(biased = False)
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(anime_data, test_size = 0.2)


# In[28]:


svd_algo_cv = cross_validate(svd_algo, 
                             anime_data, 
                             measures = ['RMSE', 'MAE'], 
                             cv = 5, 
                             verbose = True)


# In[29]:


svd_algo.fit(trainset)


# In[31]:


train_predictions = svd_algo.test(trainset.build_testset())


# In[32]:


from surprise import accuracy
accuracy.rmse(train_predictions)


# In[33]:


predictions = svd_algo.test(testset)


# In[34]:


accuracy.rmse(predictions)


# # Hypertunning (SVD)

# In[35]:


from surprise.model_selection import GridSearchCV
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)

gs.fit(anime_data)

print(gs.best_params['rmse'])


# In[36]:


svd_algo = gs.best_estimator['rmse']
svd_algo.fit(trainset)


# In[72]:


train_best = svd_algo.test(trainset.build_testset())


# In[38]:


accuracy.rmse(train_best)


# In[73]:


test_best = svd_algo.test(testset)


# In[40]:


accuracy.rmse(test_best)


# Predicted score a user would give to a specific movie:

# In[42]:


anime_pred_svd = svd_algo.predict(uid = 325, iid = 2398)
anime_pred_svd_score = anime_pred_svd.est
print(anime_pred_svd_score)


# Continue to focus on user 1 and find one item to recommend to them. In order to do this, we have to find the movies that the user didn't rate since we shouldn't recommend movies that they have already seen.

# In[43]:


import numpy as np
user_id_list = data['user_id'].unique()
user_id_l_rated = data.loc[data['user_id'] == 1, 'anime_id']

user_id_1_unrated = np.setdiff1d(user_id_list, user_id_l_rated)


# We want to predict the score of each of the animes that user 50 did not rate and find the best one.

# In[44]:


user_id_testset = [[1, anime_id, 4.] for anime_id in user_id_1_unrated]
user_id_1_predictions = svd_algo.test(user_id_testset)

### We can now see all the predictions
predictions_df = pd.DataFrame(user_id_1_predictions)
predictions_df


# # KNN

# In[64]:


from surprise import KNNBasic


# In[66]:


knn_b_algo = KNNBasic()


# In[48]:


knnb_algo_cv = cross_validate(knn_b_algo, 
                              anime_data, 
                              measures = ['RMSE', 'MAE'], 
                              cv = 5, 
                              verbose = True)


# In[67]:


knn_b_algo.fit(trainset)


# In[68]:


knn_a_predictions = knn_b_algo.test(trainset.build_testset())


# In[69]:


accuracy.rmse(knn_a_predictions)


# In[70]:


knn_b_predictions = knn_b_algo.test(testset)


# In[71]:


accuracy.rmse(knn_b_predictions)


# # Hypertunning (KNN)

# In[55]:


k_param_grid = {'k':[20, 30, 40, 50, 60],
                'min_k':[1, 2, 3],
                'sim_options':{'name':['msd','cosine'],
                               'user_based':[False]}}

knnbasic_gs = GridSearchCV(KNNBasic, k_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)

knnbasic_gs.fit(anime_data)

print(knnbasic_gs.best_params['rmse'])


# In[58]:


knn_algo = knnbasic_gs.best_estimator['rmse']
knn_algo.fit(trainset)


# In[59]:


knn_train_best = knn_algo.test(trainset.build_testset())


# In[60]:


accuracy.rmse(knn_train_best)


# In[61]:


knn_test_best = knn_algo.test(testset)


# In[62]:


accuracy.rmse(knn_test_best)


# In[63]:


anime_pred_knn = knn_algo.predict(uid = 68, iid = 5114)
anime_pred_knn_score = anime_pred_knn.est
print(anime_pred_knn_score)


# # LightGBM

# In[24]:


data.head()


# In[25]:


data['rating_x'].unique()


# In[26]:


rating = data['rating_x']
bins = [0,3,7,10] 
bin_names = ['Low', 'Medium', 'High']
data['rating_category']= pd.cut(rating,bins,labels=bin_names)


# In[27]:


data['rating_category'] = data['rating_category'].astype('category')
data['rating_category'] =data['rating_category'].cat.codes


# In[28]:


data.head()


# In[29]:


from sklearn.preprocessing import StandardScaler
vars = data.drop(['user_id','rating_x','name','rating_category'], axis=1)
vars = pd.DataFrame(vars)
target = data['rating_category']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)

x_train_1, x_val, y_train_1, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)


# In[30]:


import lightgbm as lgb
lgb_bc={
    'boosting_type': 'gbdt',
    'objective':'multiclass',
    'metric':'multi_logloss',
    'num_class' : 3, 
    'max_depth':200,
    'num_leaves': 200,
    'learning_rate': 0.03,
    'n_estimators': 200,
    'early_stopping_round':500
}


# In[31]:


lgb_los_train = lgb.Dataset(x_train_1, y_train_1)
lgb_val_train = lgb.Dataset(x_val, y_val)


# In[32]:


lgb_gbm = lgb.train(params = lgb_bc, 
                    train_set = lgb_los_train,
                    num_boost_round = 100, 
                    valid_sets = [lgb_val_train, lgb_los_train],
                    valid_names = ['Evaluation', 'Train'])


# In[33]:


lgb_mn_preds_train = lgb_gbm.predict(x_train_1)
lgb_mn_preds_test = lgb_gbm.predict(x_test)


# In[34]:


best_lgb_preds_train = np.asarray([np.argmax(line) for line in lgb_mn_preds_train])
best_lgb_preds_test = np.asarray([np.argmax(line) for line in lgb_mn_preds_test])


# In[35]:


best_lgb_preds_train = pd.DataFrame(best_lgb_preds_train).add_prefix('PRED_QUAL')
best_lgb_preds_test = pd.DataFrame(best_lgb_preds_test).add_prefix('PRED_QUAL')


# In[36]:


y_train_df = pd.DataFrame(y_train_1)
y_test_df = pd.DataFrame(y_test)


# In[37]:


y_train_df['PRED_RATING'] = best_lgb_preds_train['PRED_QUAL0'].values
y_train_df['CORRECT_PREDS'] = np.where(y_train_df['PRED_RATING'] == y_train_df['rating_category'], 1, 0)

y_test_df['PRED_RATING'] = best_lgb_preds_test['PRED_QUAL0'].values
y_test_df['CORRECT_PREDS'] = np.where(y_test_df['PRED_RATING'] == y_test_df['rating_category'], 1, 0)


# In[38]:


sum(y_train_df['CORRECT_PREDS'])/len(y_train_1)


# In[39]:


sum(y_test_df['CORRECT_PREDS'])/len(y_test)


# # Hyperband

# In[40]:


import hyperband
from hyperband import HyperbandSearchCV

#Hyperband scores models using scikit-learn:
import sklearn
sklearn.metrics.SCORERS.keys()


# In[41]:


hb_lgb_model = lgb.LGBMClassifier()

lgb_hb_param_dict = {'boosting_type' : ['gbdt'],
                    'num_leaves' : np.arange(2, 1000),
                    'max_depth' : np.arange(2, 1000),
                    'learning_rate' : [0.001, 0.005, 0.01, 0.02, 0.03],
                    'n_estimators' : [100, 500,1000, 2000, 3000],
                    #'early_stopping_round':[100, 500,1000, 2000, 3000],
                    #subsample_for_bin : [],
                    'objective' : ['multiclass'],
                    'num_class' : [3], 
                    #'class_weight' : [],
                    #'min_split_gain' : [],
                    #'min_child_weight' : [],
                    #'min_child_samples' : [],
                    #'subsample' : [],
                    #'subsample_freq' : [],
                    #'colsample_bytree' : [],
                    #'reg_alpha' : [],
                    #'reg_lambda' : [],
                    'n_jobs' : [-1]
                    }


# In[42]:


search = HyperbandSearchCV(hb_lgb_model, lgb_hb_param_dict, cv=3,
                           verbose = 1,
                           max_iter=200,min_iter=50,
                           scoring='neg_log_loss')


# In[43]:


search.fit(x_train,y_train)


# In[44]:


search.best_params_


# # Fit new

# In[56]:


from sklearn.preprocessing import StandardScaler
vars = data.drop(['user_id','rating_x','name','rating_category'], axis=1)
vars = pd.DataFrame(vars)
target = data['rating_category']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(vars, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=1)

x_train_1, x_val, y_train_1, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1)


# In[57]:


import lightgbm as lgb
lgb_bc={
    'boosting_type': 'gbdt',
    'objective':'multiclass',
    'metric':'multi_logloss',
    'num_class' : 3, 
    'max_depth':627,
    'num_leaves': 198,
    'learning_rate': 0.03,
    'n_estimators': 200,
}


# In[58]:


lgb_gbm = lgb.train(params = lgb_bc, 
                    train_set = lgb_los_train,
                    num_boost_round = 100, 
                    valid_sets = [lgb_val_train, lgb_los_train],
                    valid_names = ['Evaluation', 'Train'])


# In[59]:


lgb_mn_preds_train = lgb_gbm.predict(x_train_1)
lgb_mn_preds_test = lgb_gbm.predict(x_test)


# In[60]:


best_lgb_preds_train = np.asarray([np.argmax(line) for line in lgb_mn_preds_train])
best_lgb_preds_test = np.asarray([np.argmax(line) for line in lgb_mn_preds_test])


# In[61]:


best_lgb_preds_train = pd.DataFrame(best_lgb_preds_train).add_prefix('PRED_RATING')
best_lgb_preds_test = pd.DataFrame(best_lgb_preds_test).add_prefix('PRED_RATING')


# In[62]:


y_train_df = pd.DataFrame(y_train_1)
y_test_df = pd.DataFrame(y_test)


# In[63]:


y_train_df['PRED_RATING'] = best_lgb_preds_train['PRED_RATING0'].values
y_train_df['CORRECT_PREDS'] = np.where(y_train_df['PRED_RATING'] == y_train_df['rating_category'], 1, 0)

y_test_df['PRED_RATING'] = best_lgb_preds_test['PRED_RATING0'].values
y_test_df['CORRECT_PREDS'] = np.where(y_test_df['PRED_RATING'] == y_test_df['rating_category'], 1, 0)


# In[64]:


sum(y_train_df['CORRECT_PREDS'])/len(y_train_1)


# In[65]:


sum(y_test_df['CORRECT_PREDS'])/len(y_test)


# In[66]:


lgb_mn_preds_train = pd.DataFrame(lgb_mn_preds_train)
best_lgb_preds_test.head()

