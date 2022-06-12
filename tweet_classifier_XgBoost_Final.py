import csv ,sys,os,time,numpy as np,pandas as pd ,sklearn as sk ,xgboost as xg , nltk ,string , enchant,re
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
real_DataPath = os.path.realpath('Data')
real_GraphPath = os.path.realpath('generated_graphs')

def obtain_tweets(csvfile) :
    tweets = []
    with open(real_DataPath+csvfile,'r',encoding='utf-8') as testFile:
        rows=csv.reader(testFile,delimiter=',')
        for r in rows :
            
            tweets.append(r)
    return tweets
def create_vocabulary (tweets):
    bigrams = create_bigrams(tweets)
    vocabulary = list(bigrams.keys())
    for tweet in tweets :
        for word in tweet.split():
            if word not in vocabulary :
                vocabulary.append(word)
    return vocabulary
def remove_non_ascii(s):
    return "".join(c for c in s if ord(c)<128)
def create_bigrams(tweet_text) :
   
    bigram_frequency = {}
    for tweet in tweet_text :
        
        fuse = lambda tuple :' '.join(tuple)
        bigrams = [fuse(tuple) for tuple in zip(*[tweet.split()[i:] for i in range(2)])]
        
        for bigram in bigrams :
            bigram_frequency.update({bigram : bigram_frequency.get(bigram,0)+1})
    return {bigram : bigram_frequency.get(bigram) for bigram in bigram_frequency.keys() if bigram_frequency.get(bigram)>1}
def first_Text_cleaning (tweet_text) :
    stop_words = set(stopwords.words('english'))
    
    stop_words.update({"i'll","i'd","he's","there's"
    ,"we're","that's","they're","i'm","u","can't"})
    
    symbols = string.punctuation
    split_text = [word.lower() for word in tweet_text.split()if word not in stop_words and word.lower() not in stop_words]
    clean_text = []
    for word in split_text : 
        clean_text.append(''.join([char for char in word if char.lower() not in symbols and char.isascii()]))
    
    
    
    return ' '.join([word for word in clean_text if  not word.startswith('http') and  word not in stop_words ]) 
def uselessWords (total_text) :
    word_frequency = {}
    for text in total_text :
        for word in text.split() :
            word_frequency.update({word:word_frequency.get(word,0)+1})
    return word_frequency
    
def create_dataset(tweets) :
    headings = [word for word in tweets[0]]
    tweet_dataframe = pd.DataFrame(tweets[1:],columns=headings)
    tweet_dataframe.keyword.fillna('0')
    return tweet_dataframe
def create_XgMatrix(dt) : 

    

    
    tweet_number = len(list(dt.text))
    training_number = round(tweet_number*0.8)
    Y_value = pd.DataFrame(dt.target[:training_number])
    tweet_vocabulary = create_vocabulary(tweet_text)
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(tweet_text[:tweet_number])  
    column_names = vectorizer.get_feature_names()
    X_dt = pd.DataFrame(transformed_X.toarray(),columns=column_names)   
    enc = OneHotEncoder();
    final_X = enc.fit_transform(X_dt)[:training_number,:]
    X_validation = enc.fit_transform(X_dt)[training_number:,:]
    Y_validation =  pd.DataFrame(dt.target[training_number:])
    parameters = {  "booster":"gbtree", 
                    "eta": 0.4, 
                    "objective": "binary:logistic", 
                    "max_depth": 8,
                    "colsample_bytree" : 0.1,
                    "n_estimators" : 1000,
                    "verbosity" : 0}
    XGB_model = xg.XGBClassifier()
    XGB_model.set_params(**parameters)
    
    XGB_model.fit(final_X,Y_value.values.ravel(),verbose=True,eval_metric="logloss",
    eval_set=[(final_X,Y_value.values.ravel()),(X_validation,Y_validation.values.ravel())]
    ,early_stopping_rounds = 50
    ) 
    new_Y = pd.DataFrame(dt.target)
    transformed_X =vectorizer.fit_transform(tweet_text)  
    column_names = vectorizer.get_feature_names()
    new_X_dt = pd.DataFrame(transformed_X.toarray(),columns=column_names)   
    
    
    enc = OneHotEncoder();
    new_final_X = enc.fit_transform(new_X_dt)[:len(dt.target),:]
    print(new_final_X.shape)
    new_parameters = {  "booster":"gbtree", 
                    "eta": 0.4, 
                    "objective": "binary:logistic", 
                    "max_depth": 8,
                    "colsample_bytree" : 0.1,
                    "n_estimators" : len(XGB_model.get_booster().get_dump()),
                    "verbosity" : 0}
    new_XGB_model = xg.XGBClassifier()
    new_XGB_model.set_params(**new_parameters)
    
    new_XGB_model.fit(new_final_X,new_Y.values.ravel(),verbose=True,eval_set=[(new_final_X,new_Y.values.ravel())])
    
    return new_XGB_model
  
def predict_tweets (dt,xGB_model) :
    
    vectorizer = CountVectorizer()  
    Y = pd.DataFrame(dt.target)
    tweet_vocabulary = create_vocabulary(tweet_text)
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(tweet_text)  
    column_names = vectorizer.get_feature_names()
    X_dt = pd.DataFrame(transformed_X.toarray(),columns=column_names)   
    enc = OneHotEncoder();
    final_X = enc.fit_transform(X_dt)[len(dt.target):,:]
    result = xGB_model.predict(final_X)
    return result
def write_results (tweets,target) :
    if os.path.exists(real_DataPath+"\\submission.csv"):
        os.remove(real_DataPath+"\\submission.csv")
    with open(real_DataPath+"\\submission.csv",'w',encoding='utf-8') as submissionFile:
        writer = csv.writer(submissionFile)
        for tweet in tweets :
            if tweets.index(tweet) == 0:
                tweet_stats = (str(list(tweet)[0]),"target") 
                writer.writerow(tweet_stats) 
            else:
                writer.writerow([tweet[0],target[tweets.index(tweet)-1]])
start_time = time.time()   
training_tweets =obtain_tweets("\\train.csv")
test_tweets = obtain_tweets("\\test.csv")

non_repeated_training_tweets = []
cleaned_training_text = []

for tweet in training_tweets : 
    if not first_Text_cleaning(tweet[-2]) in cleaned_training_text:
        cleaned_training_text.append(first_Text_cleaning(tweet[-2]))
        non_repeated_training_tweets.append(tweet)
tweet_text = cleaned_training_text +[first_Text_cleaning(tweet[-1]) for tweet in test_tweets[1:]]

train_matrix = create_XgMatrix(create_dataset(non_repeated_training_tweets))
predicted_values = predict_tweets(create_dataset(non_repeated_training_tweets),train_matrix)
write_results(test_tweets,predicted_values)
print("--- %s seconds ---" % (time.time() - start_time))
