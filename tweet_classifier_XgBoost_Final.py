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
    
    stop_words.update({"\x89û" , "û\x9d", "\x89ÛÒ" , "\x89ÛÓ" , "\x89ÛÏ" , "\x89Ûªs", "\x89Û÷","\x89Û¢" , "\x89Û\x9d","i'll","i'd","he's","there's"
    ,"we're","that's","they're","i'm","u","can't",'cant','arent','dont','lets','youll','we ve','re','te','thats','you re','rea','they d','we re','werent','i d','Â‰'})

    symbols = string.punctuation    
    split_text = [word for word in tweet_text.split()]
    clean_text = []
    for word in split_text : 
        clean_text.append(''.join([char for char in word if char.lower() not in symbols]))
    revised_sentence = []
    for word in clean_text : 
        
        
        
        while  "û" in word:
               
            if  re.search("[.]+û+",word) :
            
                word=word.replace('û','') 
            elif re.search("û+[.]+",word) :
                word=word.replace('û','')
                
            elif re.search(".+û+.+",word) :
            
                word=word.replace('û',' ')  
            else :
                word=word.replace('û','')

        while  "÷" in word:
               
            if  re.search("[.]+÷+",word) :
            
                word=word.replace('÷','')
            elif re.search("÷+[.]+",word) :
                word=word.replace('÷','')
                
            elif re.search(".+÷+.+",word) :
            
                word=word.replace('÷',' ') 
            else :
                word=word.replace('÷','')

        while  "ª" in word:
               
            if  re.search("[.]+ª+",word) :
            
                word=word.replace('ª','')            
            elif re.search("ª+[.]+",word) :
                word=word.replace('ª','')
                
            elif re.search(".+ª+.+",word) :
            
                word=word.replace('ª',' ')             
            else :
                word=word.replace('ª','')
                
        while  "ï" in word:
               
            if  re.search("[.]+ï+",word) :
            
                word=word.replace('ï','')
               
                
            elif re.search("ï+[.]+",word) :
                word=word.replace('ï','')
                
            elif re.search(".+ï+.+",word) :
            
                word=word.replace('ï',' ') 
            else :
                word=word.replace('ï','')
        
        while  "ó" in word:
               
            if  re.search("[.]+ó+",word) :
            
                word=word.replace('ó','')
               
                
            elif re.search("ó+[.]+",word) :
                word=word.replace('ó','')
                
            elif re.search(".+ó+.+",word) :
            
                word=word.replace('ó',' ') 
            else :
                word=word.replace('ó','')
        while  "¢" in word:
               
            if  re.search("[.]+¢+",word) :
            
                word=word.replace('¢','')
               
                
            elif re.search("¢+[.]+",word) :
                word=word.replace('¢','')
                
            elif re.search(".+¢+.+",word) :
            
                word=word.replace('¢',' ') 
            else :
                word=word.replace('¢','')
        while  "å" in word:
               
            if  re.search("[.]+å+",word) :
            
                word=word.replace('å','')
               
                
            elif re.search("å+[.]+",word) :
                word=word.replace('å','')
                
            elif re.search(".+å+.+",word) :
            
                word=word.replace('å',' ') 
            else :
                word=word.replace('å','')
        while  "ê" in word:
               
            if  re.search("[.]+ê+",word) :
            
                word=word.replace('ê','')
               
                
            elif re.search("ê+[.]+",word) :
                word=word.replace('ê','')
                
            elif re.search(".+ê+.+",word) :
            
                word=word.replace('ê',' ') 
            else :
                word=word.replace('ê','')
        while  "ã" in word:
               
            if  re.search("[.]+ã+",word) :
            
                word=word.replace('ã','')
               
                
            elif re.search("ã+[.]+",word) :
                word=word.replace('ã','')
                
            elif re.search(".+ã+.+",word) :
            
                word=word.replace('ã',' ') 
            else :
                word=word.replace('ã','')
        while  "Â" in word:
               
            if  re.search("[.]+Â+",word) :
            
                word=word.replace('Â','')
               
                
            elif re.search("Â+[.]+",word) :
                word=word.replace('Â','')
                
            elif re.search(".+Â+.+",word) :
            
                word=word.replace('Â',' ') 
            else :
                word=word.replace('Â','')
        while  "‰" in word:
               
            if  re.search("[.]+‰+",word) :
            
                word=word.replace('‰','')
               
                
            elif re.search("‰+[.]+",word) :
                word=word.replace('‰','')
                
            elif re.search(".+‰+.+",word) :
            
                word=word.replace('‰',' ') 
            else :
                word=word.replace('‰','')   
        while  "Ã" in word:
               
            if  re.search("[.]+Ã+",word) :
            
                word=word.replace('Ã','')
               
                
            elif re.search("Ã+[.]+",word) :
                word=word.replace('Ã','')
                
            elif re.search(".+Ã+.+",word) :
            
                word=word.replace('Ã',' ') 
            else :
                word=word.replace('Ã','')  
        while  "©" in word:
               
            if  re.search("[.]+©+",word) :
            
                word=word.replace('©','')
               
                
            elif re.search("©+[.]+",word) :
                word=word.replace('©','')
                
            elif re.search(".+©+.+",word) :
            
                word=word.replace('©',' ') 
            else :
                word=word.replace('©','')   
        while  "¬" in word:
               
            if  re.search("[.]+¬+",word) :
            
                word=word.replace('¬','')
               
                
            elif re.search("¬+[.]+",word) :
                word=word.replace('¬','')
                
            elif re.search(".+¬+.+",word) :
            
                word=word.replace('¬',' ') 
            else :
                word=word.replace('¬','')  
        while  "²" in word:
               
            if  re.search("[.]+²+",word) :
            
                word=word.replace('²','')
               
                
            elif re.search("²+[.]+",word) :
                word=word.replace('²','')
                
            elif re.search(".+²+.+",word) :
            
                word=word.replace('²',' ') 
            else :
                word=word.replace('²','')           
        revised_sentence.append(word)
    return ' '.join([remove_non_ascii(word) for word in revised_sentence if len(word.replace(" ","")) > 1 and  word not in stop_words and word.replace(" ","") not in stop_words and word.lower() not in  stop_words and not word.startswith('http') ]) 
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

    

    vectorizer = CountVectorizer()  
    Y = pd.DataFrame(dt.target)
    tweet_vocabulary = create_vocabulary(tweet_text)
    vectorizer = CountVectorizer(vocabulary= tweet_vocabulary,ngram_range=(1,2))
    transformed_X =vectorizer.fit_transform(tweet_text)  
    column_names = vectorizer.get_feature_names()
    X_dt = pd.DataFrame(transformed_X.toarray(),columns=column_names)   
    enc = OneHotEncoder();
    final_X = enc.fit_transform(X_dt)[:len(dt.target),:]
   
    
    parameters = {  "booster":"gbtree", 
                    "max_depth": 10, "eta": 0.5, 
                    "objective": "binary:logistic", 
                    "nthread":2,
                    "n_estimators" : 200,
                    "verbosity" : 0}
    XGB_model = xg.XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6)
    XGB_model.set_params(**parameters)
    
    XGB_model.fit(final_X,Y.values.ravel(),verbose=True,eval_set=[(final_X,Y.values.ravel())]
    
    ) 
   
    
    return XGB_model
    
def remove_repeatedTweets (tweet_text):
    final_tweets = []
    for text in tweet_text :
        if not text in final_tweets :
            final_tweets.append(text)
    return final_tweets    
def predict_tweets (dt,xGB_model) :
    
    vectorizer = CountVectorizer()  
    Y = pd.DataFrame(dt.target)
    tweet_vocabulary = create_vocabulary(tweet_text)
    vectorizer = CountVectorizer(vocabulary= tweet_vocabulary,ngram_range=(1,2))
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
