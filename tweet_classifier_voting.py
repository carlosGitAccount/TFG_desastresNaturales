import csv ,sys,os,time,numpy as np,pandas as pd ,sklearn as sk ,xgboost as xg , nltk ,string , enchant,re
from typing import final
from sklearn import preprocessing
from sklearn import svm 
from sklearn.preprocessing import OrdinalEncoder , MinMaxScaler
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
def remove_non_ascii(s):
    return "".join(c for c in s if ord(c)<128)
def clean_text (tweet_text) :
    
    stop_words = set(stopwords.words('english'))
    stop_words.update({"\x89û" , "û\x9d", "\x89ÛÒ" , "\x89ÛÓ" , "\x89ÛÏ" , "\x89Ûªs", "\x89Û÷","\x89Û¢" , "\x89Û\x9d","i'll","i'd","he's","there's"
    ,"we're","that's","they're","i'm","u","can't",'cant','arent','dont','lets','youll','we ve','re','te','thats','you re','rea','they d','we re','werent','i d','Â‰'})
    lemmatizer = WordNetLemmatizer()
    symbols = string.punctuation 
    split_text = tweet_text.split()
    clean_text = []
    for word in split_text : 
        clean_text.append(''.join([char for char in word if char.lower() not in symbols]))
        
       
    sentence = [lemmatizer.lemmatize(word.lower()) for word in clean_text if word not in stop_words and word.lower() not in stop_words]
    revised_sentence = []
    for word in sentence : 
        
        
        
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
        revised_sentence.append(remove_non_ascii(word))
        
    
    return ' '.join(revised_sentence)
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
    tweet_length = [len(tweet[-2]) for tweet in tweets]
    tweet_dataframe['tweet_length'] = tweet_length[1:]
    return tweet_dataframe
def text_Words (total_text) :
    word_frequency = {}
    for text in total_text :
        for word in text.split() :
            word_frequency.update({word:word_frequency.get(word,0)+1})
    return word_frequency
def create_bigrams(tweet_text) :
   
    bigram_frequency = {}
    for tweet in tweet_text :
        
        fuse = lambda tuple :' '.join(tuple)
        bigrams = [fuse(tuple) for tuple in zip(*[tweet.split()[i:] for i in range(2)])]
        
        for bigram in bigrams :
            bigram_frequency.update({bigram : bigram_frequency.get(bigram,0)+1})
    return {bigram : bigram_frequency.get(bigram) for bigram in bigram_frequency.keys() if bigram_frequency.get(bigram)>1}
def create_trigrams(tweet_text) :
   
    trigram_frequency = {}
    for tweet in tweet_text :
        
        fuse = lambda tuple :' '.join(tuple)
        trigrams = [fuse(tuple) for tuple in zip(*[tweet.split()[i:] for i in range(3)])]
        
        for trigram in trigrams :
            trigram_frequency.update({trigram : trigram_frequency.get(trigram,0)+1})
    return {trigram : trigram_frequency.get(trigram) for trigram in trigram_frequency.keys() if trigram_frequency.get(trigram)>1}
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
#___________________________XgBoost_______________________________________________________    
def create_XgModel(dt,tweet_text) : 
    vectorizer = CountVectorizer()  
    Y = pd.DataFrame(dt.target)
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(tweet_text)  
    column_names = vectorizer.get_feature_names()
    X_dt = pd.DataFrame(transformed_X.toarray(),columns=column_names)   
    enc = OneHotEncoder();
    final_X = enc.fit_transform(X_dt)[:len(dt.target),:]
    print("training_keys",final_X.shape)
    
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
def predict_tweets_XGB (dt,xGB_model,tweet_text) :
    
    vectorizer = CountVectorizer()  
    Y = pd.DataFrame(dt.target)
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(tweet_text)  
    column_names = vectorizer.get_feature_names()
    X_dt = pd.DataFrame(transformed_X.toarray(),columns=column_names)   
    enc = OneHotEncoder();
    final_X = enc.fit_transform(X_dt)[len(dt.target):,:]
    result = xGB_model.predict(final_X)
    return result
#_________________________________SVM______________________________________
def create_SVM(dt,test_dt,tweet_text):  
    Y = dt.target
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(tweet_text) 
    training_X = pd.DataFrame(transformed_X.toarray(),columns=vectorizer.get_feature_names()) 
    enc = OrdinalEncoder();
    transformed_X =enc.fit_transform(training_X[:len(dt.target)])
    scaler = MinMaxScaler(feature_range=(-1,1));
    min_maxX = scaler.fit_transform(transformed_X)
    SVM_classifier = svm.SVC(kernel='linear',C=10^2,gamma=10^2,verbose=True,probability=True)
    SVM_classifier.fit(min_maxX,Y)
    return SVM_classifier
def predict_tweets_SVM(dt,test_dt,SVM,tweet_text):  
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(tweet_text) 
    training_X = pd.DataFrame(transformed_X.toarray(),columns=vectorizer.get_feature_names()) 
    enc = OrdinalEncoder();
    transformed_X =enc.fit_transform(training_X[len(dt.target):])  
    scaler = MinMaxScaler(feature_range=(-1,1));
    min_maxX = scaler.fit_transform(transformed_X)  
    Y_test = SVM.predict_proba(min_maxX)
    scores = []
    
    for Y in Y_test :
        if Y[0] > Y[1] :
            scores = scores + ['0']
        else :
            scores = scores + ['1']
    
    return scores
#______________________________________Multinomial_________________________
def create_Multinomial_dataset (tweets,alpha) :
    disasterTweet_text = [clean_text (tweet[-2]) for tweet in tweets[1:] if tweet[-1] == '1']
    non_disasterTweet_text = [clean_text (tweet[-2]) for tweet in tweets[1:] if tweet[-1] == '0']
    disaster_bigrams= create_bigrams(disasterTweet_text)
    disaster_trigrams = create_trigrams(disasterTweet_text)
    non_disaster_bigrams = create_bigrams(non_disasterTweet_text)
    non_disaster_trigrams = create_trigrams(non_disasterTweet_text)
    vectorizer = CountVectorizer(vocabulary=list(text_Words(disasterTweet_text).keys())+list(disaster_bigrams.keys())
    + list(disaster_trigrams.keys())
    )
    disasterTweet_keywords= vectorizer.fit_transform(disasterTweet_text)
    disasterTweet_dt = pd.DataFrame(disasterTweet_keywords.toarray(),columns=vectorizer.get_feature_names())
    
    vectorizer = CountVectorizer(vocabulary=list(non_disaster_bigrams.keys()) + list(text_Words(non_disasterTweet_text).keys())
    + list(non_disaster_trigrams.keys())
    ) 
    non_disasterTweet_keywords = vectorizer.fit_transform(non_disasterTweet_text)
    non_disasterTweet_dt = pd.DataFrame(non_disasterTweet_keywords.toarray(),columns=vectorizer.get_feature_names())
    word_frequencyDisasterTweets = {word :disasterTweet_dt[word].sum() for word in disasterTweet_dt.keys()}
    word_frequency_Non_DisasterTweets = {word :non_disasterTweet_dt[word].sum() for word in non_disasterTweet_dt.keys()}
    word_Count_DisasterTweets =0
    word_Count_Non_DisasterTweets=0
    for key in word_frequencyDisasterTweets.keys() :
            word_Count_DisasterTweets+= int(word_frequencyDisasterTweets.get(key))
    for key in word_frequency_Non_DisasterTweets.keys() :
            word_Count_Non_DisasterTweets+= int(word_frequency_Non_DisasterTweets.get(key))
    
    word_frequency = word_frequencyDisasterTweets| word_frequency_Non_DisasterTweets
    total_wordCount = len(word_frequency.keys())
    word_ProbabilityDisasterTweets = {word :(word_frequencyDisasterTweets.get(word) + alpha) /
    (word_Count_DisasterTweets +alpha * total_wordCount) for word in word_frequencyDisasterTweets.keys()
    }
    word_ProbabilityNonDisasterTweets = {word :(word_frequency_Non_DisasterTweets.get(word) + alpha) /
    (word_Count_Non_DisasterTweets +alpha * total_wordCount) for word in word_frequency_Non_DisasterTweets.keys()
    }
    return [word_ProbabilityDisasterTweets,word_ProbabilityNonDisasterTweets]
def predict_tweet_Multinomial (tweet,probabilities,alpha,probability_disasterTweet,probability_Non_disasterTweet) : 
  
    test_bigrams= create_bigrams([tweet[-1]])
    test_trigrams = create_trigrams([tweet[-1]])
    vectorizer = CountVectorizer(vocabulary= list(test_bigrams.keys()) + list(text_Words([clean_text (tweet[-1])]))
    + list(test_trigrams.keys())
    )
    tweet_text = [clean_text (tweet[-1])]
    words_probability = probabilities[0]| probabilities[1]
    total_wordCount = len(words_probability.keys())
    if len(tweet_text[0]) > 0 :
        vectorizer.fit_transform(tweet_text)
        tweet_keywords = vectorizer.get_feature_names()
       
        
    else :
        tweet_keywords= ['']
    for keyword in tweet_keywords :
        if keyword in probabilities[0].keys():
            
            probability_disasterTweet = probability_disasterTweet * probabilities[0].get(keyword)
        else:
            
            probability_disasterTweet =probability_disasterTweet * (alpha /(len(probabilities[0].keys())
            + alpha *total_wordCount)
            )
        if keyword in probabilities[1].keys():   
            
            probability_Non_disasterTweet = probability_Non_disasterTweet*probabilities[1].get(keyword)
        else :
            probability_Non_disasterTweet =probability_Non_disasterTweet * (alpha /(len(probabilities[1].keys())
            + alpha *total_wordCount)
            )
    if probability_disasterTweet > probability_Non_disasterTweet :
      
        return 1
    else :
        
        return 0
#__________________________________________________________________________
training_tweets =obtain_tweets("\\train.csv")
test_tweets = obtain_tweets("\\test.csv")
tweet_dt = create_dataset(training_tweets)
test_tweet_dt = create_dataset(test_tweets)
tweet_text = [clean_text(tweet[-2]) for tweet in training_tweets[1:]] +[clean_text(tweet[-1]) for tweet in test_tweets[1:]]
train_matrix = create_XgModel(create_dataset(training_tweets),tweet_text)
XGB_predicted_values = predict_tweets_XGB(create_dataset(training_tweets),train_matrix,tweet_text)
cleaned_Entrytext = []
useless_words = uselessWords(tweet_text)
for i in range(len(tweet_text)) :
    text = tweet_text[i]
    for word in tweet_text[i].split() :
        if useless_words.get(word) < 25 :
           text = text.replace(word,'')
    cleaned_Entrytext.append(text)
tweet_SVM = create_SVM(tweet_dt,test_tweet_dt,cleaned_Entrytext)
SVM_predicted_values = predict_tweets_SVM(tweet_dt,test_tweet_dt,tweet_SVM,cleaned_Entrytext)
probability_disasterTweet = len([tweet in training_tweets for tweet in training_tweets[1:] if tweet[-1]== '1'])/len([tweet in training_tweets for tweet in training_tweets[1:]])
probability_Non_disasterTweet = len([tweet in training_tweets for tweet in training_tweets[1:] if tweet[-1]== '0'])/len([tweet in training_tweets for tweet in training_tweets[1:]])
multinomial_probabilities = create_Multinomial_dataset(training_tweets,1.5)
Multinomial_predicted_Values = [predict_tweet_Multinomial(tweet,multinomial_probabilities,1.5,probability_disasterTweet,probability_Non_disasterTweet) for tweet in test_tweets[1:]]
final_result = []

for i in range(len(SVM_predicted_values)) :
    if int(SVM_predicted_values[i]) + int(XGB_predicted_values[i]) + int(Multinomial_predicted_Values[i]) >= 2 :
        final_result.append('1')
    else :
        final_result.append('0')



'''
for i in range(len(SVM_predicted_values)) :
    print(SVM_predicted_values[i],XGB_predicted_values[i],Multinomial_predicted_Values[i])
    print(SVM_predicted_values[i][1],XGB_predicted_values[i][1],Multinomial_predicted_Values[i][1])
    disaster_probability = SVM_predicted_values[i][1]+XGB_predicted_values[i][1]+ Multinomial_predicted_Values[i][1]
    non_disaster_probability = SVM_predicted_values[i][0]+XGB_predicted_values[i][0]+ Multinomial_predicted_Values[i][0]
    
    if disaster_probability > non_disaster_probability :
        final_result.append('1')
    else :
        final_result.append('0')
'''
write_results(test_tweets,final_result)
