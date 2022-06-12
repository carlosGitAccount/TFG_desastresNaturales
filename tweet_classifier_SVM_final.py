import csv ,sys,os,time,numpy as np,pandas as pd ,sklearn as sk ,matplotlib.pyplot as plt , nltk ,string,enchant,re
from graphviz.dot import Graph
from sklearn import svm 
from sklearn.preprocessing import OrdinalEncoder , MinMaxScaler
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
def create_dataset(tweets) :
    headings = [word for word in tweets[0]]
    tweet_dataframe = pd.DataFrame(tweets[1:],columns=headings)
    tweet_dataframe.keyword.fillna('0')
    tweet_length = [len(tweet[-2]) for tweet in tweets]
    tweet_dataframe['tweet_length'] = tweet_length[1:]
    return tweet_dataframe
def clean_text (tweet_text) :
    english_dictionary = enchant.Dict("en") 
    lemmatizer = WordNetLemmatizer()
    clean_text = [word.lower() for word in tweet_text.split()]
    revised_sentence = []
    for word in clean_text  :
        
        if len(word) > 0 :
            if english_dictionary.check(word) :
               revised_sentence.append(word) 
            elif not english_dictionary.check(word) and len(english_dictionary.suggest(word)) > 0 :
                
                revised_sentence.append(english_dictionary.suggest(word)[0]) 
            else :                  
                revised_sentence.append(word)
    return ' '.join([lemmatizer.lemmatize(word) for word in revised_sentence]) 
def create_SVM(dt,test_dt):  
    Y = dt.target
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(cleaned_Entrytext) 
    training_X = pd.DataFrame(transformed_X.toarray(),columns=vectorizer.get_feature_names()) 
    scaler = MinMaxScaler(feature_range=(-1,1));
    min_maxX = scaler.fit_transform(training_X[:len(dt.target)])
    SVM_classifier = svm.SVC(kernel='rbf',C=1)
    SVM_classifier.fit(min_maxX,Y)

    return SVM_classifier
    
def predict_tweets(dt,test_dt,SVM):  
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(cleaned_Entrytext) 
    training_X = pd.DataFrame(transformed_X.toarray(),columns=vectorizer.get_feature_names()) 

    scaler = MinMaxScaler(feature_range=(-1,1));
    min_maxX = scaler.fit_transform(training_X[len(dt.target):])  
    Y_test = SVM.predict(min_maxX)
    return Y_test
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
training_tweets=obtain_tweets("\\train.csv")
test_tweets = obtain_tweets("\\test.csv")
tweet_dt = create_dataset(training_tweets)
test_tweet_dt = create_dataset(test_tweets)
cleaned_training_text = [clean_text(text) for text in tweet_dt.text ]
cleaned_Entrytext= cleaned_training_text  + [clean_text(text) for text in test_tweet_dt.text]

tweet_SVM = create_SVM(tweet_dt,test_tweet_dt)
write_results(test_tweets,list(predict_tweets(tweet_dt,test_tweet_dt,tweet_SVM)))
print("--- %s seconds ---" % (time.time() - start_time))
