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
def clean_text (tweet_text) :
    stop_words = set(stopwords.words('english'))
    
    stop_words.update({"\x89û" , "û\x9d", "\x89ÛÒ" , "\x89ÛÓ" , "\x89ÛÏ" , "\x89Ûªs", "\x89Û÷","\x89Û¢" , "\x89Û\x9d","i'll","i'd","he's","there's"
    ,"we're","that's","they're","i'm","u","can't"})
    lemmatizer = WordNetLemmatizer()
    english_dictionary = enchant.Dict("en") 
    symbols = string.punctuation
    split_text = [word for word in tweet_text.split() if word not in stop_words and word.lower() not in stop_words]
    clean_text = []
    for word in split_text : 
        clean_text.append(''.join([char for char in word if char.lower() not in symbols]))
    
       
    sentence = [lemmatizer.lemmatize(word.lower()) for word in clean_text if  not word.startswith('http')]
    revised_sentence = []
    
    for word in sentence  :
        
        if len(word) > 0 :
            if english_dictionary.check(word) :
               revised_sentence.append(word) 
            elif not english_dictionary.check(word) and len(english_dictionary.suggest(word)) > 0 :
                #sentence[sentence.index(word)] = english_dictionary.suggest(word)[0]
                revised_sentence.append(english_dictionary.suggest(word)[0]) 
            else :                  
                revised_sentence.append(word)
    
    final_sentence = []
    for word in revised_sentence : 
        if not  re.search("[0-9]+",word) and not  len(word) < 2 :
            final_sentence.append(word) 
    
    return ' '.join(final_sentence)
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
def create_SVM(dt,test_dt):  
    Y = dt.target
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(cleaned_Entrytext) 
    training_X = pd.DataFrame(transformed_X.toarray(),columns=vectorizer.get_feature_names()) 
   
    enc = OrdinalEncoder();
    transformed_X =enc.fit_transform(training_X[:len(dt.target)])
    
    scaler = MinMaxScaler(feature_range=(-1,1));
    min_maxX = scaler.fit_transform(transformed_X)
    
    SVM_classifier = svm.SVC(kernel='linear',C=10^2,gamma=10^2,verbose=True,probability=True)
    SVM_classifier.fit(min_maxX,Y)
    
    return SVM_classifier
    
def predict_tweets(dt,test_dt,SVM):  
    vectorizer = CountVectorizer()
    transformed_X =vectorizer.fit_transform(cleaned_Entrytext) 
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

        
training_tweets=obtain_tweets("\\train.csv")
test_tweets = obtain_tweets("\\test.csv")
tweet_dt = create_dataset(training_tweets)
test_tweet_dt = create_dataset(test_tweets)
cleaned_Entrytext= [clean_text(text) for text in tweet_dt.text ] + [clean_text(text) for text in test_tweet_dt.text]
useless_words = uselessWords(cleaned_Entrytext)
for i in range(len(cleaned_Entrytext)) :
    text = cleaned_Entrytext[i].split()
    for word in text :
        if useless_words.get(word) < 2 :
            cleaned_Entrytext[i] = cleaned_Entrytext[i].replace(word,'')
tweet_SVM = create_SVM(tweet_dt,test_tweet_dt)
write_results(test_tweets,list(predict_tweets(tweet_dt,test_tweet_dt,tweet_SVM)))