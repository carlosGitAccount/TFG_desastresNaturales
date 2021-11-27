import csv,os,pandas as pd,nltk,string,re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
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

def create_dataset (tweets,alpha) :
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
def predict_tweet (tweet,probabilities,alpha,probability_disasterTweet,probability_Non_disasterTweet) : 
  
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

probabilities = create_dataset(training_tweets,1.75)
test_tweets = obtain_tweets("\\test.csv")
probability_disasterTweet = len([tweet in training_tweets for tweet in training_tweets[1:] if tweet[-1]== '1'])/len([tweet in training_tweets for tweet in training_tweets[1:]])
probability_Non_disasterTweet = len([tweet in training_tweets for tweet in training_tweets[1:] if tweet[-1]== '0'])/len([tweet in training_tweets for tweet in training_tweets[1:]])
results = [predict_tweet(tweet,probabilities,1.75,probability_disasterTweet,probability_Non_disasterTweet) for tweet in test_tweets[1:]]

write_results(test_tweets,results)