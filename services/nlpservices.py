from nltk.tokenize import sent_tokenize, word_tokenize 
from fastapi import File, UploadFile
import nltk
from nltk.corpus import stopwords
from heapq import nlargest
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
# import spacy
# import en_core_web_sm
import string
import re 
import numpy as np 
import heapq 
from nltk import ngrams
from fastapi import FastAPI, Response
import matplotlib.pyplot as plt
import io
from wordcloud import WordCloud
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

class NLPServices():
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('wordnet')
        nltk.download('words')
        # nltk.download('all')

    def Tokenization(self,text):
        words=word_tokenize(text)
        return {"tokenize":words,"StatusCode":201}
    
    def upload(self,file):
        try:
            contents = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(contents)
            
            return {"message":f"file '{file.filename}' saved at '{file_location}'","StatusCode":201,"content":contents}
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

    def TokenizationFile(self,file):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            words=word_tokenize(content)
            return {"tokenize":words,"StatusCode":201}
        except Exception:
            return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
        finally:
            file.file.close()

    def TokenizationStopWordFile(self,file):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            content=content.translate(str.maketrans('', '', string.punctuation))
            words=word_tokenize(content)
            # Get the English stop words
            stop_words = set(stopwords.words('english'))
            filtered_Stop_words = [word for word in words if word.lower() not in stop_words]
            porter_stemmer = PorterStemmer()
            stemmed_words = [porter_stemmer.stem(word) for word in filtered_Stop_words]
            wn.ensure_loaded()
            lemmatizer = WordNetLemmatizer()
            LemmatizeWord=[]
            for word in words:
                item=lemmatizer.lemmatize(word)
                LemmatizeWord.insert(1,item)        

            return { "tokenize":words,"filtered_Stop_words":filtered_Stop_words,"stemmed_words":stemmed_words,"LemmatizeWord":LemmatizeWord,"StatusCode":201}
        except Exception as error:
            return {"Exception": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        finally:
            file.file.close()


    def Summarization(self,file):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            nltk.download("stopwords")
            stop_words = stopwords.words('english')

            tokens = word_tokenize(content)
            punctuations = punctuation + '\n'
            word_frequencies = {}

            for word in tokens:
                if word.lower() not in stop_words:
                    if word.lower() not in punctuations:
                        if word not in word_frequencies.keys():
                            word_frequencies[word] = 1
                        else:
                            word_frequencies[word] += 1

            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                 word_frequencies[word] = word_frequencies[word]/max_frequency

            sent_token = sent_tokenize(content)
            sentence_scores = {}
            for sent in sent_token:
                    sentence = sent.split(" ")
                    for word in sentence:        
                        if word.lower() in word_frequencies.keys():
                                if sent not in sentence_scores.keys():
                                            sentence_scores[sent] = word_frequencies[word.lower()]
                                else:
                                    sentence_scores[sent] += word_frequencies[word.lower()]


            select_length = int(len(sent_token)*0.3)
            summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
            final_summary = [word for word in summary]
            summary = ' '.join(final_summary)
            return {"Summary":summary,"SummaryLength":len(summary),"ContentLength": len(content),"StatusCode":201}
        except Exception:
            return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
        finally:
            file.file.close()        

    def GetNamedEntityRecognition (self,file):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            NamedEntity=[]
            for sent in nltk.sent_tokenize(content):
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                    if hasattr(chunk, 'label'):
                        item=chunk.label(), ' '.join(c[0] for c in chunk)
                        NamedEntity.insert(0,item)
            return {"NamedEntity":NamedEntity,"StatusCode":201}
        except Exception as error:
            # return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
            return {"message": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        finally:
            file.file.close()        

    def GetPOS(self,file):
        try:
            stop_words = set(stopwords.words('english'))
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")

            content=content.translate(str.maketrans('', '', string.punctuation))

            tokenized = sent_tokenize(content)
            PartOfSpeach=[]
            for i in tokenized:
                # Word tokenizers is used to find the words and punctuation in a string
                wordsList = nltk.word_tokenize(i)
                # removing stop words from wordList
                wordsList = [w for w in wordsList if not w in stop_words]
                tagged = nltk.pos_tag(wordsList)
            PartOfSpeach.insert(1,tagged)
            return {"PartOfSpeach":PartOfSpeach,"StatusCode":201}
        except Exception as error:
            # return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
            return {"message": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        finally:
            file.file.close()        

    def GetBoW(self,file):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            text = f""" {content} """
            dataset = nltk.sent_tokenize(text)
            for i in range(len(dataset)):
                dataset[i] = dataset[i].lower()
                dataset[i] = re.sub(r'\W', ' ', dataset[i])
                dataset[i] = re.sub(r'\s+', ' ', dataset[i])     
    
            word2count = {} 
            for data in dataset:
                words = nltk.word_tokenize(data) 
                for word in words: 
                    if word not in word2count.keys():
                        word2count[word] = 1
                    else: 
                        word2count[word] += 1
            
            freq_words = heapq.nlargest(100, word2count, key=word2count.get)
            BoW = [] 
            for data in dataset: 
                vector = [] 
                for word in freq_words: 
                    if word in nltk.word_tokenize(data):
                        vector.append(1) 
                    else:
                        vector.append(0) 
                BoW.append(vector) 
            
            # BoW = np.asarray(BoW)
            return {"BoW":BoW, "WordCount":word2count,"freq_words":freq_words,"StatusCode":201}  
        except Exception as error:
            return {"Exception": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        finally:
            file.file.close()    

    def GetNGram(self,file,ngramsNumber):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            content=content.translate(str.maketrans('', '', string.punctuation))
            # words=word_tokenize(content)
            # Get the English stop words
            # stop_words = set(stopwords.words('english'))
            # filtered_Stop_words = [word for word in words if word.lower() not in stop_words]
            unigrams = ngrams(content.split(), ngramsNumber)
            gram=[]
            for grams in unigrams:
                gram.insert(0,grams)
            return { "ngrams":gram,"StatusCode":201}
        except Exception as error:
            return {"Exception": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        finally:
            file.file.close()

    def GetPhraseMatcher(self,file,Phras):
        try:
            content = file.file.read()
            file_location = f"files/{file.filename}"
            with open(file_location, 'wb') as f:
                f.write(content)
            
            content=content.decode("utf-8")
            content=content.translate(str.maketrans('', '', string.punctuation))
            phrase = re.compile (Phras)
            phrase = re.findall(phrase, content)
             
            return { "phrases":phrase,"StatusCode":201}
        except Exception as error:
            return {"Exception": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        finally:
            file.file.close()       
    
    def get_wordcloud(self,text):
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        # Save the word cloud to a BytesIO object
        buf = io.BytesIO()
        wordcloud.to_image().save(buf, format='PNG')
        buf.seek(0)
        return buf.read() 
    def create_upload_file_plot(self,file):
        try:
            file_location = f"files/{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())

            # Read the CSV file
            df = pd.read_csv(file_location)  
            # Generate a plot (example: plot the first two columns)
            plt.figure()
            df.plot(x=df.columns[0], y=df.columns[1], kind='line')
            plot_location = "files/plot.png"
            plt.savefig(plot_location)
            plt.close()
            return plot_location
        except Exception as error: 
            return {"Exception": str(error) +"@"+ type(error).__name__,"StatusCode":500}      