from nltk.tokenize import sent_tokenize, word_tokenize 
from fastapi import File, UploadFile
import nltk
from nltk.corpus import stopwords
from heapq import nlargest
from string import punctuation
from nltk.stem import PorterStemmer
# import spacy
# import en_core_web_sm
import string

class NLPServices():
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')

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
            return { "tokenize":words,"filtered_Stop_words":filtered_Stop_words,"stemmed_words":stemmed_words,"StatusCode":201}
        except Exception:
            return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
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
    
    # def GetNamedEntityRecognition (self,file):
    #     try:
    #         # nlp = spacy.load("en_core_web_sm")
    #         content = file.file.read()
    #         file_location = f"files/{file.filename}"
    #         with open(file_location, 'wb') as f:
    #             f.write(content)
            
    #         content=content.decode("utf-8")
    #         nlp = en_core_web_sm.load()
    #         doc = nlp(content)
    #         NamedEntity=[]
    #         for entity in doc.ents:
    #             NamedEntity.insert(1, f"{entity.text}-({entity.label_})")
           
    #         return {"NamedEntity":NamedEntity,"StatusCode":201}
    #     except Exception:
    #         return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
    #     finally:
    #         file.file.close()

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
            POS=[]
            for i in tokenized:
                # Word tokenizers is used to find the words and punctuation in a string
                wordsList = nltk.word_tokenize(i)
                # removing stop words from wordList
                wordsList = [w for w in wordsList if not w in stop_words]
                tagged = nltk.pos_tag(wordsList) 
            POS.insert(1,tagged)
            return {"POS":POS,"StatusCode":201}
        except Exception:
            return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
        finally:
            file.file.close()        