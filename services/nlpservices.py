from nltk.tokenize import sent_tokenize, word_tokenize 
from fastapi import File, UploadFile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
from heapq import nlargest
from string import punctuation
class NLPServices():
    def __init__(self):
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
            return {"summary":summary,"summary length":len(summary),"content length": len(content),"StatusCode":201}
        except Exception:
            return {"message": "There was an error uploading the file and word tokenize","StatusCode":500}
        finally:
            file.file.close()        
    