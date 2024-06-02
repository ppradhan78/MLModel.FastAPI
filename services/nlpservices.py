from nltk.tokenize import sent_tokenize, word_tokenize 
from fastapi import File, UploadFile

class NLPServices():

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
    