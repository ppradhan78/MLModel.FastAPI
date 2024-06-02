from fastapi import FastAPI
from services.nlpservices import NLPServices
from fastapi import File, UploadFile


app=FastAPI()
nlpServices=NLPServices();


@app.get("/HealthChecK",tags=["FASTAPI health check APIs"])
async  def HealthChecK():
    return  {"payload":"FASTAPI health check APIs.","StatusCode:":200 }


@app.get("/Tokenization",tags=["FASTAPI with Tokenization Text"])
def Tokenization(text):
    return  nlpServices.Tokenization(text)

@app.post("/TokenizationFile",tags=["FASTAPI for Tokenization Text File"])
def TokenizationFile(file: UploadFile):
    if file.content_type=='text/plain':
        return  nlpServices.TokenizationFile(file)
    else:
        return {"message": "Invalid file type. Please upload txt file","StatusCode":400}

@app.post("/upload",tags=["FASTAPI upload a Text file"])
def upload(file: UploadFile):
    if file.content_type=='text/plain':
        return  nlpServices.upload(file)
    else:
        return {"message": "Invalid file type. Please upload txt file","StatusCode":400}