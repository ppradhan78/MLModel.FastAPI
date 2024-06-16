from fastapi import FastAPI
from services.nlpservices import NLPServices
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
        return  nlpServices.TokenizationStopWordFile(file)
    else:
        return {"message": "Invalid file type. Please upload txt file","StatusCode":400}

@app.post("/upload",tags=["FASTAPI upload a Text file"])
def upload(file: UploadFile):
    if file.content_type=='text/plain':
        return  nlpServices.upload(file)
    else:
        return {"message": "Invalid file type. Please upload txt file","StatusCode":400}
    

@app.post("/Summarization",tags=["FASTAPI for Summarization Text File"])
def Summarization(file: UploadFile):
    if file.content_type=='text/plain':
        return  nlpServices.Summarization(file)
    else:
        return {"message": "Invalid file type. Please upload txt file","StatusCode":400}    
    
# @app.post("/GetNamedEntityRecognition",tags=["Get NamedEntity"])
# def GetNamedEntityRecognition(file: UploadFile):
#     if file.content_type=='text/plain':
#         return  nlpServices.GetNamedEntityRecognition(file)
#     else:
#         return {"message": "Invalid file type. Please upload txt file","StatusCode":400}    
    
@app.post("/GetPOS",tags=["Get POS"])
def GetPOS(file: UploadFile):
    try:
        if file.content_type=='text/plain':
            return  nlpServices.GetPOS(file)
        else:
            return {"message": "Invalid file type. Please upload txt file","StatusCode":400} 
    except Exception as error:
            return {"message": str(error) +"@"+ type(error).__name__,"StatusCode":500}     

@app.post("/GetBoW",tags=["Get BoW"])
def GetBoW(file: UploadFile):
        try:
             if file.content_type=='text/plain':
                  return  nlpServices.GetBoW(file)
             else:
                return {"message": "Invalid file type. Please upload txt file","StatusCode":400} 
        except Exception as error:
            return {"message": str(error) +"@"+ type(error).__name__,"StatusCode":500}   
        
@app.post("/GetNGram",tags=["Get BoW"])
def GetNGram(file: UploadFile,ngramsNumber:int):
        try:
             if file.content_type=='text/plain':
                  return  nlpServices.GetNGram(file,ngramsNumber)
             else:
                return {"message": "Invalid file type. Please upload txt file","StatusCode":400} 
        except Exception as error:
            return {"message": str(error) +"@"+ type(error).__name__,"StatusCode":500}  
        