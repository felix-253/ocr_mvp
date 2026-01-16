from fastapi import FastAPI, APIRouter, UploadFile,File, HTTPException
from app.ocr_engine import run_ocr_pipeline
import os 
import shutil
from app.config import *


app = FastAPI(title=os.getenv("APP_NAME","OCR"))

app_router = APIRouter(prefix='/api')


@app_router.get("/healthy")
async def healthy():
        return {'status': 'ok'}

@app_router.post('/test-ocr')
async def fileCertificate(file: UploadFile=File(...)):
        base_dir = f"./data/uploads"
        print("local",f".{os.getenv('UPLOAD_URI','/data/uploads')}")
        os.makedirs(base_dir,exist_ok=True)

        file_location = f"{base_dir}/{file.filename}"
        

        try:
                with open(file_location,"wb") as buffer:
                        shutil.copyfileobj(file.file,buffer)
                result=run_ocr_pipeline(file_location)
                return {
                        "success":True,
                        "result":result
                }
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        finally:
                if(os.path.exists(file_location)):
                        os.remove(file_location)
                        print("DELETE FILE",file_location)
                pass
       

app.include_router(app_router)