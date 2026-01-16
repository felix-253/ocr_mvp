from app.gemini_service import normalization_data_certificate
import cv2
import easyocr
import numpy as np


reader = easyocr.Reader(['ko', 'en'], gpu=False)




def run_ocr_pipeline(image_path):
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "cannot read this image"}

    resultOCR = "".join(reader.readtext(img, detail=0))
    resultAIFormat = normalization_data_certificate(resultOCR)

    return {
        "data": resultAIFormat,
    }