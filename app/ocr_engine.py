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

    # 2. Run OCR with optimized parameters for Korean text
    # Try multiple approaches to get best result
    resultOCR = ""
    
    # First attempt: with paragraph grouping for better context
    try:
        resultOCR = "".join(reader.readtext(
            img, 
            detail=0,
            paragraph=True,  # Group text for better context understanding
            width_ths=0.7,   # Standard threshold
            height_ths=0.7,  # Standard threshold
            slope_ths=0.3,   # Allow more slope variation for Korean characters
            ycenter_ths=0.5, # More flexible vertical centering
        ))
    except Exception as e:
        pass
    
    # If first attempt got no result, try with paragraph=False and lower thresholds
    if not resultOCR or len(resultOCR.strip()) < 10:
        try:
            resultOCR = "".join(reader.readtext(
                img, 
                detail=0,
                paragraph=False,
                width_ths=0.3,   # Very low threshold to catch everything
                height_ths=0.3,  # Very low threshold
                slope_ths=0.1,  # Very flexible slope
                ycenter_ths=0.3, # Very flexible centering
            ))
        except Exception as e:
            pass
    
    # If still no result, try with detail=1 to get more information
    if not resultOCR or len(resultOCR.strip()) < 10:
        try:
            detailed_results = reader.readtext(img, detail=1)
            # Extract text from detailed results
            resultOCR = " ".join([item[1] for item in detailed_results if len(item) > 1])
        except Exception as e:
            pass
    
    # Check if OCR returned any text
    if not resultOCR or len(resultOCR.strip()) == 0:
        return {
            "data": {
                "error": "OCR returned empty text. Image may be too blurry or contain no readable text."
            }
        }
    
    # 3. Normalize and format result using Gemini
    resultAIFormat = normalization_data_certificate(resultOCR)

    return {
        "data": resultAIFormat,
    }