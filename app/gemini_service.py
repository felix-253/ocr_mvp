from google import genai
from google.genai import types
import json
import os
import re
from app.config import *

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))




def normalization_data_certificate(fullTextOcr):
    prompt = f"""
    You are an expert AI specialized in correcting OCR errors for Korean Vehicle Registration Certificates (자동차등록증).
    
    --- RAW OCR TEXT ---
    {fullTextOcr}
    --- END RAW TEXT ---

    ### YOUR MISSION:
    Extract structured data and apply STRICT correction rules to fix OCR mistakes.

    ### FIELD 1: "vehicle_number" (License Plate)
    * **Target Regex:** ^\\d{{2,3}}[가-힣]\\d{{4}}$ (Example: 123가4567 or 12무9999)
    * **Context Keywords:** Look near "자동차등록번호", "차량번호", "등록번호".
    * **CRITICAL CORRECTION RULES (Priority High):**
        1.  **Stroke Separation Fix:** OCR often misreads the Korean character '가' (ga) as '71', '11', '7l', or '1l'.
            -   *IF* you see a sequence like "134715164" (9 digits total), *THEN* convert the middle "71" to "가" -> "134가5164".
            -   *IF* you see "134115164", *THEN* convert "11" to "가".
        2.  **Digit/Letter Confusion:**
            -   Convert 'O', 'o', 'Q' to '0' if they appear in the numeric part.
            -   Convert 'I', 'l' to '1' if they appear in the numeric part.
        3.  **Spacing:** Remove all spaces (e.g., "123 가 4567" -> "123가4567").

    ### FIELD 2: "owner_name" (Owner Name)
    * **Context Keywords:** Look near "소유자", "성명", "명칭".
    * **Format:** Typically a Korean name (2-4 characters) or a Company name.
    * **Rule:** Extract only the name value, remove labels like "성명 :".

    ### FIELD 3: "confidence" (Scoring)
    * **1.0**: Perfect match with Target Regex found in raw text without changes.
    * **0.8 - 0.9**: Logic applied (e.g., you fixed "71" -> "가", or "O" -> "0") and the result now matches the Target Regex.
    * **< 0.5**: Result does not match Target Regex or data is missing.

    ### OUTPUT FORMAT:
    Return ONLY a valid JSON object. Do not use Markdown block.
    {{
        "vehicle_number": {{ "value": "string", "confidence": float }},
        "owner_name": {{ "value": "string", "confidence": float }}
    }}
    """

    try:
        res = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL_NAME"),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        raw_json=json.loads(res.text)
        
        # final_data = rule_validate(raw_json)
        return raw_json
    except Exception as e:
        print(f"Error:{e}")
        return {
            "error":str(e)
        }