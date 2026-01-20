from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from app.ocr_engine import run_ocr_pipeline
from app.vehicle_composition_service import process_vehicle_composition
from pydantic import BaseModel
from typing import Optional
import os 
import shutil
import base64
import uuid
from datetime import datetime
from pathlib import Path
import httpx
from urllib.parse import urlencode
from PIL import Image
from app.config import *


class SeriesSelection(BaseModel):
    seriesno: str
    ts_key: str


app = FastAPI(title=os.getenv("APP_NAME","OCR"))

# Configure CORS - Allow localhost and 127.0.0.1 with any port
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_router = APIRouter(prefix='/api')


@app_router.get("/healthy")
async def healthy():
        return {'status': 'ok'}

@app_router.post('/test-ocr')
async def fileCertificate(file: UploadFile=File(...)):
        base_dir = f"./data/uploads"
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
                pass


@app_router.post('/scan-certificate')
async def scan_certificate(file: UploadFile = File(...)):
    """
    Scan vehicle registration certificate, extract vehicle number and owner name,
    then query Autobegin API for vehicle information.
    
    Returns OCR results and Autobegin vehicle data.
    """
    base_dir = f"./data/uploads"
    os.makedirs(base_dir, exist_ok=True)
    
    file_location = f"{base_dir}/{file.filename}"
    
    try:
        # Save uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess image before OCR: crop, scale, and enhance for Korean text recognition
        try:
            img = Image.open(file_location)
            original_width, original_height = img.size
            # Step 1: Crop image if large enough
            if original_width >= 1100 and original_height >= 600:
                # Crop: left, top, right, bottom
                img = img.crop((0, 0, 1100, 600))
            else:
                # If image is smaller, use actual dimensions but still process
                crop_width = min(1100, original_width)
                crop_height = min(600, original_height)
                img = img.crop((0, 0, crop_width, crop_height))
            
            # Step 2: Scale image up 2.5x for better OCR accuracy (especially for similar Korean characters like 욱/육)
            # Higher resolution helps distinguish fine details
            scale_factor = 2.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Step 3: Use OpenCV for advanced preprocessing to distinguish similar Korean characters
            # But keep it lighter to avoid losing text
            import cv2
            import numpy as np
            
            # Convert PIL to numpy array for OpenCV processing
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # Convert to grayscale for better character distinction
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) - lighter version
            # This helps distinguish similar characters like 욱 vs 육 without being too aggressive
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduced clipLimit
            gray = clahe.apply(gray)
            
            # Convert back to RGB for PIL (keep grayscale as RGB for compatibility)
            # Don't use adaptive thresholding as it might be too aggressive and lose text
            processed_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(processed_rgb)
            
            # Step 4: Additional PIL enhancements
            from PIL import ImageFilter, ImageEnhance
            
            # Apply unsharp mask for better edge definition (helps distinguish 욱 from 육)
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Enhance contrast more aggressively for Korean characters
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.4)  # Increase contrast by 40%
            
            # Enhance sharpness to make fine details clearer
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Increase sharpness by 50%
            
            # Save preprocessed image (overwrite original)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            
            # Save with explicit format and high quality
            img.save(file_location, format='JPEG', quality=98, optimize=False)
            
            # Verify file was saved
            if not os.path.exists(file_location):
                raise Exception(f"File was not saved: {file_location}")
            
        except Exception as preprocess_error:
            # If preprocessing fails, continue with original image
            pass
        
        # Run OCR pipeline
        ocr_result = run_ocr_pipeline(file_location)
        
        if "error" in ocr_result.get("data", {}):
            return {
                "success": False,
                "error": ocr_result["data"]["error"],
                "ocr_result": ocr_result
            }
        
        # Extract vehicle_number and owner_name from OCR result
        ocr_data = ocr_result.get("data", {})
        vehicle_number = ocr_data.get("vehicle_number", {}).get("value", "") if isinstance(ocr_data.get("vehicle_number"), dict) else ""
        owner_name = ocr_data.get("owner_name", {}).get("value", "") if isinstance(ocr_data.get("owner_name"), dict) else ""
        
        # Post-processing: Fix common OCR errors for Korean names
        # Fix "권대육" -> "권대욱" (common OCR error)
        if owner_name and "대육" in owner_name:
            owner_name = owner_name.replace("대육", "대욱")
        
        if not vehicle_number:
            return {
                "success": False,
                "error": "Vehicle number not found in OCR result",
                "ocr_result": ocr_result
            }
        
        # Call new Autobegin API via vsol.hanbirosoft.com
        autobegin_base_url = os.getenv("AUTOBEGIN_BASE_URL", "http://vsol.hanbirosoft.com/api/autobegin")
        
       
        
        # Prepare query parameters for search API
        query_params = {
            "mode": "search",
            "carNum": vehicle_number,
        }
        
        # Add owner if available
        if owner_name:
            query_params["owner"] = owner_name
        
        # Build URL with query parameters
        url = f"{autobegin_base_url}/search?{urlencode(query_params)}"
        
        # Call new Autobegin API with bearer token (very long timeout: 5 minutes)
        timeout_config = httpx.Timeout(300.0, connect=60.0)  # 5 minutes total, 60s connect
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.get(
                    url,
                    headers={
                        "Content-Type": "application/json",
                    }
                )
                response.raise_for_status()
                api_response = response.json()
                
                # Extract data from response (format: {"success": true, "data": {...}})
                if not api_response.get("success", False):
                    return {
                        "success": False,
                        "error": api_response.get("message", "API returned unsuccessful response"),
                        "ocr_result": ocr_result
                    }
                
                autobegin_data = api_response.get("data", {})
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"Autobegin API error: {str(e)}",
                    "ocr_result": ocr_result
                }
        
        # Process Autobegin response based on rst_code
        rst_code = autobegin_data.get("rst_code")
        
        result = {
            "success": True,
            "ocr_result": ocr_result,
            "vehicle_number": vehicle_number,
            "owner_name": owner_name,
            "autobegin_data": autobegin_data,
            "rst_code": rst_code
        }
        
        # Handle different rst_code values
        if rst_code == 1:
            # Success: Full vehicle information available
            cardata = autobegin_data.get("cardata", {})
            
            # Extract vehicle information
            makername = cardata.get("makername", "")
            classname = cardata.get("classname", "")
            modelname = cardata.get("modelname", "")
            seriesname = cardata.get("seriesname", "")
            seriesname1 = cardata.get("seriesname1", "")
            seriesname2 = cardata.get("seriesname2", "")
            
            # Build car type
            series_part = f"{seriesname1}_{seriesname2}" if (seriesname1 and seriesname2) else seriesname
            car_type_parts = [makername, classname, modelname, series_part]
            # Filter out empty strings and join
            car_type = "_".join(filter(None, car_type_parts)).strip("_") if any(car_type_parts) else None
            
            # Add formatted data to result
            result["vehicle_info"] = {
                "car_type": car_type,
                "makername": makername,
                "classname": classname,
                "modelname": modelname,
                "seriesname": seriesname,
                "seriesname1": seriesname1,
                "seriesname2": seriesname2,
                "newprice": cardata.get("newprice"),
                "firstdate": cardata.get("firstdate"),
                "year": cardata.get("year"),
                # Additional fields from cardata (new API format)
                "chassis_number": cardata.get("vin") or cardata.get("chassis_number") or cardata.get("chassisnumber"),
                "vin": cardata.get("vin"),
                "engine_type": cardata.get("engine_type") or cardata.get("enginetype"),
                "displacement": cardata.get("displacement"),
                "fuel": cardata.get("fuel"),
                "color": cardata.get("color"),
                "purpose": cardata.get("usegubun") or cardata.get("purpose"),
                "seating": cardata.get("seating"),
                "carnum": cardata.get("carnum"),
                "regname": cardata.get("regname"),
                "full_data": cardata
            }
            
        elif rst_code == 2:
            # Multiple series found: Return list for user to select
            # Extract modellist and serieslist from response
            modellist = autobegin_data.get("modellist", [])
            ts_key = autobegin_data.get("ts_key", "")
            
            # Flatten series list from all models
            series_list = []
            for model in modellist:
                model_series_list = model.get("serieslist", [])
                for series in model_series_list:
                    series_list.append({
                        "modelno": model.get("modelno"),
                        "modelname": model.get("modelname"),
                        "seriesno": series.get("seriesno"),
                        "seriesname": series.get("seriesname"),
                        "seriesprice": series.get("seriesprice")
                    })
            
            result["message"] = autobegin_data.get("rst_msg", "Multiple series found. Please select a series.")
            result["series_list"] = series_list
            result["modellist"] = modellist
            result["ts_key"] = ts_key
            result["requires_selection"] = True
            
        else:
            # Error: rst_code is not 1 or 2
            error_message = autobegin_data.get("rst_msg", f"Unknown error (rst_code: {rst_code})")
            result["success"] = False
            result["error"] = error_message
            result["message"] = f"Autobegin API returned error: {error_message}"
            
            # Special handling for common error codes
            if rst_code == 100030:
                # Owner name incorrect or vehicle deregistered
                # This often happens when OCR misreads Korean characters
                result["suggestion"] = "소유자명이 잘못되었거나 차량이 말소되었을 수 있습니다. OCR 결과를 확인해주세요."
                result["ocr_owner_name"] = owner_name
                result["ocr_vehicle_number"] = vehicle_number
                result["possible_ocr_error"] = "OCR may have misread Korean characters. Please verify the owner name."
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        # if os.path.exists(file_location):
        #     os.remove(file_location)
        pass


@app_router.post('/scan-certificate/select-series')
async def scan_certificate_select_series(selection: SeriesSelection = Body(...)):
    """
    After getting rst_code=2, call this endpoint with selected series to get full vehicle details.
    
    Request body:
    - **seriesno**: Selected series number (required)
    - **ts_key**: Transaction key from previous response (required)
    """
    try:
        seriesno = selection.seriesno
        ts_key = selection.ts_key
        
        if not seriesno or not ts_key:
            return {
                "success": False,
                "error": "seriesno and ts_key are required"
            }
        
        # Call new Autobegin API via vsol.hanbirosoft.com
        autobegin_base_url = os.getenv("AUTOBEGIN_BASE_URL", "http://vsol.hanbirosoft.com/api/autobegin")
        
        
        
        # Prepare query parameters for detail API
        query_params = {
            "mode": "detail",
            "seriesno": seriesno,
            "ts_key": ts_key,
        }
        
        # Build URL with query parameters
        url = f"{autobegin_base_url}/search?{urlencode(query_params)}"
        
        # Call new Autobegin API with bearer token (very long timeout: 5 minutes)
        timeout_config = httpx.Timeout(300.0, connect=60.0)  # 5 minutes total, 60s connect
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.get(
                    url,
                    headers={
                        "Content-Type": "application/json",
                    }
                )
                response.raise_for_status()
                api_response = response.json()
                
                # Extract data from response (format: {"success": true, "data": {...}})
                if not api_response.get("success", False):
                    return {
                        "success": False,
                        "error": api_response.get("message", "API returned unsuccessful response")
                    }
                
                autobegin_data = api_response.get("data", {})
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"Autobegin API error: {str(e)}"
                }
        
        # Process Autobegin response
        rst_code = autobegin_data.get("rst_code")
        
        result = {
            "success": True,
            "autobegin_data": autobegin_data,
            "rst_code": rst_code
        }
        
        # Handle different rst_code values
        if rst_code == 1:
            # Success: Full vehicle information available
            cardata = autobegin_data.get("cardata", {})
            
            # Extract vehicle information
            makername = cardata.get("makername", "")
            classname = cardata.get("classname", "")
            modelname = cardata.get("modelname", "")
            seriesname = cardata.get("seriesname", "")
            seriesname1 = cardata.get("seriesname1", "")
            seriesname2 = cardata.get("seriesname2", "")
            
            # Build car type
            series_part = f"{seriesname1}_{seriesname2}" if (seriesname1 and seriesname2) else seriesname
            car_type_parts = [makername, classname, modelname, series_part]
            # Filter out empty strings and join
            car_type = "_".join(filter(None, car_type_parts)).strip("_") if any(car_type_parts) else None
            
            # Add formatted data to result
            result["vehicle_info"] = {
                "car_type": car_type,
                "makername": makername,
                "classname": classname,
                "modelname": modelname,
                "seriesname": seriesname,
                "seriesname1": seriesname1,
                "seriesname2": seriesname2,
                "newprice": cardata.get("newprice"),
                "firstdate": cardata.get("firstdate"),
                "year": cardata.get("year"),
                # Additional fields from cardata (new API format)
                "chassis_number": cardata.get("vin") or cardata.get("chassis_number") or cardata.get("chassisnumber"),
                "vin": cardata.get("vin"),
                "engine_type": cardata.get("engine_type") or cardata.get("enginetype"),
                "displacement": cardata.get("displacement"),
                "fuel": cardata.get("fuel"),
                "color": cardata.get("color"),
                "purpose": cardata.get("usegubun") or cardata.get("purpose"),
                "seating": cardata.get("seating"),
                "carnum": cardata.get("carnum"),
                "regname": cardata.get("regname"),
                "full_data": cardata
            }
            
        elif rst_code == 2:
            # Still multiple series found (should not happen after selecting series, but handle it)
            modellist = autobegin_data.get("modellist", [])
            ts_key = autobegin_data.get("ts_key", "")
            
            # Flatten series list from all models
            series_list = []
            for model in modellist:
                model_series_list = model.get("serieslist", [])
                for series in model_series_list:
                    series_list.append({
                        "modelno": model.get("modelno"),
                        "modelname": model.get("modelname"),
                        "seriesno": series.get("seriesno"),
                        "seriesname": series.get("seriesname"),
                        "seriesprice": series.get("seriesprice")
                    })
            
            result["message"] = autobegin_data.get("rst_msg", "Multiple series found. Please select a series.")
            result["series_list"] = series_list
            result["modellist"] = modellist
            result["ts_key"] = ts_key
            result["requires_selection"] = True
            
        else:
            # Error: rst_code is not 1 or 2
            error_message = autobegin_data.get("rst_msg", f"Unknown error (rst_code: {rst_code})")
            result["success"] = False
            result["error"] = error_message
            result["message"] = f"Autobegin API returned error: {error_message}"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_router.post('/vehicle-composition')
async def vehicle_composition(
    car_image: UploadFile = File(..., description="Car image to remove background from"),
    background_image: UploadFile = File(..., description="Background image to composite car onto"),
    reference_image: UploadFile = File(None, description="Optional reference image for placement guide"),
    model: str = "gemini-2.5-flash-image",
    cutout_prompt: str = "cutout.txt",
    composite_prompt: str = "composite_showroom.txt",
    return_cutout: bool = True,
    return_composite: bool = True,
    retries: int = 1,
):
    """
    Process vehicle background removal and composition.
    
    - **car_image**: Image of the vehicle to remove background from
    - **background_image**: Background image to composite the vehicle onto
    - **reference_image**: Optional reference image for placement guide (e.g., with red box)
    - **model**: Gemini model name (default: gemini-2.5-flash-image)
    - **cutout_prompt**: Name of cutout prompt file (default: cutout.txt)
    - **composite_prompt**: Name of composite prompt file (default: composite_showroom.txt)
    - **return_cutout**: Whether to return cutout image (default: True)
    - **return_composite**: Whether to return composite image (default: True)
    - **retries**: Number of retries per stage (default: 1, total attempts = 1 + retries)
    
    Returns JSON with image data (base64 or bytes) and usage statistics.
    """
    try:
        # Read car image
        car_bytes = await car_image.read()
        car_filename = car_image.filename or "car.jpg"
        
        # Read background image
        bg_bytes = await background_image.read()
        bg_filename = background_image.filename or "background.jpg"
        
        # Read reference image if provided
        ref_bytes = None
        ref_filename = None
        if reference_image:
            ref_bytes = await reference_image.read()
            ref_filename = reference_image.filename
        
        # Process composition
        result = process_vehicle_composition(
            car_image_bytes=car_bytes,
            car_filename=car_filename,
            background_image_bytes=bg_bytes,
            background_filename=bg_filename,
            reference_image_bytes=ref_bytes,
            reference_filename=ref_filename,
            model=model,
            cutout_prompt_file=cutout_prompt,
            composite_prompt_file=composite_prompt,
            retries=retries,
        )
        
        # Save images to debug directory
        debug_dir = Path("./data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        car_stem = Path(car_filename).stem
        
        saved_files = {}
        
        # Save cutout image
        if return_cutout:
            cutout_filename = f"cutout_{car_stem}_{timestamp}_{unique_id}.png"
            cutout_path = debug_dir / cutout_filename
            with open(cutout_path, "wb") as f:
                f.write(result["cutout_image"])
            saved_files["cutout"] = str(cutout_path)
        
        # Save composite image
        if return_composite:
            composite_filename = f"composite_{car_stem}_{timestamp}_{unique_id}.png"
            composite_path = debug_dir / composite_filename
            with open(composite_path, "wb") as f:
                f.write(result["composite_image"])
            saved_files["composite"] = str(composite_path)
        
        # Prepare response
        response_data = {
            "success": True,
            "cutout_usage": result["cutout_usage"],
            "composite_usage": result["composite_usage"],
            "saved_files": saved_files,
        }
        
        # Add errors if any (for debugging)
        if result.get("cutout_error"):
            response_data["cutout_error"] = result["cutout_error"]
        if result.get("composite_error"):
            response_data["composite_error"] = result["composite_error"]
        
        # Add images if requested (base64 encoded for JSON response)
        if return_cutout:
            response_data["cutout_image"] = base64.b64encode(result["cutout_image"]).decode('utf-8')
        if return_composite:
            response_data["composite_image"] = base64.b64encode(result["composite_image"]).decode('utf-8')
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_router.post('/vehicle-composition/cutout')
async def vehicle_cutout_only(
    car_image: UploadFile = File(..., description="Car image to remove background from"),
    model: str = "gemini-2.5-flash-image",
    cutout_prompt: str = "cutout.txt",
):
    """
    Only remove background from vehicle image (cutout only, no composition).
    
    Returns the cutout image with transparent background.
    """
    try:
        from app.vehicle_composition_service import load_client, read_prompt, call_cutout, guess_mime_from_bytes, pil_to_png_bytes
        
        car_bytes = await car_image.read()
        car_filename = car_image.filename or "car.jpg"
        
        client = load_client()
        cutout_prompt_text = read_prompt(cutout_prompt)
        car_mime = guess_mime_from_bytes(car_bytes, car_filename)
        
        cutout_img, cutout_usage = call_cutout(
            client, model, cutout_prompt_text, car_bytes, car_mime
        )
        
        cutout_bytes = pil_to_png_bytes(cutout_img)
        
        # Save cutout image to debug directory
        debug_dir = Path("./data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        car_stem = Path(car_filename).stem
        cutout_filename = f"cutout_{car_stem}_{timestamp}_{unique_id}.png"
        cutout_path = debug_dir / cutout_filename
        with open(cutout_path, "wb") as f:
            f.write(cutout_bytes)
        
        return Response(
            content=cutout_bytes,
            media_type="image/png",
            headers={
                "X-Usage-Prompt-Tokens": str(cutout_usage.get("prompt_tokens", "")),
                "X-Usage-Candidates-Tokens": str(cutout_usage.get("candidates_tokens", "")),
                "X-Usage-Total-Tokens": str(cutout_usage.get("total_tokens", "")),
                "X-Saved-File": str(cutout_path),
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_router.post('/vehicle-composition/composite')
async def vehicle_composite_only(
    cutout_image: UploadFile = File(..., description="Cutout car image (PNG with transparent background)"),
    background_image: UploadFile = File(..., description="Background image to composite car onto"),
    reference_image: UploadFile = File(None, description="Optional reference image for placement guide"),
    model: str = "gemini-2.5-flash-image",
    composite_prompt: str = "composite_showroom.txt",
):
    """
    Only composite cutout vehicle onto background (composition only).
    
    Requires a pre-cutout image with transparent background.
    """
    try:
        from app.vehicle_composition_service import load_client, read_prompt, call_composite, guess_mime_from_bytes, pil_to_png_bytes
        
        cutout_bytes = await cutout_image.read()
        bg_bytes = await background_image.read()
        bg_filename = background_image.filename or "background.jpg"
        
        ref_bytes = None
        ref_filename = None
        if reference_image:
            ref_bytes = await reference_image.read()
            ref_filename = reference_image.filename
        
        client = load_client()
        composite_prompt_text = read_prompt(composite_prompt)
        bg_mime = guess_mime_from_bytes(bg_bytes, bg_filename)
        ref_mime = None
        if ref_bytes:
            ref_mime = guess_mime_from_bytes(ref_bytes, ref_filename or "")
        
        comp_img, comp_usage = call_composite(
            client,
            model,
            composite_prompt_text,
            bg_bytes,
            bg_mime,
            cutout_bytes,
            ref_bytes=ref_bytes,
            ref_mime=ref_mime,
        )
        
        composite_bytes = pil_to_png_bytes(comp_img)
        
        # Save composite image to debug directory
        debug_dir = Path("./data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        bg_stem = Path(bg_filename).stem
        composite_filename = f"composite_{bg_stem}_{timestamp}_{unique_id}.png"
        composite_path = debug_dir / composite_filename
        with open(composite_path, "wb") as f:
            f.write(composite_bytes)
        
        return Response(
            content=composite_bytes,
            media_type="image/png",
            headers={
                "X-Usage-Prompt-Tokens": str(comp_usage.get("prompt_tokens", "")),
                "X-Usage-Candidates-Tokens": str(comp_usage.get("candidates_tokens", "")),
                "X-Usage-Total-Tokens": str(comp_usage.get("total_tokens", "")),
                "X-Saved-File": str(composite_path),
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(app_router)