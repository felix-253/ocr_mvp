from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import Response
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
from app.config import *


class SeriesSelection(BaseModel):
    vehicle_number: str
    owner_name: Optional[str] = ""
    series: Optional[str] = ""
    seriesname1: Optional[str] = ""
    seriesname2: Optional[str] = ""


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
        vehicle_number = ocr_data.get("vehicle_number", {}).get("value", "")
        owner_name = ocr_data.get("owner_name", {}).get("value", "")
        
        if not vehicle_number:
            return {
                "success": False,
                "error": "Vehicle number not found in OCR result",
                "ocr_result": ocr_result
            }
        
        # Call Autobegin API
        autobegin_url = os.getenv("AUTOBEGINS_URL", "")
        autobegin_key = os.getenv("AUTOBEGINS_KEY", "")
        
        if not autobegin_url or not autobegin_key:
            return {
                "success": False,
                "error": "Autobegin API configuration missing (AUTOBEGINS_URL, AUTOBEGINS_KEY)",
                "ocr_result": ocr_result
            }
        
        # Prepare query parameters
        query_params = {
            "key": autobegin_key,
            "mode": "search",
            "carName": vehicle_number,  # Using vehicle_number as carName
        }
        
        # Add owner if available
        if owner_name:
            query_params["owner"] = owner_name
        
        # Build URL with query parameters
        url = f"{autobegin_url}?{urlencode(query_params)}"
        
        # Call Autobegin API
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    url,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                autobegin_data = response.json()
                print("autobegin_data",response.json())
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
                # Additional fields from cardata
                "chassis_number": cardata.get("chassis_number") or cardata.get("chassisnumber"),
                "engine_type": cardata.get("engine_type") or cardata.get("enginetype"),
                "fuel": cardata.get("fuel"),
                "color": cardata.get("color"),
                "purpose": cardata.get("purpose"),
                "full_data": cardata
            }
            
        elif rst_code == 2:
            # Multiple series found: Return list for user to select
            # The response should contain a list of series options
            series_list = autobegin_data.get("series_list", [])
            result["message"] = "Multiple series found. Please select a series."
            result["series_list"] = series_list
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
    finally:
        # Clean up uploaded file
        if os.path.exists(file_location):
            os.remove(file_location)
            print("DELETE FILE", file_location)


@app_router.post('/scan-certificate/select-series')
async def scan_certificate_select_series(selection: SeriesSelection = Body(...)):
    """
    After getting rst_code=2, call this endpoint with selected series to get full vehicle details.
    
    Request body:
    - **vehicle_number**: Vehicle registration number (required)
    - **owner_name**: Owner name (optional)
    - **series**: Selected series name (optional)
    - **seriesname1**: Series name part 1 (optional)
    - **seriesname2**: Series name part 2 (optional)
    """
    try:
        vehicle_number = selection.vehicle_number
        owner_name = selection.owner_name or ""
        series = selection.series or ""
        seriesname1 = selection.seriesname1 or ""
        seriesname2 = selection.seriesname2 or ""
        # Call Autobegin API
        autobegin_url = os.getenv("AUTOBEGINS_URL", "")
        autobegin_key = os.getenv("AUTOBEGINS_KEY", "")
        
        if not autobegin_url or not autobegin_key:
            return {
                "success": False,
                "error": "Autobegin API configuration missing (AUTOBEGINS_URL, AUTOBEGINS_KEY)"
            }
        
        # Prepare query parameters
        query_params = {
            "key": autobegin_key,
            "mode": "search",
            "carName": vehicle_number,
        }
        
        # Add owner if available
        if owner_name:
            query_params["owner"] = owner_name
        
        # Add series selection
        if series:
            query_params["series"] = series
        if seriesname1:
            query_params["seriesname1"] = seriesname1
        if seriesname2:
            query_params["seriesname2"] = seriesname2
        
        # Build URL with query parameters
        url = f"{autobegin_url}?{urlencode(query_params)}"
        
        # Call Autobegin API
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    url,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                autobegin_data = response.json()
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": f"Autobegin API error: {str(e)}"
                }
        
        # Process Autobegin response
        rst_code = autobegin_data.get("rst_code")
        
        result = {
            "success": True,
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
                # Additional fields from cardata
                "chassis_number": cardata.get("chassis_number") or cardata.get("chassisnumber"),
                "engine_type": cardata.get("engine_type") or cardata.get("enginetype"),
                "fuel": cardata.get("fuel"),
                "color": cardata.get("color"),
                "purpose": cardata.get("purpose"),
                "full_data": cardata
            }
            
        elif rst_code == 2:
            # Still multiple series found
            series_list = autobegin_data.get("series_list", [])
            result["message"] = "Multiple series found. Please select a series."
            result["series_list"] = series_list
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