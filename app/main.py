from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from app.ocr_engine import run_ocr_pipeline
from app.vehicle_composition_service import process_vehicle_composition
import os 
import shutil
import base64
import uuid
from datetime import datetime
from pathlib import Path
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