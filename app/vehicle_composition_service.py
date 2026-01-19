import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types
import os
from app.config import *

# Load prompts directory
BASE_DIR = Path(__file__).parent
PROMPT_DIR = BASE_DIR / "prompts"


def load_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (check .env)")
    return genai.Client(api_key=api_key)


def read_prompt(prompt_name: str) -> str:
    """Read prompt file from prompts directory"""
    prompt_path = PROMPT_DIR / prompt_name
    if not prompt_path.exists():
        raise RuntimeError(f"Prompt file not found: {prompt_path}")
    txt = prompt_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise RuntimeError(f"Prompt file is empty: {prompt_path}")
    return txt


def guess_mime_from_bytes(data: bytes, filename: str = "") -> str:
    """Guess MIME type from bytes or filename"""
    if filename:
        ext = Path(filename).suffix.lower()
        if ext == ".png":
            return "image/png"
        if ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        if ext == ".webp":
            return "image/webp"
    
    # Try to detect from image header
    if data.startswith(b'\x89PNG'):
        return "image/png"
    if data.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    if data.startswith(b'RIFF') and b'WEBP' in data[:12]:
        return "image/webp"
    
    return "image/jpeg"  # default


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_first_image_from_response(resp) -> Image.Image:
    """Extract first image from Gemini response"""
    cand = resp.candidates[0]
    for part in cand.content.parts:
        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
            return Image.open(io.BytesIO(part.inline_data.data))
    raise RuntimeError("No image returned from Gemini response")


def usage_to_dict(resp) -> Dict[str, Any]:
    """Extract usage metadata from response"""
    u = getattr(resp, "usage_metadata", None)
    if not u:
        return {}
    return {
        "prompt_tokens": getattr(u, "prompt_token_count", None),
        "candidates_tokens": getattr(u, "candidates_token_count", None),
        "total_tokens": getattr(u, "total_token_count", None),
    }


def gemini_generate_image(
    client: genai.Client,
    model: str,
    prompt: str,
    image_parts: list[types.Part],
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Call Gemini API to generate image"""
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=prompt), *image_parts],
            )
        ],
    )
    img = extract_first_image_from_response(resp)
    return img, usage_to_dict(resp)


def call_cutout(
    client: genai.Client,
    model: str,
    cutout_prompt: str,
    car_bytes: bytes,
    car_mime: str,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Remove background from car image"""
    parts = [types.Part.from_bytes(data=car_bytes, mime_type=car_mime)]
    img, usage = gemini_generate_image(client, model, cutout_prompt, parts)
    return img.convert("RGBA"), usage


def call_composite(
    client: genai.Client,
    model: str,
    composite_prompt: str,
    bg_bytes: bytes,
    bg_mime: str,
    cutout_png_bytes: bytes,
    ref_bytes: Optional[bytes] = None,
    ref_mime: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Composite cutout car onto background"""
    parts = [
        types.Part.from_bytes(data=bg_bytes, mime_type=bg_mime),
        types.Part.from_bytes(data=cutout_png_bytes, mime_type="image/png"),
    ]
    # Add reference image if provided
    if ref_bytes and ref_mime:
        parts.append(types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime))

    img, usage = gemini_generate_image(client, model, composite_prompt, parts)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    return img, usage


def process_vehicle_composition(
    car_image_bytes: bytes,
    car_filename: str,
    background_image_bytes: bytes,
    background_filename: str,
    reference_image_bytes: Optional[bytes] = None,
    reference_filename: Optional[str] = None,
    model: str = "gemini-2.5-flash-image",
    cutout_prompt_file: str = "cutout.txt",
    composite_prompt_file: str = "composite_showroom.txt",
) -> Dict[str, Any]:
    """
    Main function to process vehicle background removal and composition
    
    Args:
        car_image_bytes: Car image as bytes
        car_filename: Original car image filename (for MIME detection)
        background_image_bytes: Background image as bytes
        background_filename: Original background image filename
        reference_image_bytes: Optional reference image bytes
        reference_filename: Optional reference image filename
        model: Gemini model name
        cutout_prompt_file: Name of cutout prompt file
        composite_prompt_file: Name of composite prompt file
    
    Returns:
        Dictionary with cutout_image, composite_image (as bytes), and usage stats
    """
    client = load_client()
    
    # Load prompts
    cutout_prompt = read_prompt(cutout_prompt_file)
    composite_prompt = read_prompt(composite_prompt_file)
    
    # Detect MIME types
    car_mime = guess_mime_from_bytes(car_image_bytes, car_filename)
    bg_mime = guess_mime_from_bytes(background_image_bytes, background_filename)
    ref_mime = None
    if reference_image_bytes:
        ref_mime = guess_mime_from_bytes(reference_image_bytes, reference_filename or "")
    
    # Stage 1: Cutout (remove background)
    cutout_img, cutout_usage = call_cutout(
        client, model, cutout_prompt, car_image_bytes, car_mime
    )
    
    # Convert cutout to PNG bytes for composite
    cutout_png_bytes = pil_to_png_bytes(cutout_img)
    
    # Stage 2: Composite
    comp_img, comp_usage = call_composite(
        client,
        model,
        composite_prompt,
        background_image_bytes,
        bg_mime,
        cutout_png_bytes,
        ref_bytes=reference_image_bytes,
        ref_mime=ref_mime,
    )
    
    # Convert results to bytes
    cutout_bytes = pil_to_png_bytes(cutout_img)
    composite_bytes = pil_to_png_bytes(comp_img)
    
    return {
        "cutout_image": cutout_bytes,
        "composite_image": composite_bytes,
        "cutout_usage": cutout_usage,
        "composite_usage": comp_usage,
    }
