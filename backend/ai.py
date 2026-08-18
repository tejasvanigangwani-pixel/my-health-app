"""
Vrikka AI (educational chat) + OCR extraction — both via Claude Sonnet 4.6
through the Emergent universal key. Includes AI safety validation.
"""
import os
import io
import json
import base64
import logging
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from medical import VRIKKA_SYSTEM_PROMPT, validate_ai_output, AI_DISCLAIMER

logger = logging.getLogger(__name__)
EMERGENT_LLM_KEY = os.environ["EMERGENT_LLM_KEY"]
MODEL_PROVIDER = "anthropic"
MODEL_NAME = "claude-sonnet-4-6"


def _new_chat(session_id: str, system_message: str) -> LlmChat:
    return LlmChat(api_key=EMERGENT_LLM_KEY, session_id=session_id,
                   system_message=system_message).with_model(MODEL_PROVIDER, MODEL_NAME)


async def vrikka_reply(session_id: str, history: list[dict], user_text: str) -> dict:
    """
    Educational answer. history: [{role, content}]. Rebuilds context each call
    (in-memory thread is per-restart), then runs the safety validator.
    """
    chat = _new_chat(session_id, VRIKKA_SYSTEM_PROMPT)
    # Replay prior turns as compact context so the model has continuity.
    context_lines = []
    for m in history[-8:]:
        who = "User" if m.get("role") == "user" else "Vrikka AI"
        context_lines.append(f"{who}: {m.get('content','')}")
    prompt = user_text
    if context_lines:
        prompt = ("Conversation so far:\n" + "\n".join(context_lines) +
                  f"\n\nNew user message: {user_text}")
    try:
        raw = await chat.send_message(UserMessage(text=prompt))
        text = raw if isinstance(raw, str) else str(raw)
    except Exception as e:
        logger.exception("Vrikka AI error")
        return {"ok": False, "text": ("Vrikka AI is temporarily unavailable. Please try again shortly.\n\n"
                                      + AI_DISCLAIMER), "safety_flags": ["error"], "blocked": False}
    is_safe, cleaned, flags = validate_ai_output(text)
    return {"ok": True, "text": cleaned, "safety_flags": flags, "blocked": (not is_safe)}


# --------------------------- OCR pipeline ---------------------------
_OCR_PROMPT = (
    "You are an OCR extraction engine for kidney-related laboratory reports. "
    "Extract every lab parameter you can read from this report image. "
    "Return STRICT JSON only (no prose) in this exact shape:\n"
    '{"lab_name": string|null, "report_date": "YYYY-MM-DD"|null, "parameters": '
    '[{"test_name": string, "standard_name": one of '
    "[egfr,creatinine,cystatin_c,urea,bun,albumin,uacr,urine_protein,sodium,potassium,other], "
    '"value": number|null, "unit": string|null, "reference_range": string|null, '
    '"confidence": number between 0 and 1}]}\n'
    "Rules: Never invent values you cannot read. Distinguish urine protein vs urine albumin vs UACR/ACR — "
    "keep the original test name and unit. If unsure about a value, set a low confidence. "
    "Do NOT diagnose or interpret. Output JSON only."
)


def _pdf_to_images(data: bytes) -> list[str]:
    """Render PDF pages to PNG base64 using PyMuPDF. Returns [] if unavailable."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    out = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc[:3]:  # cap at 3 pages
            pix = page.get_pixmap(dpi=150)
            out.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
        doc.close()
    except Exception:
        logger.exception("PDF render failed")
    return out


def _image_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


async def ocr_extract(session_id: str, data: bytes, mime: str) -> dict:
    """Returns {supported, parameters, lab_name, report_date, raw}."""
    images_b64 = []
    if mime == "application/pdf":
        images_b64 = _pdf_to_images(data)
        if not images_b64:
            return {"supported": False,
                    "message": "Automatic reading of this PDF is not available. Please enter the values manually.",
                    "parameters": []}
    elif mime in ("image/jpeg", "image/png", "image/webp"):
        images_b64 = [_image_to_b64(data)]
    else:
        return {"supported": False,
                "message": "This file type can't be read automatically. Please enter values manually.",
                "parameters": []}

    chat = _new_chat(session_id, "You are a precise OCR extraction engine. Output JSON only.")
    imgs = [ImageContent(image_base64=b) for b in images_b64]
    try:
        raw = await chat.send_message(UserMessage(text=_OCR_PROMPT, file_contents=imgs))
        text = raw if isinstance(raw, str) else str(raw)
    except Exception:
        logger.exception("OCR error")
        return {"supported": True, "message": "Couldn't read the report automatically. Please verify or enter values manually.",
                "parameters": []}

    parsed = _extract_json(text)
    if not parsed:
        return {"supported": True, "message": "Couldn't reliably read the report. Please enter values manually.",
                "parameters": []}
    params = parsed.get("parameters", []) or []
    for p in params:
        p["verification_status"] = "unverified"
    return {"supported": True, "lab_name": parsed.get("lab_name"),
            "report_date": parsed.get("report_date"), "parameters": params,
            "message": "Please verify these values before saving."}


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None
