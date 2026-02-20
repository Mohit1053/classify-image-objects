from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import List, Optional
import traceback
import pandas as pd
import io
import sys
import os
import requests
from urllib.parse import urlparse
import re

# Ensure local imports work when running as a top-level script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your classifier + labels (NO relative dot import)
from file1 import EnsembleClassifier, PICKUP_CLASSES

app = FastAPI(title="Scrap Image Classifier API")
classifier = EnsembleClassifier()


def _result_item(
    image_url: str,
    predicted_class: Optional[str],
    confidence: Optional[float],
    blip_description: Optional[str],
    failure_reason: Optional[str] = None,
):
    item = {
        "image_url": image_url,
        "predicted_class": predicted_class if predicted_class is not None else None,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else (float(confidence) if confidence else None),
        "blip_description": blip_description or "",
    }
    if failure_reason:
        item["failure_reason"] = failure_reason
    return item


def _looks_like_url(x: str) -> bool:
    x = (x or "").strip().lower()
    return x.startswith("http://") or x.startswith("https://")


def _firebase_hint(url: str) -> Optional[str]:
    """
    Return a quick hint if a Firebase Storage URL looks malformed.
    """
    if "firebasestorage.googleapis.com" not in url:
        return None
    if "alt=media" not in url:
        return "Firebase URL missing 'alt=media' parameter"
    if "token=" not in url:
        return "Firebase URL missing 'token=' parameter (signed URL likely truncated)"
    # token should usually look like a UUID; we do a light sanity check:
    m = re.search(r"[?&]token=([0-9a-fA-F-]{32,})", url)
    if not m:
        return "Firebase token param looks malformed/truncated"
    return None


def _preflight_url(url: str, timeout: float = 15.0):
    """
    Return (ok: bool, reason: str|None, status: int|None).
    Use GET (not HEAD) since some CDNs reject HEAD or give 404s.
    Stream to keep it lightweight.
    """
    try:
        u = (url or "").strip()
        if not _looks_like_url(u):
            return False, "invalid URL", None

        # Quick Firebase hints
        hint = _firebase_hint(u)
        # Lightweight GET
        r = requests.get(
            u, stream=True, allow_redirects=True, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if 200 <= r.status_code < 400:
            return True, None, r.status_code
        # If 404/403/etc, return with hint if any
        if hint and r.status_code in (400, 401, 403, 404):
            return False, f"HTTP {r.status_code} {r.reason} ({hint})", r.status_code
        return False, f"HTTP {r.status_code} {r.reason}", r.status_code

    except requests.exceptions.RequestException as e:
        return False, f"RequestError: {e}", None


def _parse_csv_first_column(raw_bytes: bytes) -> List[str]:
    """
    Robust CSV parsing:
    - Accept header or no header
    - Use only the first column
    - Trim whitespace
    - Drop empties
    - Deduplicate while preserving order
    - If first row isn't a URL (e.g., a header like 'imageUrls'), drop it
    """
    # Try reading with header; fall back to no header
    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            engine="python",
            dtype=str,
            encoding_errors="ignore",
            keep_default_na=False,  # don't convert 'NA' to NaN (tokens might contain 'NA')
        )
    except Exception:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            engine="python",
            header=None,
            dtype=str,
            encoding_errors="ignore",
            keep_default_na=False,
        )

    # Always take first column
    first_col = df.columns[0]
    s = df[first_col].astype(str)

    # If first row is not URL-like (e.g., header text), drop it
    if len(s) > 0 and not _looks_like_url(s.iloc[0]):
        s = s.iloc[1:]

    # Trim -> drop blanks -> dedupe
    s = s.map(lambda x: x.strip())
    s = s.replace({"": pd.NA}).dropna()
    s = s[~s.duplicated(keep="first")]

    # Final sanity filter: keep only URL-like entries
    urls = [u for u in s.tolist() if _looks_like_url(u)]
    return urls


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "models": list(classifier.models.keys())}


@app.post("/classify-url/")
async def classify_image_url(image_url: str, debug: bool = Query(False, description="Return debug info")):
    """
    Classify a single image URL.
    Returns SAME envelope as /classify-file:
    { "results": [ { "image_url", "predicted_class", "confidence", "blip_description", ... } ] }
    """
    try:
        ok, why, status = _preflight_url(image_url)
        if not ok:
            item = _result_item(
                image_url=image_url,
                predicted_class=None,
                confidence=None,
                blip_description="",
                failure_reason=why or "url preflight failed",
            )
            if debug:
                item["debug"] = {"status": status, "url_len": len(image_url)}
            return {"results": [item]}

        predicted_class, confidence, model_predictions, blip_description = classifier.process_image(
            image_url,
            pickup_classes=PICKUP_CLASSES,
        )

        if predicted_class is None:
            item = _result_item(
                image_url=image_url,
                predicted_class=None,
                confidence=None,
                blip_description=blip_description,
                failure_reason="unreachable image or low signal",
            )
            if debug:
                item["debug"] = {"status": status, "url_len": len(image_url)}
            return {"results": [item]}

        item = _result_item(
            image_url=image_url,
            predicted_class=predicted_class,
            confidence=confidence,
            blip_description=blip_description,
        )
        if debug:
            item["debug"] = {"status": status, "url_len": len(image_url)}
        return {"results": [item]}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"/classify-url failed: {e}")


@app.post("/classify-file/")
async def classify_file(file: UploadFile = File(...), debug: bool = Query(False, description="Return debug info")):
    """
    CSV whose FIRST COLUMN has image URLs.
    Robust parsing + per-row error isolation.
    Returns:
    {
      "parsed_count": N,
      "results": [ one item per URL ... ]
    }
    """
    try:
        raw = await file.read()
        urls = _parse_csv_first_column(raw)

        results = []
        for image_url in urls:
            try:
                ok, why, status = _preflight_url(image_url)
                if not ok:
                    item = _result_item(
                        image_url=image_url,
                        predicted_class=None,
                        confidence=None,
                        blip_description="",
                        failure_reason=why or "url preflight failed",
                    )
                    if debug:
                        item["debug"] = {"status": status, "url_len": len(image_url)}
                    results.append(item)
                    continue

                predicted_class, confidence, model_predictions, blip_description = classifier.process_image(
                    image_url,
                    pickup_classes=PICKUP_CLASSES,
                )

                if predicted_class is None:
                    item = _result_item(
                        image_url=image_url,
                        predicted_class=None,
                        confidence=None,
                        blip_description=blip_description,
                        failure_reason="unreachable image or low signal",
                    )
                    if debug:
                        item["debug"] = {"status": status, "url_len": len(image_url)}
                    results.append(item)
                else:
                    item = _result_item(
                        image_url=image_url,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        blip_description=blip_description,
                    )
                    if debug:
                        item["debug"] = {"status": status, "url_len": len(image_url)}
                    results.append(item)

            except Exception as row_err:
                item = _result_item(
                    image_url=image_url,
                    predicted_class=None,
                    confidence=None,
                    blip_description="",
                    failure_reason=f"{type(row_err).__name__}: {row_err}",
                )
                if debug:
                    item["debug"] = {"url_len": len(image_url)}
                results.append(item)

        resp = {"parsed_count": len(urls), "results": results}
        if debug:
            resp["parsed_urls"] = [{"url": u, "len": len(u)} for u in urls]
        return resp

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to read/process CSV: {e}")


# Run with: uvicorn main:app --reload
# Ensure file1.py is in the same folder and that imports above remain absolute.

 
 #   P r o d u c t i o n   s t a r t u p 
 i f   _ _ n a m e _ _   = =   " _ _ m a i n _ _ " : 
         i m p o r t   u v i c o r n 
         i m p o r t   o s 
         f r o m   d o t e n v   i m p o r t   l o a d _ d o t e n v 
         
         l o a d _ d o t e n v ( ) 
         
         h o s t   =   o s . g e t e n v ( " H O S T " ,   " 0 . 0 . 0 . 0 " ) 
         p o r t   =   i n t ( o s . g e t e n v ( " P O R T " ,   " 8 0 0 1 " ) ) 
         
         p r i n t ( f "   S t a r t i n g   I m a g e   C l a s s i f i c a t i o n   A P I   s e r v e r   o n   h t t p : / / { h o s t } : { p o r t } " ) 
         p r i n t ( f "   A P I   d o c s   a v a i l a b l e   a t :   h t t p : / / { h o s t } : { p o r t } / d o c s " ) 
         
         u v i c o r n . r u n ( 
                 " m a i n : a p p " , 
                 h o s t = h o s t , 
                 p o r t = p o r t , 
                 r e l o a d = F a l s e ,     #   D i s a b l e   r e l o a d   i n   p r o d u c t i o n 
                 l o g _ l e v e l = " i n f o " 
         ) 
 
 