"""Input photos: Wikimedia Commons search / download (with licence metadata), local files,
background removal, contact sheets."""
import json
import os
import re
import shutil
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "car3d-pipeline/0.1 (research script)"}


def _api(params):
    url = API + "?" + urllib.parse.urlencode({"format": "json", **params})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def _meta(page):
    ii = page["imageinfo"][0]
    m = ii.get("extmetadata", {})
    strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
    return dict(title=page["title"], license=strip(m.get("LicenseShortName", {}).get("value")),
                artist=strip(m.get("Artist", {}).get("value")), width=ii.get("width"), height=ii.get("height"),
                thumb=ii.get("thumburl") or ii.get("url"), page=ii.get("descriptionurl"))


def wikimedia_search(query, limit=24, width=1600):
    d = _api(dict(action="query", generator="search", gsrsearch=f"{query} filetype:bitmap", gsrnamespace=6,
                  gsrlimit=limit, prop="imageinfo", iiprop="url|size|extmetadata",
                  iiextmetadatafilter="LicenseShortName|Artist", iiurlwidth=width))
    pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    return [_meta(p) for p in pages if "imageinfo" in p]


def wikimedia_info(title, width=1600):
    if not title.startswith("File:"):
        title = "File:" + title
    d = _api(dict(action="query", titles=title, prop="imageinfo", iiprop="url|size|extmetadata",
                  iiextmetadatafilter="LicenseShortName|Artist", iiurlwidth=width))
    page = list(d["query"]["pages"].values())[0]
    if "imageinfo" not in page:
        raise FileNotFoundError(f"Wikimedia file not found: {title}")
    return _meta(page)


def download(url, out):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    return out


def fetch_image(key, entry, in_dir, config_dir):
    """entry: {wikimedia: 'File:...'} or {path: 'photo.jpg'} -> (jpg path, metadata dict)."""
    out = os.path.join(in_dir, f"{key}.jpg")
    if "wikimedia" in entry:
        meta = wikimedia_info(entry["wikimedia"])
        if not os.path.exists(out):
            download(meta["thumb"], out)
    elif "path" in entry:
        src = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(config_dir, entry["path"])
        Image.open(src).convert("RGB").save(out, quality=95)
        meta = dict(title=os.path.basename(src), license=entry.get("license", "user-supplied"),
                    artist=entry.get("artist", ""), page=src)
    else:
        raise ValueError(f"image '{key}': give 'wikimedia:' or 'path:'")
    meta["view"] = entry.get("view", "")
    return out, meta


_SESSION = None


def remove_background(path, out):
    """BiRefNet (rembg) matte, keep only the largest blob (drops neighbouring cars / people)."""
    global _SESSION
    if os.path.exists(out):
        return Image.open(out)
    import cv2
    import rembg
    img = Image.open(path).convert("RGB")
    s = min(1.0, 2048 / max(img.size))
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
    if _SESSION is None:
        # GPU: onnxruntime finds CUDA 12 / cuDNN from the pip nvidia-* wheels torch installed.
        # On CPU BiRefNet needs ~12 GB RAM (OOM-killed on a 16 GB machine); on GPU ~1 s / 2 GB.
        import onnxruntime as ort
        try:
            ort.preload_dlls()
        except Exception:
            pass
        _SESSION = rembg.new_session("birefnet-general", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    rgba = np.array(rembg.remove(img, session=_SESSION))
    a = rgba[:, :, 3]
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8), 8)
    if n > 2:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        rgba[:, :, 3] = np.where(lab == keep, a, 0)
    im = Image.fromarray(rgba)
    im.save(out)
    return im


def contact_sheet(paths, out, labels=None, cols=4, w=480):
    ch = int(w * 0.66)                       # fixed cell: portrait photos are letterboxed
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((w, ch), Image.Resampling.LANCZOS)
        ims.append(im)
    h = ch + 28
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    for i, im in enumerate(ims):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x + (w - im.width) // 2, y + 28))
        dr.text((x + 6, y + 4), (labels[i] if labels else str(i))[:60], fill="black", font=font)
    sheet.save(out)
    return out


def write_attribution(metas, out):
    lines = ["# Input photo attribution", "",
             "| key | view | file | author | licence |", "|---|---|---|---|---|"]
    for key, m in metas.items():
        lines.append(f"| {key} | {m.get('view', '')} | [{m['title']}]({m.get('page', '')}) | {m.get('artist', '')} | {m.get('license', '')} |")
    sa = [k for k, m in metas.items() if "SA" in (m.get("license") or "")]
    lines += ["", "Derived 3D models inherit the attribution requirement of CC BY / CC BY-SA photos."
              + (f" Share-alike photos used: {', '.join(sa)}." if sa else "")]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
