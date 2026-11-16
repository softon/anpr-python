# benchmark_anpr_excel.py
import sys
sys.path.insert(0, "/opt/traffic")
try:
    import offline_env  # MUST be first
except ImportError:
    pass

import cv2
import re
import os
import tempfile
import numpy as np
from pathlib import Path
import paddlex
from paddleocr import TextDetection, TextRecognition
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.styles import Font, PatternFill, Alignment

# Disable remote model checks when running locally
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

OUTPUT_DIR = Path("./anpr_benchmark_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# High-quality thumbnail display settings for Excel
EXCEL_IMG_HEIGHT = 100   # Display height in Excel (pixels)
SOURCE_IMG_HEIGHT = 280  # High-DPI source resolution (prevents blurriness/pixelation)

# All 36 Indian States / Union Territories + Bharat (BH) Series
INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "DNH", "DL", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TG", "TR",
    "UP", "UK", "UA", "WB", "BH"
}

# Unwanted text common on Indian commercial vehicles (trucks, buses, autos)
UNWANTED_WORDS = {
    "GOODS", "CARRIER", "TATA", "LEYLAND", "MAHINDRA", "MARUTI", "EICHER",
    "HORN", "PLEASE", "STOP", "NATIONAL", "PERMIT", "ALLINDIA", "ALL", "INDIA",
    "SPEED", "KMPH", "DIESEL", "PETROL", "CNG", "POLICE", "ARMY", "GOVT",
    "FASTAG", "GOLD", "SUPER", "POWER", "TURBO", "KING", "BOSS", "BHARAT"
}

# Common OCR character confusions between letters and digits in plates
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
LETTER_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "T": "1", "J": "1",
    "Z": "2", "S": "5", "G": "6", "B": "8"
}


def clean_ocr(text):
    """Strip spaces, special symbols, and HSRP 'IND'/'INO' prefix."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if (t.startswith("IND") or t.startswith("INO") or t.startswith("1ND")) and len(t) > 7:
        t = t[3:]
    return t


def normalize_indian_plate(text):
    """
    Validates and disambiguates Indian number plate strings.
    Handles:
      - Standard: State(2) + RTO(1-2) + Series(0-3) + Number(4) (e.g., MH03EL8407, KL07CD1234, DL11BF1339)
      - Bharat Series: YY BH #### XX (e.g., 22BH1234AA)
      - OCR letter/digit position corrections and thin-crossbar 'H' confusions (MII, M11, MI1 -> MH, NH -> MH)
    Returns: (cleaned_plate_str, is_valid)
    """
    t = clean_ocr(text)
    if len(t) < 7 or len(t) > 11:
        return t, False

    # Filter out known commercial truck slogans, phone numbers, or vehicle brands
    for w in UNWANTED_WORDS:
        if w in t:
            return t, False
    if re.search(r"(\d{10}|MO\d+|MOB\d+)", t):
        return t, False

    # Disambiguate common OCR confusions in state codes (especially H with thin crossbar)
    if t.startswith("MII") or t.startswith("M11") or t.startswith("MI1") or t.startswith("M1I"):
        t = "MH" + t[3:]
    elif t.startswith("NH"):
        t = "MH" + t[2:]

    # 1. Bharat (BH) Series
    m_bh = re.match(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$", t)
    if m_bh:
        return t, True

    # 2. Standard State Code Disambiguation
    s1 = DIGIT_TO_LETTER.get(t[0], t[0])
    s2 = DIGIT_TO_LETTER.get(t[1], t[1])
    state = s1 + s2
    if state not in INDIAN_STATES:
        return t, False

    rem = t[2:]
    if len(rem) >= 5:
        # Last 4 characters are registration digits (0001 - 9999)
        num_part = "".join(LETTER_TO_DIGIT.get(c, c) for c in rem[-4:])
        mid = rem[:-4]

        # First 1 or 2 characters of mid are RTO district digits
        rto_len = 2 if len(mid) >= 2 and (mid[1].isdigit() or mid[1] in LETTER_TO_DIGIT) else 1
        rto_part = "".join(LETTER_TO_DIGIT.get(c, c) for c in mid[:rto_len])

        # Remaining characters of mid are series letters (0 to 3 letters)
        series_part = "".join(DIGIT_TO_LETTER.get(c, c) for c in mid[rto_len:])

        candidate = f"{state}{rto_part}{series_part}{num_part}"
        if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$", candidate):
            return candidate, True

    return t, False


# Initialize Models on GPU
print("Loading GPU models (Vehicle Detector, Bike/Motorcycle Detector, Text Detector, Text Recognizer)...")
VEHICLE_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PP-YOLOE-S_vehicle"
PICODET_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PicoDet-S"
DET_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PP-OCRv6_medium_det"
REC_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PP-OCRv6_medium_rec"

# 1. Primary 4-Wheel Vehicle Detector
v_kwargs = {"model_name": "PP-YOLOE-S_vehicle", "device": "gpu:0"}
if Path(VEHICLE_MODEL_DIR).exists():
    v_kwargs["model_dir"] = VEHICLE_MODEL_DIR
vehicle_det = paddlex.create_model(**v_kwargs)

# 2. Bike & Two-Wheeler Detector (detects motorcycle, bicycle, and seated riders)
p_kwargs = {"model_name": "PicoDet-S", "device": "gpu:0"}
if Path(PICODET_MODEL_DIR).exists():
    p_kwargs["model_dir"] = PICODET_MODEL_DIR
bike_det = paddlex.create_model(**p_kwargs)

# 3. High-Accuracy Text Detector & Recognizer
det_kwargs = {"model_name": "PP-OCRv6_medium_det", "device": "gpu:0"}
if Path(DET_MODEL_DIR).exists():
    det_kwargs["model_dir"] = DET_MODEL_DIR
text_det = TextDetection(**det_kwargs)

rec_kwargs = {"model_name": "PP-OCRv6_medium_rec", "device": "gpu:0"}
if Path(REC_MODEL_DIR).exists():
    rec_kwargs["model_dir"] = REC_MODEL_DIR
text_rec = TextRecognition(**rec_kwargs)
print("✅ All 4 GPU models loaded successfully on GPU!\n")


def compute_iou(boxA, boxB):
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def extract_unified_detections(frame):
    """
    Combines PP-YOLOE-S_vehicle (for 4-wheel vehicles) and PicoDet-S
    (for motorcycles, scooters, bicycles, and any missed vehicles).
    For two-wheelers, merges the motorcycle box with any seated rider.
    Returns: list of dicts: {"coordinate": [x1, y1, x2, y2], "label": "Vehicle"|"Motorcycle", "score": float}
    """
    unified = []

    # 1. 4-wheel vehicle detections
    v_res = list(vehicle_det.predict(frame, batch_size=1))[0]
    for b in v_res.json["res"]["boxes"]:
        if b.get("score", 0) >= 0.45:
            unified.append({
                "coordinate": b["coordinate"],
                "label": "Vehicle",
                "score": float(b["score"])
            })

    # 2. Bike & motorcycle detections
    p_res = list(bike_det.predict(frame, batch_size=1))[0]
    p_boxes = p_res.json["res"]["boxes"]

    bike_boxes = [b for b in p_boxes if b.get("label") in {"motorcycle", "bicycle"} and b.get("score", 0) >= 0.35]
    person_boxes = [b for b in p_boxes if b.get("label") == "person" and b.get("score", 0) >= 0.35]
    other_veh_boxes = [b for b in p_boxes if b.get("label") in {"car", "truck", "bus"} and b.get("score", 0) >= 0.45]

    used_persons = set()
    for b in bike_boxes:
        bx1, by1, bx2, by2 = [int(v) for v in b["coordinate"]]
        # Merge with rider (person box directly above or overlapping bike)
        for p_idx, p in enumerate(person_boxes):
            px1, py1, px2, py2 = [int(v) for v in p["coordinate"]]
            if not (px2 < bx1 or px1 > bx2) and (py2 >= by1 - 60 and py1 < by2):
                bx1 = min(bx1, px1)
                by1 = min(by1, py1)
                bx2 = max(bx2, px2)
                by2 = max(by2, py2)
                used_persons.add(p_idx)
        unified.append({
            "coordinate": [bx1, by1, bx2, by2],
            "label": "Motorcycle",
            "score": float(b.get("score", 0))
        })

    # Include standalone person detections on the roadway (riders whose bike was occluded/missed)
    for p_idx, p in enumerate(person_boxes):
        if p_idx in used_persons:
            continue
        px1, py1, px2, py2 = [int(v) for v in p["coordinate"]]
        pw, ph = px2 - px1, py2 - py1
        if ph >= 90 and py2 > frame.shape[0] * 0.25:
            expand_bottom = int(ph * 0.25)
            expand_sides = int(pw * 0.15)
            bx1 = max(0, px1 - expand_sides)
            bx2 = min(frame.shape[1], px2 + expand_sides)
            by1 = max(0, py1)
            by2 = min(frame.shape[0], py2 + expand_bottom)
            unified.append({
                "coordinate": [bx1, by1, bx2, by2],
                "label": "Motorcycle",
                "score": float(p.get("score", 0))
            })

    # Add other vehicles from PicoDet only if not already detected by PP-YOLOE
    for ob in other_veh_boxes:
        coord = ob["coordinate"]
        if not any(compute_iou(coord, ub["coordinate"]) > 0.45 for ub in unified if ub["label"] == "Vehicle"):
            unified.append({
                "coordinate": coord,
                "label": "Vehicle",
                "score": float(ob.get("score", 0))
            })

    return unified


def extract_plates_from_crop(crop, v_type):
    """
    Detects text, validates single-line plates, and conditionally pairs
    two-line stacked plates without corrupting valid plates with adjacent badges.
    Returns: list of tuples: (normalized_plate, conf, (x1, y1, x2, y2), raw_text)
    """
    h, w = crop.shape[:2]
    t_res = list(text_det.predict(crop, batch_size=1))[0]
    polys = t_res.json["res"]["dt_polys"]
    if not polys:
        return []

    # 1. Inspect and OCR each text candidate polygon
    items = []
    for poly in polys:
        parr = np.asarray(poly)
        px1, py1 = parr.min(axis=0).astype(int)
        px2, py2 = parr.max(axis=0).astype(int)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        bw, bh = px2 - px1, py2 - py1
        rel_y = py1 / max(1, h)

        if v_type == "Vehicle":
            # On cars/trucks, plates are mounted on bumper/grille (lower 60%)
            # This suppresses commercial truck roof text, slogans, and windshield banners
            if rel_y < 0.40 or bw > 0.65 * w or bw < 28 or bh < 10:
                continue
        else:
            # On bikes, plates can be on handlebar / front apron / mudguard or rear
            if rel_y < 0.05 or bw > 0.85 * w or bw < 16 or bh < 8:
                continue

        c = crop[py1:py2, px1:px2]
        if c.size == 0:
            continue
        rec_res = list(text_rec.predict(c, batch_size=1))[0]
        raw_text = rec_res.json["res"]["rec_text"]
        conf = float(rec_res.json["res"]["rec_score"])
        norm_text, is_valid = normalize_indian_plate(raw_text)

        items.append({
            "box": (px1, py1, px2, py2),
            "raw": raw_text,
            "conf": conf,
            "norm": norm_text,
            "is_valid": is_valid
        })

    found = []
    used = set()

    # Step A: Accept intact, complete single-line plates immediately
    # (prevents brand badges like 'HONDA', '150', 'TATA' from corrupting complete plates)
    for idx, it in enumerate(items):
        if it["is_valid"] and it["conf"] >= 0.50:
            found.append((it["norm"], it["conf"], it["box"], it["raw"]))
            used.add(idx)

    # Step B: Pair two-line stacked plates for remaining unused text boxes
    # (e.g. commercial trucks: top 'MH 05', bottom 'FJ 9670')
    items_sorted = sorted([i for i in range(len(items)) if i not in used], key=lambda i: items[i]["box"][1])
    for i in range(len(items_sorted)):
        idx1 = items_sorted[i]
        if idx1 in used:
            continue
        b1 = items[idx1]["box"]
        for j in range(i + 1, len(items_sorted)):
            idx2 = items_sorted[j]
            if idx2 in used:
                continue
            b2 = items[idx2]["box"]

            vert_dist = b2[1] - b1[3]
            c1 = (b1[0] + b1[2]) / 2.0
            c2 = (b2[0] + b2[2]) / 2.0
            center_diff = abs(c1 - c2)
            bw1 = b1[2] - b1[0]
            bw2 = b2[2] - b2[0]
            bh1 = b1[3] - b1[1]

            if -10 <= vert_dist <= bh1 * 2.2 and center_diff < max(bw1, bw2) * 0.65:
                comb_raw = items[idx1]["raw"] + items[idx2]["raw"]
                comb_norm, is_val = normalize_indian_plate(comb_raw)
                comb_conf = (items[idx1]["conf"] + items[idx2]["conf"]) / 2.0
                if is_val and comb_conf >= 0.50:
                    mx1 = min(b1[0], b2[0])
                    my1 = min(b1[1], b2[1])
                    mx2 = max(b1[2], b2[2])
                    my2 = max(b1[3], b2[3])
                    found.append((comb_norm, comb_conf, (mx1, my1, mx2, my2), comb_raw))
                    used.add(idx1)
                    used.add(idx2)
                    break

    return found


def process_video(video_path, every_n_frames=6):
    cap = cv2.VideoCapture(str(video_path))
    unique_plates = {}
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % every_n_frames != 0:
            continue

        h_f, w_f = frame.shape[:2]

        # Step 1: Detect vehicles & motorcycles in the frame
        v_boxes = extract_unified_detections(frame)

        for vb in v_boxes:
            v_type = vb["label"]  # "Vehicle" or "Motorcycle"
            vx1, vy1, vx2, vy2 = [int(v) for v in vb["coordinate"]]
            vw_raw, vh_raw = vx2 - vx1, vy2 - vy1

            # Ignore tiny detections (too small for license plate readability)
            min_dim = 45 if v_type == "Motorcycle" else 75
            if vw_raw < min_dim or vh_raw < min_dim:
                continue

            # Add margin around vehicle/bike
            pad_ratio = 0.06 if v_type == "Motorcycle" else 0.04
            pad_vx = int(vw_raw * pad_ratio)
            pad_vy = int(vh_raw * pad_ratio)
            vx1 = max(0, vx1 - pad_vx)
            vy1 = max(0, vy1 - pad_vy)
            vx2 = min(w_f, vx2 + pad_vx)
            vy2 = min(h_f, vy2 + pad_vy)
            vw, vh = vx2 - vx1, vy2 - vy1

            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue

            # Step 2: Extract validated plates from vehicle crop
            detected_plates = extract_plates_from_crop(vehicle_crop, v_type)

            for norm_plate, conf, (px1, py1, px2, py2), raw_text in detected_plates:
                pad_x = max(int((px2 - px1) * 0.08), 2)
                pad_y = max(int((py2 - py1) * 0.10), 2)
                pcx1 = max(0, px1 - pad_x)
                pcy1 = max(0, py1 - pad_y)
                pcx2 = min(vw, px2 + pad_x)
                pcy2 = min(vh, py2 + pad_y)
                plate_crop = vehicle_crop[pcy1:pcy2, pcx1:pcx2]

                if plate_crop.size == 0:
                    continue

                # Create high-quality vehicle thumbnail with detected plate highlighted
                vehicle_disp = vehicle_crop.copy()
                box_thickness = max(2, int(min(vw, vh) / 80))
                cv2.rectangle(vehicle_disp, (pcx1, pcy1), (pcx2, pcy2), (0, 255, 0), box_thickness)

                record = {
                    "video": Path(video_path).name,
                    "vehicle_type": v_type,
                    "frame": frame_id,
                    "vehicle_crop": vehicle_disp,
                    "crop": plate_crop,
                    "ocr_text": norm_plate,
                    "raw_ocr": raw_text,
                    "confidence": round(conf, 3),
                    "valid_format": True,
                    "plate_width_px": pcx2 - pcx1,
                }

                # STRICT DEDUPLICATION: Record each number plate only ONCE
                if norm_plate not in unique_plates:
                    unique_plates[norm_plate] = {
                        "best_record": record,
                        "count": 1,
                        "first_frame": frame_id,
                        "last_frame": frame_id,
                    }
                else:
                    unique_plates[norm_plate]["count"] += 1
                    unique_plates[norm_plate]["last_frame"] = frame_id
                    # Keep the clearest, highest-confidence image of the plate and vehicle
                    if conf > unique_plates[norm_plate]["best_record"]["confidence"]:
                        unique_plates[norm_plate]["best_record"] = record

    cap.release()
    return list(unique_plates.values())


def make_high_res_excel_image(img, tmp_files, display_height=EXCEL_IMG_HEIGHT, source_height=SOURCE_IMG_HEIGHT, prefix="img"):
    """
    Downsamples the source image using INTER_AREA to a high-DPI resolution
    and configures OpenPyXL display dimensions for razor-sharp rendering in Excel.
    """
    h, w = img.shape[:2]
    target_w = max(int(w * source_height / max(h, 1)), 1)
    high_res = cv2.resize(img, (target_w, source_height), interpolation=cv2.INTER_AREA)

    tmp_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{os.getpid()}_{len(tmp_files)}.jpg")
    cv2.imwrite(tmp_path, high_res, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    tmp_files.append(tmp_path)

    xl_img = XlImage(tmp_path)
    xl_img.height = display_height
    xl_img.width = max(int(w * display_height / max(h, 1)), 1)
    return xl_img


def main():
    all_unique_plates = {}
    video_files = list(Path("./sample_videos").glob("*.mp4"))
    if not video_files:
        video_files = sorted(Path(".").glob("*.mp4"))

    for video in video_files:
        print(f"🎬 Processing {video.name}...")
        video_results = process_video(video)
        for item in video_results:
            plate = item["best_record"]["ocr_text"]
            if plate not in all_unique_plates:
                all_unique_plates[plate] = item
            else:
                all_unique_plates[plate]["count"] += item["count"]
                if item["best_record"]["confidence"] > all_unique_plates[plate]["best_record"]["confidence"]:
                    all_unique_plates[plate]["best_record"] = item["best_record"]
                all_unique_plates[plate]["last_frame"] = item["last_frame"]

    # Build Professional Excel Report (Each Number Plate Recorded Only Once)
    print(f"\n📊 Building Excel with {len(all_unique_plates)} unique vehicle number plates...")
    wb = Workbook()
    tmp_files = []

    ws = wb.active
    ws.title = "Vehicle Number Plates"

    headers = [
        "Vehicle Image", "Plate Image", "Video", "Vehicle Type", "Plate Number",
        "Confidence", "Detection Count", "Best Frame", "Frame Range", "Plate Width (px)"
    ]
    ws.append(headers)

    # Style Header Row
    header_fill = PatternFill(start_color="0D47A1", end_color="0D47A1", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    for col_idx in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col_idx)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    # Generous column widths to display high-resolution images cleanly
    ws.column_dimensions["A"].width = 28  # Vehicle Image
    ws.column_dimensions["B"].width = 34  # Plate Image
    ws.column_dimensions["C"].width = 16  # Video
    ws.column_dimensions["D"].width = 18  # Vehicle Type
    ws.column_dimensions["E"].width = 22  # Plate Number
    ws.column_dimensions["F"].width = 14  # Confidence
    ws.column_dimensions["G"].width = 16  # Detection Count
    ws.column_dimensions["H"].width = 14  # Best Frame
    ws.column_dimensions["I"].width = 20  # Frame Range
    ws.column_dimensions["J"].width = 16  # Plate Width

    # Sort unique plates chronologically by the frame they first appeared in the video
    sorted_plates = sorted(all_unique_plates.values(), key=lambda x: x["first_frame"])

    for row_idx, item in enumerate(sorted_plates, start=2):
        r = item["best_record"]

        # 1. High-resolution Vehicle image with detected plate highlighted
        xl_v_img = make_high_res_excel_image(r["vehicle_crop"], tmp_files, prefix=f"veh_{row_idx}")
        ws.add_image(xl_v_img, f"A{row_idx}")

        # 2. High-resolution Plate image
        xl_p_img = make_high_res_excel_image(r["crop"], tmp_files, prefix=f"plate_{row_idx}")
        ws.add_image(xl_p_img, f"B{row_idx}")
        ws.row_dimensions[row_idx].height = EXCEL_IMG_HEIGHT * 0.75

        ws.cell(row=row_idx, column=3, value=r["video"]).alignment = Alignment(vertical="center")

        # Vehicle Type badge
        type_cell = ws.cell(row=row_idx, column=4, value=r.get("vehicle_type", "Vehicle"))
        type_cell.alignment = Alignment(horizontal="center", vertical="center")
        if r.get("vehicle_type") == "Motorcycle":
            type_cell.font = Font(bold=True, color="E65100")  # Orange for bikes
        else:
            type_cell.font = Font(bold=True, color="1565C0")  # Blue for 4-wheel vehicles

        text_cell = ws.cell(row=row_idx, column=5, value=r["ocr_text"])
        text_cell.font = Font(bold=True, size=12)
        text_cell.alignment = Alignment(vertical="center")

        conf_cell = ws.cell(row=row_idx, column=6, value=r["confidence"])
        conf_cell.alignment = Alignment(vertical="center")
        if r["confidence"] >= 0.85:
            conf_cell.font = Font(color="1B5E20", bold=True)
        else:
            conf_cell.font = Font(color="0D47A1")

        ws.cell(row=row_idx, column=7, value=item["count"]).alignment = Alignment(vertical="center")
        ws.cell(row=row_idx, column=8, value=r["frame"]).alignment = Alignment(vertical="center")

        frame_range = f"Frames {item['first_frame']} - {item['last_frame']}" if item['first_frame'] != item['last_frame'] else f"Frame {item['first_frame']}"
        ws.cell(row=row_idx, column=9, value=frame_range).alignment = Alignment(vertical="center")

        ws.cell(row=row_idx, column=10, value=r["plate_width_px"]).alignment = Alignment(vertical="center")

    # Save Excel File
    xlsx_path = OUTPUT_DIR / "anpr_benchmark_with_images.xlsx"
    wb.save(str(xlsx_path))

    # Cleanup temporary image files
    for f in tmp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    # Summary
    bike_count = sum(1 for item in sorted_plates if item["best_record"].get("vehicle_type") == "Motorcycle")
    car_count = len(sorted_plates) - bike_count

    print(f"\n✅ Excel successfully saved to: {xlsx_path}")
    print(f"   Unique vehicles identified (read once): {len(all_unique_plates)}")
    print(f"     🏍️ Motorcycles / Two-Wheelers: {bike_count}")
    print(f"     🚗 Cars / Trucks / Buses: {car_count}")
    for item in sorted_plates:
        plate = item["best_record"]["ocr_text"]
        conf = item["best_record"]["confidence"]
        v_type = item["best_record"].get("vehicle_type", "Vehicle")
        icon = "🏍️" if v_type == "Motorcycle" else "🚗"
        print(f"     {icon} [{v_type}] {plate}: {item['count']} reads (Best Conf: {conf:.2f}, Frame: {item['best_record']['frame']})")


if __name__ == "__main__":
    main()
