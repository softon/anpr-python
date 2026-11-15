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
THUMB_HEIGHT = 50

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
    if (t.startswith("IND") or t.startswith("INO")) and len(t) > 7:
        t = t[3:]
    return t


def normalize_indian_plate(text):
    """
    Validates and disambiguates Indian number plate strings.
    Handles:
      - Standard: State(2) + RTO(1-2) + Series(0-3) + Number(4) (e.g., MH03EL8407, KL07CD1234)
      - Bharat Series: YY BH #### XX (e.g., 22BH1234AA)
      - OCR letter/digit position corrections (e.g. MH03DY0G22 -> MH03DY0622)
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

    # 1. Bharat (BH) Series
    m_bh = re.match(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$", t)
    if m_bh:
        return t, True

    # 2. Standard State Code Disambiguation
    s1 = DIGIT_TO_LETTER.get(t[0], t[0])
    s2 = DIGIT_TO_LETTER.get(t[1], t[1])
    state = s1 + s2
    if state == "NH":  # Frequent OCR confusion for MH (Maharashtra)
        state = "MH"
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
print("Loading GPU models (Vehicle Detector, Text Detector, Text Recognizer)...")
VEHICLE_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PP-YOLOE-S_vehicle"
DET_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PP-OCRv6_medium_det"
REC_MODEL_DIR = "/opt/traffic/models/.paddlex/official_models/PP-OCRv6_medium_rec"

v_kwargs = {"model_name": "PP-YOLOE-S_vehicle", "device": "gpu:0"}
if Path(VEHICLE_MODEL_DIR).exists():
    v_kwargs["model_dir"] = VEHICLE_MODEL_DIR
vehicle_det = paddlex.create_model(**v_kwargs)

det_kwargs = {"model_name": "PP-OCRv6_medium_det", "device": "gpu:0"}
if Path(DET_MODEL_DIR).exists():
    det_kwargs["model_dir"] = DET_MODEL_DIR
text_det = TextDetection(**det_kwargs)

rec_kwargs = {"model_name": "PP-OCRv6_medium_rec", "device": "gpu:0"}
if Path(REC_MODEL_DIR).exists():
    rec_kwargs["model_dir"] = REC_MODEL_DIR
text_rec = TextRecognition(**rec_kwargs)
print("✅ Models loaded successfully on GPU!\n")


def process_video(video_path, every_n_frames=10):
    cap = cv2.VideoCapture(str(video_path))
    results = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % every_n_frames != 0:
            continue

        h_f, w_f = frame.shape[:2]

        # Step 1: Detect vehicles in the frame
        v_res = list(vehicle_det.predict(frame, batch_size=1))[0]
        v_boxes = [b for b in v_res.json["res"]["boxes"] if b.get("score", 0) >= 0.50]

        for vb in v_boxes:
            vx1, vy1, vx2, vy2 = [int(v) for v in vb["coordinate"]]
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w_f, vx2), min(h_f, vy2)
            vw, vh = vx2 - vx1, vy2 - vy1

            # Ignore tiny vehicle detections (too small for license plate readability)
            if vw < 80 or vh < 80:
                continue

            vehicle_crop = frame[vy1:vy2, vx1:vx2]

            # Step 2: Detect text in vehicle region
            t_res = list(text_det.predict(vehicle_crop, batch_size=1))[0]
            polys = t_res.json["res"]["dt_polys"]
            if not polys:
                continue

            # Step 3: Filter text boxes to candidate number plate regions
            candidate_boxes = []
            for poly in polys:
                p_arr = np.asarray(poly)
                px1, py1 = p_arr.min(axis=0).astype(int)
                px2, py2 = p_arr.max(axis=0).astype(int)
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(vw, px2), min(vh, py2)

                bw, bh = px2 - px1, py2 - py1
                rel_y = py1 / max(1, vh)

                # Number plates are mounted on the bumper / grille (lower 60% of vehicle)
                # This rejects truck slogans, cab writing, rooftop text, and windshield stickers
                if rel_y < 0.40:
                    continue

                # Filter out huge banners spanning > 65% of vehicle width or tiny specks
                if bw > 0.65 * vw or bw < 30 or bh < 12:
                    continue

                candidate_boxes.append((px1, py1, px2, py2, bw, bh, rel_y))

            if not candidate_boxes:
                continue

            # Step 4: Handle Two-Line Stacked Plates (e.g. commercial trucks: top 'MH 05', bottom 'FJ 9670')
            candidate_boxes.sort(key=lambda b: b[1])  # sort top to bottom
            used = set()
            paired_candidates = []

            for i in range(len(candidate_boxes)):
                if i in used:
                    continue
                b1 = candidate_boxes[i]
                merged = False
                for j in range(i + 1, len(candidate_boxes)):
                    if j in used:
                        continue
                    b2 = candidate_boxes[j]

                    vert_dist = b2[1] - b1[3]  # vertical gap between lines
                    c1 = (b1[0] + b1[2]) / 2.0  # center X
                    c2 = (b2[0] + b2[2]) / 2.0
                    center_diff = abs(c1 - c2)

                    # Two lines stacked in close proximity with similar horizontal centering
                    if -10 <= vert_dist <= b1[5] * 2.0 and center_diff < max(b1[4], b2[4]) * 0.6:
                        mx1 = min(b1[0], b2[0])
                        my1 = min(b1[1], b2[1])
                        mx2 = max(b1[2], b2[2])
                        my2 = max(b1[3], b2[3])
                        paired_candidates.append(("merged", (mx1, my1, mx2, my2), [b1, b2]))
                        used.add(i)
                        used.add(j)
                        merged = True
                        break

                if not merged:
                    paired_candidates.append(("single", (b1[0], b1[1], b1[2], b1[3]), [b1]))
                    used.add(i)

            # Step 5: Recognize Plate Text & Validate
            for ctype, (cx1, cy1, cx2, cy2), src_boxes in paired_candidates:
                # Add padding for clean Excel thumbnail display
                pad_x = max(int((cx2 - cx1) * 0.06), 2)
                pad_y = max(int((cy2 - cy1) * 0.08), 2)
                pcx1 = max(0, cx1 - pad_x)
                pcy1 = max(0, cy1 - pad_y)
                pcx2 = min(vw, cx2 + pad_x)
                pcy2 = min(vh, cy2 + pad_y)
                plate_crop = vehicle_crop[pcy1:pcy2, pcx1:pcx2]

                if plate_crop.size == 0:
                    continue

                if ctype == "merged":
                    part_texts = []
                    scores = []
                    for sb in src_boxes:
                        c = vehicle_crop[sb[1]:sb[3], sb[0]:sb[2]]
                        if c.size == 0:
                            continue
                        rec_res = list(text_rec.predict(c, batch_size=1))[0]
                        part_texts.append(rec_res.json["res"]["rec_text"])
                        scores.append(float(rec_res.json["res"]["rec_score"]))
                    raw_text = "".join(part_texts)
                    conf = sum(scores) / max(len(scores), 1)
                else:
                    c = vehicle_crop[cy1:cy2, cx1:cx2]
                    rec_res = list(text_rec.predict(c, batch_size=1))[0]
                    raw_text = rec_res.json["res"]["rec_text"]
                    conf = float(rec_res.json["res"]["rec_score"])

                normalized_plate, is_valid = normalize_indian_plate(raw_text)

                # STRICT FILTERING: Only accept genuine, valid Indian vehicle registration numbers
                if is_valid and conf >= 0.50:
                    results.append({
                        "video": Path(video_path).name,
                        "frame": frame_id,
                        "crop": plate_crop,
                        "ocr_text": normalized_plate,
                        "raw_ocr": raw_text,
                        "confidence": round(conf, 3),
                        "valid_format": True,
                        "plate_width_px": pcx2 - pcx1,
                    })

    cap.release()
    return results


# Process Videos
all_results = []
video_files = list(Path("./sample_videos").glob("*.mp4"))
if not video_files:
    video_files = sorted(Path(".").glob("*.mp4"))

for video in video_files:
    print(f"🎬 Processing {video.name}...")
    all_results.extend(process_video(video))

# Build Professional Excel Report with Two Sheets
print(f"\n📊 Building Excel with {len(all_results)} detected license plate images...")
wb = Workbook()

# --- Sheet 1: All Plate Reads (frame by frame) ---
ws1 = wb.active
ws1.title = "All Plate Reads"

headers = [
    "Plate Image", "Video", "Frame", "OCR Text (Normalized)",
    "Confidence", "Valid Format", "Plate Width (px)"
]
ws1.append(headers)

# Style Header Row
header_fill = PatternFill(start_color="0D47A1", end_color="0D47A1", fill_type="solid")
header_font = Font(color="FFFFFF", bold=True, size=11)
for col_idx in range(1, len(headers) + 1):
    c = ws1.cell(row=1, column=col_idx)
    c.fill = header_fill
    c.font = header_font
    c.alignment = Alignment(horizontal="center", vertical="center")

ws1.column_dimensions["A"].width = 20
ws1.column_dimensions["B"].width = 18
ws1.column_dimensions["C"].width = 10
ws1.column_dimensions["D"].width = 25
ws1.column_dimensions["E"].width = 14
ws1.column_dimensions["F"].width = 14
ws1.column_dimensions["G"].width = 16

tmp_files = []
for row_idx, r in enumerate(all_results, start=2):
    crop = r["crop"]
    h, w = crop.shape[:2]
    thumb_w = max(int(w * THUMB_HEIGHT / max(h, 1)), 1)
    thumb = cv2.resize(crop, (thumb_w, THUMB_HEIGHT))

    tmp_path = os.path.join(tempfile.gettempdir(), f"plate_all_{row_idx}.jpg")
    cv2.imwrite(tmp_path, thumb)
    tmp_files.append(tmp_path)

    xl_img = XlImage(tmp_path)
    ws1.add_image(xl_img, f"A{row_idx}")
    ws1.row_dimensions[row_idx].height = THUMB_HEIGHT * 0.75

    ws1.cell(row=row_idx, column=2, value=r["video"]).alignment = Alignment(vertical="center")
    ws1.cell(row=row_idx, column=3, value=r["frame"]).alignment = Alignment(vertical="center")

    text_cell = ws1.cell(row=row_idx, column=4, value=r["ocr_text"])
    text_cell.font = Font(bold=True)
    text_cell.alignment = Alignment(vertical="center")

    conf_cell = ws1.cell(row=row_idx, column=5, value=r["confidence"])
    conf_cell.alignment = Alignment(vertical="center")
    if r["confidence"] >= 0.85:
        conf_cell.font = Font(color="1B5E20", bold=True)
    else:
        conf_cell.font = Font(color="0D47A1")

    valid_cell = ws1.cell(row=row_idx, column=6, value="True")
    valid_cell.font = Font(color="1B5E20", bold=True)
    valid_cell.alignment = Alignment(vertical="center")

    ws1.cell(row=row_idx, column=7, value=r["plate_width_px"]).alignment = Alignment(vertical="center")

# --- Sheet 2: Unique Vehicles (Best Highest-Confidence Read per Vehicle) ---
ws2 = wb.create_sheet(title="Unique Vehicles")
ws2.append(["Plate Image", "Plate Number", "Best Confidence", "Total Detections", "First Frame", "Last Frame"])

for col_idx in range(1, 7):
    c = ws2.cell(row=1, column=col_idx)
    c.fill = header_fill
    c.font = header_font
    c.alignment = Alignment(horizontal="center", vertical="center")

ws2.column_dimensions["A"].width = 20
ws2.column_dimensions["B"].width = 22
ws2.column_dimensions["C"].width = 16
ws2.column_dimensions["D"].width = 16
ws2.column_dimensions["E"].width = 12
ws2.column_dimensions["F"].width = 12

# Aggregate unique vehicles
unique_dict = {}
for r in all_results:
    plate = r["ocr_text"]
    if plate not in unique_dict:
        unique_dict[plate] = {
            "best_r": r,
            "count": 1,
            "first_frame": r["frame"],
            "last_frame": r["frame"]
        }
    else:
        unique_dict[plate]["count"] += 1
        unique_dict[plate]["last_frame"] = r["frame"]
        if r["confidence"] > unique_dict[plate]["best_r"]["confidence"]:
            unique_dict[plate]["best_r"] = r

for row_idx, (plate, data) in enumerate(unique_dict.items(), start=2):
    best_r = data["best_r"]
    crop = best_r["crop"]
    h, w = crop.shape[:2]
    thumb_w = max(int(w * THUMB_HEIGHT / max(h, 1)), 1)
    thumb = cv2.resize(crop, (thumb_w, THUMB_HEIGHT))

    tmp_path = os.path.join(tempfile.gettempdir(), f"plate_uniq_{row_idx}.jpg")
    cv2.imwrite(tmp_path, thumb)
    tmp_files.append(tmp_path)

    xl_img = XlImage(tmp_path)
    ws2.add_image(xl_img, f"A{row_idx}")
    ws2.row_dimensions[row_idx].height = THUMB_HEIGHT * 0.75

    p_cell = ws2.cell(row=row_idx, column=2, value=plate)
    p_cell.font = Font(bold=True, size=12)
    p_cell.alignment = Alignment(vertical="center")

    c_cell = ws2.cell(row=row_idx, column=3, value=best_r["confidence"])
    c_cell.font = Font(color="1B5E20", bold=True)
    c_cell.alignment = Alignment(vertical="center")

    ws2.cell(row=row_idx, column=4, value=data["count"]).alignment = Alignment(vertical="center")
    ws2.cell(row=row_idx, column=5, value=data["first_frame"]).alignment = Alignment(vertical="center")
    ws2.cell(row=row_idx, column=6, value=data["last_frame"]).alignment = Alignment(vertical="center")

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
print(f"\n✅ Excel successfully saved to: {xlsx_path}")
print(f"   Total valid plate reads: {len(all_results)}")
print(f"   Unique vehicles identified: {len(unique_dict)}")
for plate, data in unique_dict.items():
    print(f"     🚗 {plate}: {data['count']} reads (Best Conf: {data['best_r']['confidence']:.2f})")
