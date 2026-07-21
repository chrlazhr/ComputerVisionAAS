import os
import re
import csv
import glob
import base64
import time
import io
import sys
import requests
from PIL import Image

# =============================================================
# KONFIGURASI
# =============================================================
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen2-vl-2b-instruct" 

# Root folder dataset
DATASET_ROOT = "./Indonesian License Plate Dataset"

IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "test")
LABELS_LP_DIR = os.path.join(DATASET_ROOT, "labelswithLP", "test")

OUTPUT_CSV = "hasil_ocr_platmobil.csv"
OUTPUT_SUMMARY = "hasil_ocr_plat_summary.txt"

# Padding tambahan di sekitar bbox crop (persentase dari lebar/tinggi bbox),
# supaya karakter di tepi plat tidak terpotong.
CROP_PADDING_RATIO = 0.10

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Prompt versi lengkap (menangani kasus nyata plat Indonesia: ada tanggal
# berlaku kecil di bagian bawah plat yang sering ikut terbaca oleh VLM kalau
# tidak diberi instruksi eksplisit).
PROMPT = (
    "What is the license plate number shown in this image? Respond only with the plate number."
)

# Kalau dosen meminta persis sesuai contoh di soal, tinggal aktifkan baris ini:
# PROMPT = "What is the license plate number shown in this image? Respond only with the plate number."

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5  # akan dikali (attempt+1) -> 1.5s, 3s, 4.5s, ...


# =============================================================
# CEK KONEKSI KE LM STUDIO
# =============================================================
def check_lmstudio_connection():
    base_url = LMSTUDIO_URL.replace("/v1/chat/completions", "/v1/models")
    try:
        resp = requests.get(base_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        available_models = [m.get("id", "") for m in data.get("data", [])]
        print(f"[INFO] Berhasil terhubung ke LM Studio di {LMSTUDIO_URL}")
        if available_models:
            print(f"[INFO] Model yang tersedia di server: {available_models}")
            if MODEL_NAME not in available_models:
                print(
                    f"[WARNING] MODEL_NAME='{MODEL_NAME}' tidak ada di daftar model "
                    f"yang sedang di-load. Pastikan model sudah di-load di LM Studio."
                )
        return True
    except Exception as e:
        print(f"[ERROR] Tidak bisa terhubung ke LM Studio di {LMSTUDIO_URL}")
        print(f"[ERROR] Detail: {e}")
        print("[ERROR] Pastikan LM Studio sudah dijalankan dan server API-nya aktif (tab 'Local Server').")
        return False


# =============================================================
# PARSING LABEL (labelswithLP)
# =============================================================
def find_image_path(images_dir: str, stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
        p_upper = os.path.join(images_dir, stem + ext.upper())
        if os.path.isfile(p_upper):
            return p_upper
    return None


def parse_label_line(line: str):
    """
    Format: class_id x_center y_center width height PLATE_TEXT
    PLATE_TEXT bisa mengandung spasi, jadi ambil 5 token pertama sebagai
    angka, sisanya (token ke-6 dst, digabung) sebagai teks plat.
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    try:
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])
    except ValueError:
        return None
    plate_text = " ".join(parts[5:]).strip().upper()
    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "ground_truth": plate_text,
    }


# =============================================================
# CROP BOUNDING BOX
# =============================================================
def crop_plate(image: Image.Image, box: dict) -> Image.Image:
    img_w, img_h = image.size
    xc, yc, w, h = box["x_center"], box["y_center"], box["width"], box["height"]

    w_pad = w * (1 + CROP_PADDING_RATIO)
    h_pad = h * (1 + CROP_PADDING_RATIO)

    xmin = (xc - w_pad / 2) * img_w
    xmax = (xc + w_pad / 2) * img_w
    ymin = (yc - h_pad / 2) * img_h
    ymax = (yc + h_pad / 2) * img_h

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(img_w, int(xmax))
    ymax = min(img_h, int(ymax))

    return image.crop((xmin, ymin, xmax, ymax))


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================
# PANGGIL LM STUDIO
# =============================================================
def query_lmstudio(crop_image: Image.Image, retries: int = MAX_RETRIES) -> str:
    b64_img = image_to_base64(crop_image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return clean_prediction(text)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    print(f"[WARNING] Gagal query LM Studio setelah {retries + 1} percobaan: {last_err}")
    return ""


def clean_prediction(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return extract_main_plate_pattern(text)


PLATE_PATTERN = re.compile(r"([A-Z]{0,2})(\d{1,4})([A-Z]{1,3}\d?)")


def extract_main_plate_pattern(text: str) -> str:
    match = PLATE_PATTERN.search(text)
    if match:
        letters1, digits, suffix = match.groups()
        return f"{letters1}{digits}{suffix}"
    # kalau pola tidak cocok sama sekali, kembalikan teks asli (tanpa spasi)
    # supaya tetap bisa dihitung CER-nya (biasanya akan menghasilkan CER tinggi,
    # menandakan prediksi memang gagal total)
    return text.replace(" ", "")


# =============================================================
# CHARACTER ERROR RATE (CER)
# CER = (S + D + I) / N
# =============================================================
def levenshtein_distance(ref: str, hyp: str):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    op[0][0] = None

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                if best == sub:
                    op[i][j] = "S"
                elif best == dele:
                    op[i][j] = "D"
                else:
                    op[i][j] = "I"

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "M":
            i -= 1
            j -= 1
        elif o == "S":
            S += 1
            i -= 1
            j -= 1
        elif o == "D":
            D += 1
            i -= 1
        elif o == "I":
            I += 1
            j -= 1
        else:
            break

    return dp[n][m], S, D, I


def compute_cer(ground_truth: str, prediction: str) -> float:
    ref = ground_truth.replace(" ", "")
    hyp = prediction.replace(" ", "")
    n = len(ref)
    if n == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    _, S, D, I = levenshtein_distance(ref, hyp)
    return round((S + D + I) / n, 4)


# =============================================================
# MAIN
# =============================================================
def main():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_LP_DIR):
        raise FileNotFoundError(f"Folder label (labelswithLP) tidak ditemukan: {LABELS_LP_DIR}")

    if not check_lmstudio_connection():
        print("[ERROR] Program dihentikan karena LM Studio tidak dapat diakses.")
        sys.exit(1)

    label_files = sorted(glob.glob(os.path.join(LABELS_LP_DIR, "*.txt")))
    if not label_files:
        raise FileNotFoundError(f"Tidak ada file label di {LABELS_LP_DIR}")

    print(f"[INFO] Ditemukan {len(label_files)} file label di {LABELS_LP_DIR}")

    total_cer = 0.0
    total_plates = 0
    exact_match_count = 0

    # Buka CSV di awal dan tulis per baris (flush setiap saat) supaya kalau
    # program berhenti di tengah jalan, hasil yang sudah diproses tetap aman.
    csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(csv_file, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
    writer.writeheader()
    csv_file.flush()

    try:
        for lf_idx, label_path in enumerate(label_files, 1):
            stem = os.path.splitext(os.path.basename(label_path))[0]
            image_path = find_image_path(IMAGES_DIR, stem)

            if image_path is None:
                print(f"[WARNING] Gambar untuk label {stem} tidak ditemukan, dilewati.")
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if l.strip()]

            boxes = [parse_label_line(l) for l in lines]
            boxes = [b for b in boxes if b is not None]

            if not boxes:
                continue

            image = Image.open(image_path).convert("RGB")

            print(f"[{lf_idx}/{len(label_files)}] {stem}: {len(boxes)} plat terdeteksi")

            for plate_idx, box in enumerate(boxes):
                ground_truth = box["ground_truth"]
                crop = crop_plate(image, box)
                prediction = query_lmstudio(crop)
                cer_score = compute_cer(ground_truth, prediction)

                total_cer += cer_score
                total_plates += 1
                if prediction.replace(" ", "") == ground_truth.replace(" ", ""):
                    exact_match_count += 1

                image_label = f"{stem}_{plate_idx}"
                print(f"    [{image_label}] GT: '{ground_truth}' | Pred: '{prediction}' | CER: {cer_score}")

                writer.writerow({
                    "image": image_label,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "CER_score": cer_score,
                })
                csv_file.flush()
    finally:
        csv_file.close()

    avg_cer = round(total_cer / total_plates, 4) if total_plates else 0.0
    accuracy = round(exact_match_count / total_plates, 4) if total_plates else 0.0

    summary_lines = [
        f"Total plat diproses : {total_plates}",
        f"Rata-rata CER       : {avg_cer}",
        f"Exact match         : {exact_match_count}/{total_plates}",
        f"Accuracy (exact)    : {accuracy}",
        f"Model               : {MODEL_NAME}",
    ]

    print("\n[SELESAI]")
    for line in summary_lines:
        print(f"[HASIL] {line}")
    print(f"[HASIL] CSV disimpan di {OUTPUT_CSV}")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[HASIL] Ringkasan disimpan di {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()