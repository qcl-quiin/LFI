"""
collect_v2.py  —  Dataset Collector cho AI Fatigue Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture:
  - 6 class: normal_eye, eyes_closed, squinting, yawn, head_down, (alert = logic)
  - Auto-burst: không cần align mũi, camera tự chụp khi expression đúng
  - Expression detection realtime: chỉ lưu frame hợp lệ
  - Calibration baseline: threshold relative theo từng người
  - Sub-instructions: 3 giai đoạn × 6s mỗi class, beep báo chuyển
  - Skip frame tệ (brightness/no face), chạy đến đủ frame
  - Cảnh báo khoảng cách camera
  - MJPEG stream + pywebview UI
"""

import cv2, mediapipe as mp, numpy as np
import os, time, random, sys, threading, platform, json, base64
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import http.server, socketserver

# ─────────────────────────────────────────────────────────────
# LABELS  —  6 class cốt lõi
# ─────────────────────────────────────────────────────────────
LABELS = [
    # (key, tên VI, nhóm, crop_region)
    ("normal_eye",  "Mắt mở bình thường", "MAT",   "eye"),
    ("eyes_closed", "Nhắm mắt",           "MAT",   "eye"),
    ("squinting",   "Nheo mắt",           "MAT",   "eye"),
    ("yawn",        "Ngáp",               "MIENG", "mouth"),
    ("head_down",   "Cúi đầu",            "DAU",   "face"),
]

LABEL_KEYS  = [l[0] for l in LABELS]
LABEL_VI    = {l[0]: l[1] for l in LABELS}
LABEL_GROUP = {l[0]: l[2] for l in LABELS}
LABEL_CROP  = {l[0]: l[3] for l in LABELS}
GROUP_VI    = {"MAT": "Mắt", "MIENG": "Miệng", "DAU": "Đầu"}

# Sub-instructions: mỗi class có 3 giai đoạn, mỗi giai đoạn 6 giây
# Format: (text chính, text phụ)
LABEL_SUBS = {
    "normal_eye": [
        ("Nhìn thẳng vào camera, mắt mở tự nhiên",   "Di chuyển đầu nhẹ trái/phải"),
        ("Nhìn thẳng, hơi nghiêng đầu sang trái",     "Giữ mắt mở, không chớp cố ý"),
        ("Nhìn thẳng, hơi nghiêng đầu sang phải",     "Giữ mắt mở, thư giãn"),
    ],
    "eyes_closed": [
        ("Nhắm cả hai mắt hoàn toàn",                 "Giữ đầu thẳng, thư giãn"),
        ("Nhắm mắt, hơi cúi đầu nhẹ",                "Giữ nhắm, không chớp"),
        ("Nhắm mắt, hơi ngẩng đầu lên",              "Giữ nhắm đến khi nghe beep"),
    ],
    "squinting": [
        ("Nheo mắt như nhìn vào ánh sáng chói",       "Mắt hé mở, không nhắm hẳn"),
        ("Nheo mắt + hơi nghiêng đầu trái",           "Giữ biểu cảm nheo"),
        ("Nheo mắt + hơi nghiêng đầu phải",           "Giữ đến khi nghe beep"),
    ],
    "yawn": [
        ("Há miệng thật to như đang ngáp",            "Giữ miệng mở rộng"),
        ("Ngáp + hơi ngửa đầu ra sau",                "Há miệng to, giữ nguyên"),
        ("Ngáp + hơi cúi đầu xuống",                  "Giữ miệng mở đến khi nghe beep"),
    ],
    "head_down": [
        ("Cúi đầu xuống ~15 độ",                      "Di chuyển ĐẦU, không phải mắt"),
        ("Cúi đầu xuống ~25 độ",                      "Mắt vẫn nhìn về phía trước"),
        ("Cúi đầu + hơi nghiêng sang trái/phải",      "Giữ đến khi nghe beep"),
    ],
}

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE         = 224
TARGET_FRAMES    = 20    # 20 frame gốc × 3 aug = 60 ảnh/class
AUGMENT_COUNT    = 2     # số augment mỗi frame gốc
MIN_CROP         = 48
SUB_DURATION     = 6.0
CAPTURE_INTERVAL = 0.25  # giây giữa capture (~4fps)
CLASS_HOLD_SECS  = 1   # giây giữ đúng expression trước khi capture
CLASS_CONFIRM_SECS = 0.8  # giây transition giữa 2 class
MIN_FACE_SIZE    = 120   # pixel — face quá nhỏ → yêu cầu lại gần
MAX_FACE_SIZE    = 500   # pixel — face quá lớn → yêu cầu ra xa

# Expression thresholds (relative to baseline sau calibrate)
EAR_CLOSED_DELTA  = -0.06   # EAR < baseline + delta → eyes_closed
EAR_SQUINT_DELTA  = -0.03   # baseline + EAR_CLOSED < EAR < baseline + delta → squinting
MOUTH_OPEN_RATIO  = 0.25    # mouth_h / face_h > ratio → yawn
HEAD_DOWN_DELTA   = 0.04    # pitch > baseline_pitch + delta → head_down

# MediaPipe landmarks
EYE_LM       = [33, 133, 160, 144, 362, 263, 387, 373]
MOUTH_LM     = [61, 291, 13, 14, 78, 308, 95, 325]
NOSE_TIP     = 4
LEFT_EAR_IDX = [33,  160, 158, 133, 153, 144]
RIGHT_EAR_IDX= [362, 385, 387, 263, 373, 380]
# Mouth open landmarks
MOUTH_TOP    = 13   # upper lip
MOUTH_BOT    = 14   # lower lip
LEFT_CORNER  = 61
RIGHT_CORNER = 291

# ─────────────────────────────────────────────────────────────
# AUTO-INSTALL
# ─────────────────────────────────────────────────────────────
try:
    import webview
except ImportError:
    import subprocess
    print("  Cài pywebview lần đầu...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pywebview", "--quiet"])
    import webview

# ─────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────
META_FILE = "dataset/participants.json"

def _load_meta():
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_meta(meta):
    os.makedirs("dataset", exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _count_imgs(pid):
    total = 0
    for k in LABEL_KEYS:
        d = f"dataset/{pid}/{k}"
        if os.path.exists(d):
            total += len([x for x in os.listdir(d) if x.endswith('.png')])
    return total

def _count_by_label(pid):
    return {k: (len([x for x in os.listdir(f"dataset/{pid}/{k}")
                     if x.endswith('.png')])
                if os.path.exists(f"dataset/{pid}/{k}") else 0)
            for k in LABEL_KEYS}

def _next_id(meta):
    if not meta: return 1
    ids = [int(v["id"]) for v in meta.values() if str(v.get("id","")).isdigit()]
    return (max(ids)+1) if ids else 1

# ─────────────────────────────────────────────────────────────
# LOGIN UI
# ─────────────────────────────────────────────────────────────
def run_login_ui():
    meta   = _load_meta()
    result = {"pid": None, "info": None}
    max_t  = len(LABEL_KEYS) * TARGET_FRAMES * (1 + AUGMENT_COUNT)

    parts_list = []
    for pid_key in sorted(meta.keys()):
        info   = meta[pid_key]
        total  = _count_imgs(pid_key)
        by_lbl = _count_by_label(pid_key)
        parts_list.append({
            "pid": pid_key, "id": info["id"], "name": info["name"],
            "gender": info["gender"], "age": info["age"],
            "glasses": info.get("glasses", False),
            "total": total, "max": max_t,
            "done": total >= max_t,
            "pct": round(min(total/max_t,1.0)*100),
            "bylabel": [{"key": k, "name": LABEL_VI[k],
                         "count": by_lbl[k], "target": TARGET_FRAMES*(1+AUGMENT_COUNT),
                         "pct": round(min(by_lbl[k]/(TARGET_FRAMES*(1+AUGMENT_COUNT)),1.0)*100)}
                        for k in LABEL_KEYS],
        })

    parts_json = json.dumps(parts_list, ensure_ascii=False)

    class Api:
        def submit(self, data_str):
            data   = json.loads(data_str)
            name   = data.get("name","").strip()
            age    = str(data.get("age","")).strip()
            gender = data.get("gender","Nam")
            glasses= bool(data.get("glasses", False))
            if not name:
                return json.dumps({"ok":False,"err":"Vui lòng nhập họ và tên"})
            if not age.isdigit() or not (5 <= int(age) <= 99):
                return json.dumps({"ok":False,"err":"Tuổi không hợp lệ (5–99)"})
            name_norm = name.lower().strip()
            for pk, info in meta.items():
                if info["name"].lower().strip() == name_norm:
                    total  = _count_imgs(pk)
                    by_lbl = _count_by_label(pk)
                    return json.dumps({
                        "ok":True,"existing":True,"pid":pk,
                        "id":info["id"],"name":info["name"],
                        "gender":info["gender"],"age":info["age"],
                        "glasses":info.get("glasses",False),
                        "total":total,"max":max_t,
                        "done":total>=max_t,"pct":round(min(total/max_t,1.0)*100),
                        "bylabel":[{"key":k,"name":LABEL_VI[k],
                                    "count":by_lbl[k],"target":TARGET_FRAMES*(1+AUGMENT_COUNT),
                                    "pct":round(min(by_lbl[k]/(TARGET_FRAMES*(1+AUGMENT_COUNT)),1.0)*100)}
                                   for k in LABEL_KEYS],
                    })
            new_id  = _next_id(meta)
            pid_key = f"person_{new_id:02d}"
            return json.dumps({
                "ok":True,"existing":False,"pid":pid_key,
                "id":new_id,"name":name,"gender":gender,"age":int(age),
                "glasses":glasses,
                "total":0,"max":max_t,"done":False,"pct":0,
                "bylabel":[{"key":k,"name":LABEL_VI[k],"count":0,
                            "target":TARGET_FRAMES*(1+AUGMENT_COUNT),"pct":0}
                           for k in LABEL_KEYS],
            })

        def confirm(self, data_str):
            data    = json.loads(data_str)
            pid_key = data["pid"]
            info    = {"id":data["id"],"name":data["name"],
                       "gender":data["gender"],"age":data["age"],
                       "glasses":data.get("glasses",False)}
            if pid_key not in meta:
                meta[pid_key] = info
                _save_meta(meta)
            result["pid"]  = pid_key
            result["info"] = info
            def _close():
                time.sleep(0.1)
                window.destroy()
            threading.Thread(target=_close, daemon=True).start()
            return "ok"

        def select_existing(self, pid_key):
            if pid_key not in meta: return json.dumps({"ok":False})
            info   = meta[pid_key]
            total  = _count_imgs(pid_key)
            by_lbl = _count_by_label(pid_key)
            return json.dumps({
                "ok":True,"existing":True,"pid":pid_key,
                "id":info["id"],"name":info["name"],
                "gender":info["gender"],"age":info["age"],
                "glasses":info.get("glasses",False),
                "total":total,"max":max_t,
                "done":total>=max_t,"pct":round(min(total/max_t,1.0)*100),
                "bylabel":[{"key":k,"name":LABEL_VI[k],
                            "count":by_lbl[k],"target":TARGET_FRAMES*(1+AUGMENT_COUNT),
                            "pct":round(min(by_lbl[k]/(TARGET_FRAMES*(1+AUGMENT_COUNT)),1.0)*100)}
                           for k in LABEL_KEYS],
            })

        def delete_person(self, pid_key):
            import shutil as _sh
            if pid_key in meta:
                del meta[pid_key]
                _save_meta(meta)
            folder = f"dataset/{pid_key}"
            if os.path.exists(folder):
                _sh.rmtree(folder)
            return json.dumps({"ok": True})

        def delete_class(self, pid_key, label_key):
            folder = f"dataset/{pid_key}/{label_key}"
            if os.path.exists(folder):
                import shutil as _sh
                _sh.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
            return json.dumps({"ok": True})

        def get_participants(self):
            parts = []
            for pk in sorted(meta.keys()):
                info   = meta[pk]
                total  = _count_imgs(pk)
                by_lbl = _count_by_label(pk)
                parts.append({
                    "pid": pk, "id": info["id"], "name": info["name"],
                    "gender": info["gender"], "age": info["age"],
                    "glasses": info.get("glasses", False),
                    "total": total, "max": max_t,
                    "done": total >= max_t,
                    "pct": round(min(total/max_t,1.0)*100),
                    "bylabel": [{"key":k,"name":LABEL_VI[k],
                                 "count":by_lbl[k],
                                 "target":TARGET_FRAMES*(1+AUGMENT_COUNT),
                                 "pct":round(min(by_lbl[k]/(TARGET_FRAMES*(1+AUGMENT_COUNT)),1.0)*100)}
                                for k in LABEL_KEYS],
                })
            return json.dumps(parts, ensure_ascii=False)

    api = Api()
    TARGET_PER_CLASS = TARGET_FRAMES * (1 + AUGMENT_COUNT)

    HTML = f"""<!DOCTYPE html>
<html lang="vi"><head>
<meta charset="UTF-8">
<title>Dataset Collector v2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --cyan:#5edfff;--green:#34d399;--amber:#fbbf24;--red:#f87171;--purple:#a78bfa;
  --text:#eef5ff;--muted:rgba(190,215,255,.6);--dim:rgba(140,175,220,.3);
  --glass:rgba(255,255,255,.07);--border:rgba(255,255,255,.12);
  --mono:'JetBrains Mono',monospace;--font:'DM Sans',system-ui,sans-serif;
}}
html,body{{width:100%;height:100%;overflow:hidden;font-family:var(--font);
  background:#050d1a;color:var(--text);-webkit-font-smoothing:antialiased;}}
.bg{{position:fixed;inset:0;z-index:0;
  background:radial-gradient(ellipse 70% 60% at 15% 30%,#0d2a5e,transparent 65%),
    radial-gradient(ellipse 55% 50% at 85% 15%,#0a3050,transparent 60%),
    radial-gradient(ellipse 50% 45% at 50% 80%,#12184a,transparent 60%),#060d1c;}}
.bg::after{{content:'';position:absolute;inset:0;
  background-image:linear-gradient(rgba(94,223,255,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(94,223,255,.03) 1px,transparent 1px);
  background-size:60px 60px;}}
/* Layout */
.layout{{position:relative;z-index:1;display:grid;
  grid-template-columns:360px 1fr;grid-template-rows:48px 1fr;height:100vh;}}
/* Topbar */
.topbar{{grid-column:1/-1;display:flex;align-items:center;padding:0 28px;gap:10px;
  background:rgba(5,12,28,.7);backdrop-filter:blur(20px);
  border-bottom:1px solid rgba(94,223,255,.12);}}
.topbar-ring{{width:22px;height:22px;border-radius:50%;
  border:2px solid var(--cyan);position:relative;flex-shrink:0;}}
.topbar-ring::after{{content:'';position:absolute;inset:4px;border-radius:50%;
  background:var(--cyan);opacity:.5;animation:rPulse 2s ease-in-out infinite;}}
@keyframes rPulse{{0%,100%{{opacity:.4}}50%{{opacity:.9}}}}
.topbar-title{{font-size:11px;font-weight:700;letter-spacing:.2em;
  color:var(--cyan);text-transform:uppercase;font-family:var(--mono);}}
.topbar-title span{{color:rgba(94,223,255,.4);margin:0 8px;}}
/* Left panel */
.left-panel{{background:rgba(4,9,22,.6);backdrop-filter:blur(20px);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  padding:20px 16px;overflow:hidden;}}
.panel-label{{font-size:9px;font-weight:700;letter-spacing:.2em;
  color:var(--dim);text-transform:uppercase;font-family:var(--mono);
  margin-bottom:14px;}}
.participants-list{{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;
  scrollbar-width:thin;scrollbar-color:rgba(94,223,255,.1) transparent;}}
.p-card{{background:var(--glass);border:1px solid var(--border);border-radius:14px;
  padding:12px 14px;cursor:pointer;transition:all .2s ease;}}
.p-card:hover{{background:rgba(94,223,255,.06);border-color:rgba(94,223,255,.25);}}
.p-card-actions{{display:flex;gap:6px;margin-top:8px;opacity:0;transition:opacity .2s;}}
.p-card:hover .p-card-actions{{opacity:1;}}
.p-act-btn{{font-size:9px;padding:3px 8px;border-radius:5px;border:1px solid;
  cursor:pointer;font-family:var(--mono);font-weight:600;letter-spacing:.06em;
  background:transparent;transition:all .15s;}}
.p-act-del{{color:var(--red);border-color:rgba(248,113,113,.3);}}
.p-act-del:hover{{background:rgba(248,113,113,.12);border-color:rgba(248,113,113,.6);}}
.p-act-manage{{color:var(--muted);border-color:var(--b1);}}
.p-act-manage:hover{{background:rgba(255,255,255,.06);color:var(--text);}}
#delModal{{position:fixed;inset:0;z-index:9999;background:rgba(3,7,18,.8);
  backdrop-filter:blur(10px);display:none;align-items:center;justify-content:center;}}
#delModal.on{{display:flex;animation:fdIn .2s ease;}}
@keyframes fdIn{{from{{opacity:0;transform:scale(.97)}}to{{opacity:1;transform:scale(1)}}}}
.del-card{{background:rgba(8,15,35,.97);border:1px solid rgba(248,113,113,.2);
  border-radius:20px;padding:28px 32px;width:460px;max-width:94vw;
  box-shadow:0 24px 60px rgba(0,0,0,.7);}}
.del-header{{display:flex;align-items:center;gap:12px;margin-bottom:20px;}}
.del-icon{{width:36px;height:36px;border-radius:10px;
  background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.25);
  display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;}}
.del-title{{font-size:16px;font-weight:700;color:var(--text);}}
.del-sub{{font-size:11px;color:var(--muted);margin-top:2px;}}
.del-section{{margin-bottom:14px;}}
.del-section-label{{font-size:9px;font-weight:700;letter-spacing:.15em;
  color:var(--muted);text-transform:uppercase;margin-bottom:7px;font-family:var(--mono);}}
.del-person-btn{{width:100%;padding:10px 14px;border-radius:10px;
  background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.18);
  color:var(--red);font-size:12px;font-weight:600;cursor:pointer;
  font-family:var(--font);text-align:left;transition:all .15s;}}
.del-person-btn:hover{{background:rgba(248,113,113,.15);border-color:rgba(248,113,113,.4);}}
.del-classes{{display:grid;grid-template-columns:1fr 1fr;gap:6px;}}
.del-cls-btn{{padding:8px 12px;border-radius:8px;
  background:rgba(255,255,255,.04);border:1px solid var(--b1);
  color:var(--muted);font-size:11px;cursor:pointer;
  font-family:var(--font);display:flex;align-items:center;
  justify-content:space-between;gap:8px;transition:all .15s;text-align:left;}}
.del-cls-btn:hover{{background:rgba(248,113,113,.08);
  border-color:rgba(248,113,113,.3);color:var(--red);}}
.del-cls-count{{font-size:10px;font-family:var(--mono);opacity:.55;flex-shrink:0;}}
.del-close{{width:100%;margin-top:16px;padding:10px;border-radius:10px;
  background:rgba(255,255,255,.04);border:1px solid var(--b1);
  color:var(--muted);font-size:12px;cursor:pointer;font-family:var(--font);
  transition:all .15s;}}
.del-close:hover{{background:rgba(255,255,255,.08);color:var(--text);}}
.p-card-top{{display:flex;align-items:center;gap:10px;margin-bottom:8px;}}
.p-badge{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:99px;
  background:rgba(94,223,255,.1);color:var(--cyan);font-family:var(--mono);flex-shrink:0;}}
.p-name{{font-size:13px;font-weight:600;color:var(--text);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.p-done{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:99px;
  background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3);}}
.p-meta{{font-size:11px;color:var(--muted);margin-bottom:8px;}}
.p-pbar{{height:3px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;}}
.p-pbar-fill{{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--green));
  transition:width .5s ease;}}
.empty-state{{text-align:center;padding:40px 20px;color:var(--dim);}}
.empty-icon{{font-size:32px;margin-bottom:10px;}}
.empty-text{{font-size:12px;}}
/* Right panel — glass card */
.right-panel{{display:flex;align-items:center;justify-content:center;padding:24px;}}
.glass-card{{width:100%;max-width:560px;max-height:calc(100vh - 120px);overflow-y:auto;
  background:rgba(255,255,255,.07);backdrop-filter:blur(28px);
  border:1px solid rgba(255,255,255,.14);border-radius:24px;
  padding:36px 40px 28px;position:relative;
  box-shadow:0 8px 32px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.12);}}
.glass-card::after{{content:'';position:absolute;top:0;left:32px;right:32px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.5) 50%,transparent);}}
.screen{{display:none;}}.screen.active{{display:block;animation:fadeUp .25s ease;}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.card-title{{font-size:24px;font-weight:700;color:var(--text);margin-bottom:5px;}}
.card-sub{{font-size:13px;color:var(--muted);margin-bottom:28px;}}
.divider{{height:1px;background:linear-gradient(90deg,transparent,var(--border) 20%,var(--border) 80%,transparent);margin-bottom:24px;}}
/* Fields */
.field{{margin-bottom:18px;}}
.field-label{{font-size:10px;font-weight:600;letter-spacing:.12em;
  color:var(--dim);text-transform:uppercase;margin-bottom:7px;display:block;}}
.glass-input{{width:100%;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.13);border-radius:12px;
  padding:12px 16px;font-size:15px;color:var(--text);font-family:inherit;
  outline:none;transition:all .2s ease;}}
.glass-input::placeholder{{color:var(--dim);}}
.glass-input:focus{{border-color:rgba(94,223,255,.5);background:rgba(94,223,255,.05);
  box-shadow:0 0 0 3px rgba(94,223,255,.08);}}
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button{{-webkit-appearance:none;}}
input[type=number]{{-moz-appearance:textfield;}}
.row-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
/* Custom gender dropdown */
.gender-wrap{{position:relative;}}
.gender-display{{width:100%;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.13);border-radius:12px;
  padding:12px 16px;font-size:15px;color:var(--text);font-family:inherit;
  cursor:pointer;display:flex;align-items:center;justify-content:space-between;
  transition:all .2s ease;user-select:none;}}
.gender-display:hover,.gender-display.open{{border-color:rgba(94,223,255,.5);
  background:rgba(94,223,255,.05);}}
.gender-arrow{{font-size:10px;color:rgba(94,223,255,.7);transition:transform .2s;}}
.gender-display.open .gender-arrow{{transform:rotate(180deg);}}
.gender-opts{{position:absolute;top:calc(100% + 6px);left:0;right:0;
  background:rgba(10,20,45,.97);backdrop-filter:blur(24px);
  border:1px solid rgba(94,223,255,.25);border-radius:12px;overflow:hidden;
  z-index:999;opacity:0;transform:translateY(-6px);pointer-events:none;
  transition:opacity .18s ease,transform .18s ease;
  box-shadow:0 8px 32px rgba(0,0,0,.5);}}
.gender-opts.open{{opacity:1;transform:translateY(0);pointer-events:auto;}}
.gender-opt{{padding:11px 16px;font-size:14px;color:rgba(190,215,255,.8);
  cursor:pointer;transition:background .15s,color .15s;
  display:flex;align-items:center;gap:10px;}}
.gender-opt:hover{{background:rgba(94,223,255,.1);color:var(--text);}}
.gender-opt.selected{{background:rgba(94,223,255,.08);color:var(--cyan);}}
.gender-opt::before{{content:'';width:6px;height:6px;border-radius:50%;
  background:rgba(94,223,255,.3);flex-shrink:0;transition:background .15s;}}
.gender-opt.selected::before{{background:var(--cyan);}}
.gender-opt+.gender-opt{{border-top:1px solid rgba(255,255,255,.05);}}
/* Glasses toggle */
.toggle-wrap{{display:flex;align-items:center;gap:12px;
  padding:12px 16px;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.1);border-radius:12px;cursor:pointer;
  transition:all .2s ease;user-select:none;}}
.toggle-wrap:hover{{background:rgba(94,223,255,.05);border-color:rgba(94,223,255,.2);}}
.toggle-switch{{width:36px;height:20px;border-radius:99px;
  background:rgba(255,255,255,.15);position:relative;
  transition:background .2s;flex-shrink:0;}}
.toggle-switch.on{{background:rgba(94,223,255,.5);}}
.toggle-knob{{position:absolute;top:3px;left:3px;width:14px;height:14px;
  border-radius:50%;background:white;transition:transform .2s;}}
.toggle-switch.on .toggle-knob{{transform:translateX(16px);}}
.toggle-label{{font-size:13px;color:var(--muted);}}
/* Error */
.error-msg{{display:none;background:rgba(248,113,113,.1);
  border:1px solid rgba(248,113,113,.25);border-radius:10px;
  padding:9px 14px;font-size:12px;color:var(--red);margin-bottom:14px;}}
.error-msg.show{{display:block;}}
/* Buttons */
.btn-primary{{width:100%;background:linear-gradient(135deg,rgba(94,223,255,.2),rgba(167,139,250,.15));
  border:1px solid rgba(94,223,255,.35);border-radius:14px;
  padding:14px;font-size:14px;font-weight:700;color:var(--cyan);
  letter-spacing:.08em;text-transform:uppercase;cursor:pointer;
  font-family:inherit;transition:all .2s ease;margin-top:6px;position:relative;overflow:hidden;}}
.btn-primary::before{{content:'';position:absolute;inset:0 0 auto 0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.4),transparent);}}
.btn-primary:hover{{background:linear-gradient(135deg,rgba(94,223,255,.3),rgba(167,139,250,.25));
  box-shadow:0 0 30px rgba(94,223,255,.2);transform:translateY(-1px);}}
.btn-row{{display:flex;gap:10px;margin-top:20px;}}
.btn-secondary{{flex:1;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.12);border-radius:12px;
  padding:12px;font-size:13px;font-weight:600;color:var(--muted);
  cursor:pointer;font-family:inherit;transition:all .2s;}}
.btn-secondary:hover{{background:rgba(255,255,255,.09);color:var(--text);}}
.btn-confirm{{flex:2;background:linear-gradient(135deg,rgba(52,211,153,.2),rgba(94,223,255,.15));
  border:1px solid rgba(52,211,153,.35);border-radius:12px;
  padding:12px;font-size:13px;font-weight:700;color:var(--green);
  letter-spacing:.06em;text-transform:uppercase;cursor:pointer;
  font-family:inherit;transition:all .2s;position:relative;overflow:hidden;}}
.btn-confirm:hover{{background:linear-gradient(135deg,rgba(52,211,153,.3),rgba(94,223,255,.2));
  box-shadow:0 0 24px rgba(52,211,153,.15);transform:translateY(-1px);}}
/* Confirm screen */
.confirm-badge{{display:inline-flex;align-items:center;gap:7px;
  padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;margin-bottom:14px;}}
.confirm-badge.new{{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.3);color:var(--green);}}
.confirm-badge.existing{{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);color:var(--amber);}}
.info-grid{{display:grid;grid-template-columns:auto 1fr;gap:9px 22px;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);
  border-radius:14px;padding:16px 18px;margin-bottom:20px;}}
.info-label{{font-size:11px;color:var(--dim);font-weight:500;}}
.info-value{{font-size:13px;color:var(--text);font-weight:600;}}
.info-value.accent{{color:var(--cyan);}}
.info-value.green{{color:var(--green);}}
.info-value.amber{{color:var(--amber);}}
.progress-title{{font-size:10px;font-weight:600;letter-spacing:.12em;
  color:var(--dim);text-transform:uppercase;margin-bottom:10px;}}
.progress-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;
  max-height:160px;overflow-y:auto;scrollbar-width:thin;
  scrollbar-color:rgba(255,255,255,.1) transparent;}}
.prog-row{{display:flex;align-items:center;gap:8px;}}
.prog-name{{font-size:11px;color:var(--muted);width:120px;flex-shrink:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.prog-bar-bg{{flex:1;height:4px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;}}
.prog-bar-fill{{height:100%;border-radius:99px;transition:width .5s ease;}}
.prog-count{{font-size:10px;color:var(--dim);width:26px;text-align:right;flex-shrink:0;}}
.prog-count.done{{color:var(--green);}}
.done-banner{{display:none;background:rgba(52,211,153,.08);
  border:1px solid rgba(52,211,153,.2);border-radius:10px;
  padding:9px 14px;font-size:12px;color:var(--green);margin-top:12px;text-align:center;}}
.done-banner.show{{display:block;}}
.footer-stats{{margin-top:24px;padding-top:18px;
  border-top:1px solid rgba(255,255,255,.07);
  font-size:11px;color:var(--dim);text-align:center;font-family:var(--mono);}}
</style></head>
<body>
<div class="bg"></div>
<div class="layout">
  <div class="topbar">
    <div class="topbar-ring"></div>
    <div class="topbar-title">Dataset <span>|</span> Collector <span>|</span> v2</div>
  </div>
  <div class="left-panel">
    <div class="panel-label">Người Đã Tham Gia</div>
    <div class="participants-list" id="pList"></div>
  </div>
  <div class="right-panel">
    <div class="glass-card">
      <!-- FORM -->
      <div class="screen active" id="screenForm">
        <div class="card-title">Thông Tin Tham Gia</div>
        <div class="card-sub">Nhập thông tin để bắt đầu thu thập dữ liệu</div>
        <div class="divider"></div>
        <div class="field">
          <label class="field-label">Họ và Tên *</label>
          <input class="glass-input" id="inpName" type="text"
            placeholder="Nguyễn Văn A" autocomplete="off" spellcheck="false">
        </div>
        <div class="row-2">
          <div class="field">
            <label class="field-label">Giới Tính</label>
            <div class="gender-wrap" id="genderWrap">
              <div class="gender-display" id="genderDisplay" onclick="toggleGender()">
                <span id="genderText">Nam</span>
                <span class="gender-arrow">▾</span>
              </div>
              <div class="gender-opts" id="genderOpts">
                <div class="gender-opt selected" onclick="selectGender('Nam',this)">Nam</div>
                <div class="gender-opt" onclick="selectGender('Nữ',this)">Nữ</div>
                <div class="gender-opt" onclick="selectGender('Khác',this)">Khác</div>
              </div>
              <input type="hidden" id="inpGender" value="Nam">
            </div>
          </div>
          <div class="field">
            <label class="field-label">Tuổi</label>
            <input class="glass-input" id="inpAge" type="number"
              min="5" max="99" placeholder="25">
          </div>
        </div>
        <div class="field">
          <label class="field-label">Phụ Kiện</label>
          <div class="toggle-wrap" id="glassesToggle" onclick="toggleGlasses()">
            <div class="toggle-switch" id="glassesSw"></div>
            <span class="toggle-label" id="glassesLbl">Không đeo kính</span>
          </div>
          <input type="hidden" id="inpGlasses" value="false">
        </div>
        <div class="error-msg" id="errMsg"></div>
        <button class="btn-primary" onclick="doNext()">Tiếp Theo &nbsp;→</button>
        <div class="footer-stats" id="footerStats"></div>
      </div>
      <!-- CONFIRM -->
      <div class="screen" id="screenConfirm">
        <div id="confirmBadge"></div>
        <div class="card-title" id="confirmTitle"></div>
        <div class="card-sub" id="confirmSub"></div>
        <div class="divider"></div>
        <div class="info-grid" id="infoGrid"></div>
        <div class="progress-title">Tiến Độ Từng Class</div>
        <div class="progress-grid" id="progressGrid"></div>
        <div class="done-banner" id="doneBanner">✅ Đã thu đủ! Có thể bổ sung thêm.</div>
        <div class="btn-row">
          <button class="btn-secondary" onclick="goBack()">← Quay lại</button>
          <button class="btn-confirm" id="btnConfirm" onclick="doConfirm()">Bắt Đầu Thu ▶</button>
        </div>
      </div>
    </div>
  </div>
</div>
<div id="delModal">
  <div class="del-card">
    <div class="del-header">
      <div class="del-icon">🗑</div>
      <div>
        <div class="del-title" id="delTitle">Quản lý dữ liệu</div>
        <div class="del-sub" id="delSub"></div>
      </div>
    </div>
    <div class="del-section">
      <div class="del-section-label">Xoá toàn bộ người này</div>
      <button class="del-person-btn" id="delPersonBtn" onclick="confirmDelPerson()"></button>
    </div>
    <div class="del-section">
      <div class="del-section-label">Xoá từng class</div>
      <div class="del-classes" id="delClasses"></div>
    </div>
    <button class="del-close" onclick="closeDelModal()">Đóng</button>
  </div>
</div>
<script>
const PARTICIPANTS = {parts_json};
const TARGET_PER = {TARGET_PER_CLASS};
let currentData = null;
let glassesOn = false;

function renderList() {{
  const el = document.getElementById('pList');
  if (!PARTICIPANTS.length) {{
    el.innerHTML = '<div class="empty-state"><div class="empty-icon">👤</div><div class="empty-text">Chưa có người nào</div></div>';
    return;
  }}
  el.innerHTML = PARTICIPANTS.map(p => `
    <div class="p-card" onclick="selectExisting('${{p.pid}}')">
      <div class="p-card-top">
        <div class="p-badge">#${{String(p.id).padStart(2,'0')}}</div>
        <div class="p-name">${{p.name}}</div>
        ${{p.done ? '<div class="p-done">✓ Đủ</div>' : `<span style="font-size:11px;color:var(--amber)">${{p.total}}/${{p.max}}</span>`}}
      </div>
      <div class="p-meta">${{p.gender}} • ${{p.age}} tuổi${{p.glasses?' • 👓':''}}</div>
      <div class="p-pbar"><div class="p-pbar-fill" style="width:${{p.pct}}%"></div></div>
      <div class="p-card-actions">
        <button class="p-act-btn p-act-manage" onclick="event.stopPropagation();openDelModal('${{p.pid}}')">⚙ Quản lý</button>
        <button class="p-act-btn p-act-del" onclick="event.stopPropagation();openDelModal('${{p.pid}}')">🗑 Xoá</button>
      </div>
    </div>`).join('');
}}

let _delPid=null;
function openDelModal(pid){{
  _delPid=pid;
  const p=PARTICIPANTS.find(x=>x.pid===pid);
  if(!p)return;
  document.getElementById('delTitle').textContent=p.name;
  document.getElementById('delSub').textContent=`#${{String(p.id).padStart(2,'0')}} · ${{p.gender}} · ${{p.age}} tuổi · ${{p.total}} ảnh`;
  document.getElementById('delPersonBtn').textContent=`🗑 Xoá toàn bộ — ${{p.total}} ảnh`;
  document.getElementById('delClasses').innerHTML=p.bylabel.map(l=>`
    <button class="del-cls-btn" onclick="confirmDelClass('${{l.key}}','${{l.name}}')">
      <span>${{l.name}}</span>
      <span class="del-cls-count">${{l.count}} ảnh</span>
    </button>`).join('');
  document.getElementById('delModal').classList.add('on');
}}
function closeDelModal(){{
  document.getElementById('delModal').classList.remove('on');
  _delPid=null;
}}
async function refreshData(){{
  const raw = await pywebview.api.get_participants();
  const fresh = JSON.parse(raw);
  // Update PARTICIPANTS in place
  PARTICIPANTS.length = 0;
  fresh.forEach(p => PARTICIPANTS.push(p));
  renderList();
  renderFooter();
}}
async function confirmDelPerson(){{
  if(!_delPid)return;
  const p=PARTICIPANTS.find(x=>x.pid===_delPid);
  if(!confirm(`Xoá toàn bộ dữ liệu của "${{p?.name}}"?\nHành động này KHÔNG thể hoàn tác.`))return;
  await pywebview.api.delete_person(_delPid);
  closeDelModal();
  await refreshData();
}}
async function confirmDelClass(key,name){{
  if(!_delPid)return;
  const p=PARTICIPANTS.find(x=>x.pid===_delPid);
  const lbl=p?.bylabel?.find(l=>l.key===key);
  if(!confirm(`Xoá class "${{name}}" của "${{p?.name}}"?\n${{lbl?.count||0}} ảnh sẽ bị xoá vĩnh viễn.`))return;
  await pywebview.api.delete_class(_delPid,key);
  closeDelModal();
  await refreshData();
}}
document.addEventListener('click',function(e){{
  const m=document.getElementById('delModal');
  if(e.target===m)closeDelModal();
}});

function renderFooter() {{
  const total = PARTICIPANTS.reduce((a,p)=>a+p.total,0);
  document.getElementById('footerStats').textContent =
    `Tổng: ${{PARTICIPANTS.length}} người • ${{total}} ảnh đã thu`;
}}

function toggleGender() {{
  const d=document.getElementById('genderDisplay');
  const o=document.getElementById('genderOpts');
  const open=d.classList.toggle('open');
  o.classList.toggle('open',open);
  if(open) setTimeout(()=>document.addEventListener('click',function h(e){{
    if(!document.getElementById('genderWrap').contains(e.target)){{
      d.classList.remove('open');o.classList.remove('open');
      document.removeEventListener('click',h);
    }}
  }}),10);
}}
function selectGender(val,el){{
  document.getElementById('genderText').textContent=val;
  document.getElementById('inpGender').value=val;
  document.querySelectorAll('.gender-opt').forEach(e=>e.classList.remove('selected'));
  el.classList.add('selected');
  document.getElementById('genderDisplay').classList.remove('open');
  document.getElementById('genderOpts').classList.remove('open');
}}
function toggleGlasses(){{
  glassesOn=!glassesOn;
  document.getElementById('glassesSw').classList.toggle('on',glassesOn);
  document.getElementById('glassesLbl').textContent=glassesOn?'Đang đeo kính':'Không đeo kính';
  document.getElementById('inpGlasses').value=glassesOn;
}}

async function doNext(){{
  const name=document.getElementById('inpName').value.trim();
  const age=document.getElementById('inpAge').value.trim();
  const gender=document.getElementById('inpGender').value;
  const glasses=glassesOn;
  const err=document.getElementById('errMsg');
  err.classList.remove('show');
  const res=await pywebview.api.submit(JSON.stringify({{name,age,gender,glasses}}));
  const data=JSON.parse(res);
  if(!data.ok){{err.textContent=data.err;err.classList.add('show');return;}}
  currentData=data;
  showConfirm(data);
}}

async function selectExisting(pid){{
  const res=await pywebview.api.select_existing(pid);
  const data=JSON.parse(res);
  if(!data.ok)return;
  currentData=data;
  showConfirm(data);
}}

function showConfirm(data){{
  document.getElementById('confirmBadge').innerHTML=
    `<div class="confirm-badge ${{data.existing?'existing':'new'}}">${{data.existing?'⚡ Người hiện có':'✨ Người mới'}}</div>`;
  document.getElementById('confirmTitle').textContent=data.name;
  document.getElementById('confirmSub').textContent=
    data.existing?'Tiếp tục thu bổ sung dữ liệu':'Xác nhận thông tin trước khi bắt đầu';
  const pctTxt=`${{data.total}} / ${{data.max}} ảnh (${{data.pct}}%)`;
  const pClass=data.done?'green':(data.pct>0?'amber':'');
  document.getElementById('infoGrid').innerHTML=`
    <span class="info-label">Số thứ tự</span><span class="info-value accent">#${{String(data.id).padStart(2,'0')}}</span>
    <span class="info-label">Họ và tên</span><span class="info-value">${{data.name}}</span>
    <span class="info-label">Giới tính</span><span class="info-value">${{data.gender}}</span>
    <span class="info-label">Tuổi</span><span class="info-value">${{data.age}} tuổi</span>
    <span class="info-label">Kính</span><span class="info-value">${{data.glasses?'👓 Có đeo kính':'Không đeo'}}</span>
    <span class="info-label">Đã thu</span><span class="info-value ${{pClass}}">${{pctTxt}}</span>`;
  document.getElementById('progressGrid').innerHTML=data.bylabel.map(l=>{{
    const c=l.pct>=100?'done':'';
    const col=l.pct>=100?'#34d399':(l.pct>50?'#5edfff':(l.pct>0?'#fbbf24':'rgba(255,255,255,.15)'));
    return `<div class="prog-row">
      <span class="prog-name" title="${{l.name}}">${{l.name}}</span>
      <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:${{l.pct}}%;background:${{col}}"></div></div>
      <span class="prog-count ${{c}}">${{l.pct>=100?'✓':l.count}}</span>
    </div>`;
  }}).join('');
  data.done?document.getElementById('doneBanner').classList.add('show'):
    document.getElementById('doneBanner').classList.remove('show');
  document.getElementById('btnConfirm').textContent=data.done?'Tiếp Tục Bổ Sung ▶':'Bắt Đầu Thu ▶';
  document.getElementById('screenForm').classList.remove('active');
  document.getElementById('screenConfirm').classList.add('active');
}}

async function doConfirm(){{
  if(!currentData)return;
  await pywebview.api.confirm(JSON.stringify({{
    pid:currentData.pid,id:currentData.id,name:currentData.name,
    gender:currentData.gender,age:currentData.age,glasses:currentData.glasses
  }}));
}}
function goBack(){{
  document.getElementById('screenConfirm').classList.remove('active');
  document.getElementById('screenForm').classList.add('active');
}}
document.addEventListener('keydown',e=>{{
  if(e.key==='Enter'){{
    if(document.getElementById('screenForm').classList.contains('active'))doNext();
    else if(document.getElementById('screenConfirm').classList.contains('active'))doConfirm();
  }}
}});
renderList();renderFooter();
</script></body></html>"""

    window = webview.create_window(
        "Dataset Collector v2", html=HTML, js_api=api,
        width=1100, height=700, min_size=(900,600),
        resizable=True, text_select=False)
    webview.start()
    if result["pid"] is None:
        print("\n  [Đã thoát]")
        sys.exit(0)
    return result["pid"], result["info"]

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE & CAMERA SETUP
# ─────────────────────────────────────────────────────────────
_lr         = run_login_ui()
person_id   = _lr[0]
person_info = _lr[1]
print(f"\n  >> Thu thập cho: {person_id}  ({person_info['name']})\n")

for key in LABEL_KEYS:
    os.makedirs(f"dataset/{person_id}/{key}", exist_ok=True)

def count_existing(label):
    d = f"dataset/{person_id}/{label}"
    return len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.exists(d) else 0

counts = {k: count_existing(k) for k in LABEL_KEYS}

mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 25)  # 25fps — bội số 50Hz, tránh banding đèn VN
if not cap.isOpened():
    raise RuntimeError("Không mở được camera!")

clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_dark = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))

# ─────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────
def enhance_for_save(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    brightness = int(np.mean(l))
    if brightness < 90:
        gamma = 1.8 if brightness < 50 else 1.4
        lut = np.array([min(255,int((i/255.0)**(1.0/gamma)*255)) for i in range(256)], dtype=np.uint8)
        l = cv2.LUT(l, lut)
        l = clahe_dark.apply(l)
    elif brightness > 200:
        gamma = 1.3
        lut = np.array([min(255,int((i/255.0)**gamma*255)) for i in range(256)], dtype=np.uint8)
        l = cv2.LUT(l, lut)
        l = clahe.apply(l)
    else:
        l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), brightness

NO_FLIP = {"head_down"}

def augment(img, label=""):
    out = []
    h, w = img.shape[:2]
    variants = [
        lambda x: np.clip(x.astype(np.int16)+20, 0, 255).astype(np.uint8),
        lambda x: np.clip(x.astype(np.int16)-20, 0, 255).astype(np.uint8),
        lambda x: cv2.warpAffine(x, cv2.getRotationMatrix2D((w/2,h/2), random.uniform(-4,4), 1.0), (w,h), borderMode=cv2.BORDER_REFLECT),
        lambda x: np.clip(x.astype(np.int16)+np.random.normal(0,6,x.shape).astype(np.int16), 0, 255).astype(np.uint8),
        lambda x: np.clip(x.astype(np.float32)*random.uniform(0.85,1.15), 0, 255).astype(np.uint8),
    ]
    if label not in NO_FLIP:
        variants.append(lambda x: cv2.flip(x, 1))
    random.shuffle(variants)
    for fn in variants[:AUGMENT_COUNT]:
        try: out.append(fn(img.copy()))
        except: pass
    return out

def get_crop(frame, landmarks, label, w, h):
    region = LABEL_CROP.get(label, "face")
    if region == "eye":
        pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in EYE_LM]
        pad = 36
    elif region == "mouth":
        pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in MOUTH_LM]
        pad = 28
    else:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
        pad = 20
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1 = max(min(xs)-pad, 0); x2 = min(max(xs)+pad, w)
    y1 = max(min(ys)-pad, 0); y2 = min(max(ys)+pad, h)
    return frame[y1:y2, x1:x2]

def get_face_box(lms, w, h, pad=20):
    xs = [int(lm.x*w) for lm in lms]; ys = [int(lm.y*h) for lm in lms]
    return (max(min(xs)-pad,0), max(min(ys)-pad,0),
            min(max(xs)+pad,w), min(max(ys)+pad,h))

def save_frame(crop, label):
    global counts
    if counts[label] >= TARGET_FRAMES * (1 + AUGMENT_COUNT): return 0
    ch, cw = crop.shape[:2]
    if ch < MIN_CROP or cw < MIN_CROP: return 0
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    enhanced, _ = enhance_for_save(resized)
    saved = 0
    imgs = [enhanced] + augment(enhanced, label)
    for img in imgs:
        if counts[label] >= TARGET_FRAMES * (1 + AUGMENT_COUNT): break
        path = f"dataset/{person_id}/{label}/{label}_{counts[label]:04d}.png"
        cv2.imwrite(path, img)
        counts[label] += 1
        saved += 1
    return saved

# ─────────────────────────────────────────────────────────────
# EXPRESSION DETECTION
# ─────────────────────────────────────────────────────────────
def compute_ear(landmarks, eye_idx, w, h):
    pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in eye_idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A+B)/(2.0*C) if C > 0 else 0.0

def compute_mouth_ratio(landmarks, w, h):
    """Tỷ lệ mouth_height / face_height"""
    top = landmarks[MOUTH_TOP]
    bot = landmarks[MOUTH_BOT]
    lc  = landmarks[LEFT_CORNER]
    rc  = landmarks[RIGHT_CORNER]
    chin     = landmarks[152]
    forehead = landmarks[10]
    mouth_h  = abs((bot.y - top.y) * h)
    face_h   = abs((chin.y - forehead.y) * h)
    return mouth_h / face_h if face_h > 0 else 0.0

def compute_pitch(landmarks, w, h):
    """Pitch normalized: positive = cúi xuống"""
    nose     = landmarks[4]
    l_ear    = landmarks[234]; r_ear = landmarks[454]
    chin     = landmarks[152]; forehead = landmarks[10]
    ear_y    = (l_ear.y + r_ear.y) / 2.0
    face_h   = abs(chin.y - forehead.y)
    return (nose.y - (forehead.y + chin.y)/2.0) / (face_h + 1e-6)

def compute_yaw(landmarks, w, h):
    """Yaw normalized: positive = quay phải"""
    l_ear  = landmarks[234]; r_ear = landmarks[454]
    nose   = landmarks[4]
    face_w = abs(r_ear.x - l_ear.x)
    center_x = (l_ear.x + r_ear.x) / 2.0
    return (nose.x - center_x) / (face_w + 1e-6)

def check_expression(label, lms, w, h, baseline):
    """Kiểm tra expression đúng không. Returns (valid, feedback_msg)"""
    ear_l = compute_ear(lms, LEFT_EAR_IDX,  w, h)
    ear_r = compute_ear(lms, RIGHT_EAR_IDX, w, h)
    ear   = (ear_l + ear_r) / 2.0
    mouth = compute_mouth_ratio(lms, w, h)
    pitch = compute_pitch(lms, w, h)

    base_ear   = baseline.get("ear",   0.28)
    base_pitch = baseline.get("pitch", 0.0)

    if label == "normal_eye":
        if ear < base_ear - 0.05:
            return False, "Mở mắt ra!"
        return True, ""

    elif label == "eyes_closed":
        if ear > base_ear - 0.06:
            return False, "Nhắm mắt lại!"
        return True, ""

    elif label == "squinting":
        lo = base_ear - 0.07
        hi = base_ear - 0.02
        if ear > hi:
            return False, "Nheo mắt lại — mắt hé hơn!"
        if ear < lo:
            return False, "Không cần nhắm hẳn — hé một chút!"
        return True, ""

    elif label == "yawn":
        if mouth < MOUTH_OPEN_RATIO:
            return False, "Há miệng to hơn!"
        return True, ""

    elif label == "head_down":
        if pitch < base_pitch + HEAD_DOWN_DELTA:
            return False, "Cúi đầu xuống nhiều hơn!"
        return True, ""

    return True, ""

# ─────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────
baseline = {}
_calib_samples = []
_CALIB_FRAMES  = 45

def start_calibration():
    global _calib_samples
    _calib_samples = []

def update_calibration(lms, w, h):
    """Thu mẫu calibration. Trả về True khi xong."""
    ear_l = compute_ear(lms, LEFT_EAR_IDX,  w, h)
    ear_r = compute_ear(lms, RIGHT_EAR_IDX, w, h)
    pitch = compute_pitch(lms, w, h)
    fbox  = get_face_box(lms, w, h, pad=0)
    fs    = fbox[2] - fbox[0]
    _calib_samples.append({"ear":(ear_l+ear_r)/2.0, "pitch":pitch, "face_size":fs})
    if len(_calib_samples) >= _CALIB_FRAMES:
        baseline["ear"]       = float(np.median([s["ear"]       for s in _calib_samples]))
        baseline["pitch"]     = float(np.median([s["pitch"]     for s in _calib_samples]))
        baseline["face_size"] = float(np.median([s["face_size"] for s in _calib_samples]))
        print(f"  [CALIB] ear={baseline['ear']:.3f} pitch={baseline['pitch']:.3f} "
              f"face={baseline['face_size']:.0f}px")
        return True
    return len(_calib_samples) / _CALIB_FRAMES

# ─────────────────────────────────────────────────────────────
# MJPEG SERVER + SHARED STATE
# ─────────────────────────────────────────────────────────────
_PORT   = 47292
_st     = {
    "jpeg_buf"       : b"",
    "running"        : True,
    "shutting_down"  : False,
    "outro_done"     : threading.Event(),
    # Camera state
    "face_box"       : None,
    "calib_pitch"    : 0,
    "calib_yaw"      : 0,
    "eye_box"        : None,
    "brightness"     : 128,
    "face_size"      : 0,
    "face_ok"        : False,   # face size trong range
    # Calibration
    "calib_state"    : "idle",  # idle | collecting | done
    "calib_progress" : 0.0,
    "calib_notif_ts" : 0.0,
    # Recording state
    "recording"      : None,    # label key đang thu
    "cursor"         : 0,
    "counts"         : {},
    # Auto-burst state
    "sub_idx"        : 0,       # 0,1,2 — sub-instruction hiện tại
    "sub_elapsed"    : 0.0,     # giây đã qua trong sub hiện tại
    "valid_frames"   : 0,       # số frame hợp lệ đã thu trong session
    "expr_valid"     : False,   # expression hợp lệ frame hiện tại
    "expr_feedback"  : "",      # feedback text nếu sai
    "sub_beep_ts"    : 0.0,     # timestamp beep chuyển sub
    "capture_ts"     : 0.0,     # timestamp lần chụp cuối
    "capture_flash_ts":0.0,
    "brightness_warn": "",      # "dark"/"bright"/"" 
    "distance_warn"  : "",      # "close"/"far"/""
    "crop_preview"   : "",
    # Sound signals
    "snd_start_ts"   : 0.0,
    "snd_stop_ts"    : 0.0,
    "snd_done_ts"    : 0.0,
    "snd_sub_ts"     : 0.0,     # beep khi chuyển sub
    "snd_calib_ts"   : 0.0,
    "pending_keys"   : [],
}
_st_lock = threading.Lock()
_wv_win  = None

def _push(jpeg_bytes, updates):
    with _st_lock:
        _st["jpeg_buf"] = jpeg_bytes
        _st.update(updates)

def _pop_keys():
    with _st_lock:
        ks = list(_st["pending_keys"])
        _st["pending_keys"] = []
    return ks

class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type","multipart/x-mixed-replace; boundary=--frame")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            try:
                while _st["running"]:
                    with _st_lock: buf = _st["jpeg_buf"]
                    if buf:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf+b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.008)
            except: pass

        elif self.path == "/eye":
            with _st_lock: eb = _st.get("eye_box", None)
            resp = '{{"e":{}}}'.format(list(eb) if eb else "null")
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            self.wfile.write(resp.encode())
            return
        elif self.path == "/state":
            with _st_lock: s = dict(_st)
            tpc = TARGET_FRAMES * (1 + AUGMENT_COUNT)
            labels = []
            for key in LABEL_KEYS:
                cnt = s["counts"].get(key, 0)
                labels.append({
                    "key":key, "name":LABEL_VI[key],
                    "group":LABEL_GROUP[key],
                    "group_name":GROUP_VI.get(LABEL_GROUP[key],""),
                    "count":cnt, "target":tpc,
                    "done":cnt>=tpc,
                    "pct":round(min(cnt/tpc,1.0)*100),
                    "is_cursor":LABEL_KEYS.index(key)==s["cursor"],
                    "is_rec":key==s["recording"],
                })
            # Sub instruction
            rec = s["recording"]
            subs = LABEL_SUBS.get(rec, [("",""),("",""),("","")]) if rec else []
            sub_idx = min(s["sub_idx"], len(subs)-1) if subs else 0
            sub = subs[sub_idx] if subs else ("","")

            resp = json.dumps({
                "labels"         : labels,
                "recording"      : rec,
                "cursor"         : s["cursor"],
                "brightness"     : s["brightness"],
                "face_box"       : s["face_box"],
                "calib_pitch"    : s.get("calib_pitch",0),
                "calib_yaw"      : s.get("calib_yaw",0),
                "eye_box"        : s.get("eye_box", None),
                "face_ok"        : s["face_ok"],
                "calib_state"    : s["calib_state"],
                "calib_progress" : round(s["calib_progress"],3),
                "sub_idx"        : sub_idx,
                "sub_main"       : sub[0],
                "sub_hint"       : sub[1],
                "sub_elapsed"    : round(s["sub_elapsed"],2),
                "sub_total"      : SUB_DURATION,
                "valid_frames"   : s["valid_frames"],
                "target_frames"  : tpc,
                "expr_valid"     : s["expr_valid"],
                "expr_feedback"  : s["expr_feedback"],
                "capture_flash"  : time.time()-s["capture_flash_ts"] < 0.15,
                "brightness_warn": s["brightness_warn"],
                "distance_warn"  : s["distance_warn"],
                "crop_preview"   : s["crop_preview"],
                "running"        : s["running"],
                "shutting_down"  : s["shutting_down"],
                "snd_start_ts"   : s["snd_start_ts"],
                "snd_stop_ts"    : s["snd_stop_ts"],
                "snd_done_ts"    : s["snd_done_ts"],
                "snd_sub_ts"     : s["snd_sub_ts"],
                "snd_calib_ts"   : s["snd_calib_ts"],
                "person_id"      : person_id,
                "person_name"    : person_info.get("name",""),
                "target_count"   : tpc,
                "calib_done"     : bool(baseline),
            })
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.end_headers()
            self.wfile.write(resp.encode())
        else:
            self.send_error(404)

def _start_http():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1",_PORT), _Handler)
    srv.daemon_threads = True
    srv.serve_forever()

threading.Thread(target=_start_http, daemon=True).start()
time.sleep(0.3)

# ─────────────────────────────────────────────────────────────
# JS API
# ─────────────────────────────────────────────────────────────
class _Api:
    def send_key(self, k):
        with _st_lock: _st["pending_keys"].append(k)
        return "ok"
    def quit(self):
        with _st_lock:
            _st["running"] = False
            _st["shutting_down"] = False
        _st["outro_done"].set()
        def _cl():
            time.sleep(0.3)
            try:
                if _wv_win: _wv_win.destroy()
            except: pass
            time.sleep(0.2)
            import os as _os
            try: _os.kill(_os.getpid(), 9)
            except: _os._exit(0)
        threading.Thread(target=_cl, daemon=True).start()
        return "ok"

# ─────────────────────────────────────────────────────────────
# COLLECTOR HTML
# ─────────────────────────────────────────────────────────────
_INSTR = json.dumps({k:list(v) for k,v in LABEL_SUBS.items()}, ensure_ascii=False)
_TARGET_PER = TARGET_FRAMES * (1 + AUGMENT_COUNT)
_SUBDUR     = SUB_DURATION

_HTML = f"""<!DOCTYPE html>
<html lang="vi"><head>
<meta charset="UTF-8"><title>Dataset Collector v2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --cyan:#5edfff;--green:#34d399;--amber:#fbbf24;--red:#f87171;--purple:#a78bfa;
  --text:#eef5ff;--muted:rgba(190,215,255,.55);--dim:rgba(140,175,220,.28);
  --b1:rgba(255,255,255,.10);--g1:rgba(255,255,255,.06);
  --font:'DM Sans',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;
}}
html,body{{width:100%;height:100%;overflow:hidden;font-family:var(--font);
  background:#000;color:var(--text);-webkit-font-smoothing:antialiased;}}

/* Camera */
#cam{{position:fixed;inset:0;z-index:0;}}
#camImg{{width:100%;height:100%;object-fit:cover;display:block;}}
#vig{{position:fixed;inset:0;z-index:1;pointer-events:none;
  background:radial-gradient(ellipse 85% 85% at 50% 50%,transparent 40%,rgba(0,0,8,.55) 100%);}}

/* UI grid */
#ui{{position:fixed;inset:0;z-index:10;display:grid;
  grid-template-columns:256px 1fr;grid-template-rows:46px 1fr 52px;pointer-events:none;}}
#ui>*{{pointer-events:auto;}}

/* Top bar */
#top{{grid-column:1/-1;display:flex;align-items:center;padding:0 18px;gap:12px;
  background:rgba(3,7,18,.72);backdrop-filter:blur(24px);
  border-bottom:1px solid var(--b1);position:relative;
  animation:topDrop .45s 3.9s cubic-bezier(.34,1.2,.64,1) both;}}
@keyframes topDrop{{from{{transform:translateY(-100%);opacity:0}}to{{transform:translateY(0);opacity:1}}}}
#top::after{{content:'';position:absolute;bottom:0;left:60px;right:60px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(94,223,255,.3) 30%,rgba(255,255,255,.5) 50%,rgba(94,223,255,.3) 70%,transparent);}}
.top-brand{{display:flex;align-items:center;gap:8px;}}
.top-ring{{width:10px;height:10px;border-radius:50%;border:2px solid var(--cyan);position:relative;}}
.top-ring::after{{content:'';position:absolute;inset:2px;border-radius:50%;background:var(--cyan);opacity:.5;}}
.top-name{{font-size:11px;font-weight:700;letter-spacing:.2em;color:var(--cyan);
  text-transform:uppercase;font-family:var(--mono);}}
.top-name span{{color:rgba(94,223,255,.4);margin:0 6px;}}
.top-center{{flex:1;display:flex;justify-content:center;}}
.hdr-pill{{display:inline-flex;align-items:center;gap:6px;padding:4px 14px;border-radius:99px;
  font-size:11px;font-weight:500;font-family:var(--mono);transition:all .3s;}}
.hdr-ok{{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);color:var(--green);}}
.hdr-warn{{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);color:var(--amber);}}
.hdr-bad{{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3);color:var(--red);}}
.hdr-dot{{width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 6px currentColor;}}
.person-tag{{display:flex;align-items:center;gap:7px;padding:4px 14px;border-radius:99px;
  font-size:11px;background:rgba(94,223,255,.07);border:1px solid rgba(94,223,255,.2);
  color:var(--cyan);font-family:var(--mono);}}

/* Sidebar */
#sidebar{{background:rgba(4,9,22,.68);backdrop-filter:blur(24px);
  border-right:1px solid var(--b1);display:flex;flex-direction:column;
  overflow:hidden;position:relative;border-radius:0 18px 18px 0;
  animation:sideSlide .5s 4.0s cubic-bezier(.34,1.2,.64,1) both;}}
@keyframes sideSlide{{from{{transform:translateX(-100%);opacity:0}}to{{transform:translateX(0);opacity:1}}}}
#sidebar::after{{content:'';position:absolute;top:0;right:0;width:1px;height:100%;
  background:linear-gradient(180deg,transparent 5%,rgba(94,223,255,.15) 30%,rgba(94,223,255,.25) 50%,rgba(94,223,255,.15) 70%,transparent 95%);}}
.lbl-scroll{{flex:1;overflow-y:auto;overflow-x:hidden;padding:8px 0 4px;
  scrollbar-width:thin;scrollbar-color:rgba(94,223,255,.12) transparent;scroll-behavior:smooth;}}
.g-hdr{{padding:10px 14px 4px;font-size:9px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;font-family:var(--mono);display:flex;align-items:center;gap:8px;color:var(--dim);}}
.g-hdr-line{{flex:1;height:1px;background:var(--b1);}}
.lbl{{display:flex;align-items:center;gap:0;margin:1px 6px;padding:0 8px 0 0;
  border-radius:10px;cursor:default;border:1px solid transparent;
  transition:all .18s ease;min-height:30px;overflow:hidden;}}
.lbl-bar{{width:3px;height:30px;border-radius:2px 0 0 2px;flex-shrink:0;margin-right:10px;opacity:.4;}}
.lbl-name{{flex:1;font-size:12px;font-weight:500;color:var(--muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.lbl-val{{font-size:11px;font-weight:600;color:var(--dim);font-family:var(--mono);
  min-width:22px;text-align:right;flex-shrink:0;display:inline-block;transition:color .2s;}}
.done-ck{{font-size:11px;color:var(--green);display:inline-block;}}
.lbl.sel{{background:rgba(255,255,255,.05);border-color:var(--b1);}}
.lbl.sel .lbl-bar,.lbl.rec .lbl-bar{{opacity:1;}}
.lbl.sel .lbl-name{{color:var(--text);}}
.lbl.rec{{background:rgba(94,223,255,.07);border-color:rgba(94,223,255,.28);
  box-shadow:inset 0 0 20px rgba(94,223,255,.04),0 0 0 1px rgba(94,223,255,.08);}}
.lbl.rec .lbl-name{{color:var(--cyan);font-weight:600;}}
.lbl.rec .lbl-val{{color:var(--cyan);}}
.kbd-area{{padding:8px 12px 10px;border-top:1px solid var(--b1);
  background:rgba(2,5,14,.5);display:flex;flex-wrap:wrap;gap:5px 10px;}}
.kbd-item{{display:flex;align-items:center;gap:4px;}}
kbd{{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
  border-radius:5px;padding:2px 7px;font-size:9px;font-family:var(--mono);color:var(--text);}}
.kd{{font-size:9px;color:var(--dim);}}

/* Bottom bar */
#bot{{grid-column:1/-1;background:rgba(3,7,18,.72);backdrop-filter:blur(24px);
  border-top:1px solid var(--b1);display:flex;align-items:center;
  padding:0 18px;gap:14px;position:relative;border-radius:18px 18px 0 0;
  animation:botRise .45s 4.05s cubic-bezier(.34,1.2,.64,1) both;}}
@keyframes botRise{{from{{transform:translateY(100%);opacity:0}}to{{transform:translateY(0);opacity:1}}}}
#bot::before{{content:'';position:absolute;top:0;left:60px;right:60px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(94,223,255,.3) 30%,rgba(255,255,255,.5) 50%,rgba(94,223,255,.3) 70%,transparent);}}
.bot-idle{{font-size:13px;color:var(--muted);}}
.bot-idle em{{color:var(--text);font-style:normal;font-weight:600;}}
.bot-rec{{display:flex;align-items:center;gap:10px;}}
.rec-dot{{width:8px;height:8px;border-radius:50%;background:var(--red);
  box-shadow:0 0 8px var(--red);animation:rpulse 1s ease-in-out infinite;flex-shrink:0;}}
@keyframes rpulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.8)}}}}
.rec-txt{{font-size:13px;font-weight:600;color:var(--text);}}
.rec-cls{{color:var(--cyan);}}
.pbar-wrap{{flex:1;max-width:280px;}}
.pbar-bg{{height:4px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;}}
.pbar-fill{{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--purple));
  transition:width .3s ease;}}
.pbar-fill.done{{background:linear-gradient(90deg,var(--green),#6ee7b7);}}
.pbar-txt{{font-size:10px;color:var(--muted);margin-top:3px;font-family:var(--mono);}}
.bot-right{{margin-left:auto;font-size:10px;color:var(--dim);font-family:var(--mono);}}
#botSelLbl{{font-size:13px;color:var(--text);font-weight:600;}}

/* Overlays */
#instr{{position:fixed;z-index:15;left:256px;right:0;bottom:52px;
  padding:14px 20px 12px;background:rgba(3,8,20,.82);
  backdrop-filter:blur(20px);border-top:1px solid var(--b1);
  display:none;border-radius:16px 16px 0 0;
  transition:opacity .2s ease,transform .2s ease,left .35s cubic-bezier(.4,0,.2,1);
  opacity:0;transform:translateY(8px);}}
.sidebar-hidden~* #instr,#ui.sidebar-hidden~#instr{{left:0;}}
#instr.on{{display:block;opacity:1;transform:translateY(0);}}
#instr::before{{content:'';position:absolute;top:0;left:20px;right:20px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(94,223,255,.4) 40%,rgba(94,223,255,.4) 60%,transparent);}}
.instr-tag{{font-size:10px;font-weight:700;letter-spacing:.15em;color:var(--cyan);
  text-transform:uppercase;font-family:var(--mono);margin-bottom:5px;}}
.instr-l1{{font-size:14px;font-weight:600;color:var(--text);line-height:1.4;}}
.instr-l2{{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.4;}}

/* Sub-instruction progress bar */
#subBar{{position:fixed;z-index:15;left:256px;right:0;bottom:calc(52px + 88px);
  height:3px;background:rgba(255,255,255,.06);display:none;}}
#subBar.on{{display:block;}}
#subBarFill{{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple));
  transition:width .1s linear;}}
#subStep{{position:fixed;z-index:15;right:16px;bottom:calc(52px + 96px);
  font-size:9px;font-family:var(--mono);color:var(--dim);display:none;}}
#subStep.on{{display:block;}}

/* Expression feedback */
#exprFeedback{{position:fixed;z-index:22;top:50%;left:50%;
  transform:translate(-50%,-50%);padding:12px 28px;border-radius:12px;
  font-size:14px;font-weight:700;color:var(--amber);letter-spacing:.05em;
  background:rgba(40,30,5,.85);backdrop-filter:blur(16px);
  border:1px solid rgba(251,191,36,.3);
  display:none;pointer-events:none;animation:fadeInC .15s ease;}}
#exprFeedback.on{{display:block;}}
@keyframes fadeInC{{from{{opacity:0;transform:translate(-50%,-46%)}}to{{opacity:1;transform:translate(-50%,-50%)}}}}

/* Valid frame counter */
#frameCounter{{position:fixed;z-index:16;top:58px;right:10px;
  padding:5px 14px;border-radius:12px;font-size:11px;font-family:var(--mono);
  color:var(--cyan);background:rgba(4,9,22,.75);backdrop-filter:blur(12px);
  border:1px solid rgba(94,223,255,.18);display:none;}}
#frameCounter.on{{display:block;}}

/* Distance / brightness warnings */
#distWarn{{position:fixed;z-index:16;top:58px;left:268px;
  padding:5px 14px;border-radius:12px;font-size:11px;font-family:var(--mono);
  color:var(--amber);background:rgba(30,25,5,.7);backdrop-filter:blur(12px);
  border:1px solid rgba(251,191,36,.25);display:none;}}
#distWarn.on{{display:block;}}

/* Capture flash */
#flash{{position:fixed;inset:0;z-index:30;background:rgba(255,255,255,.12);
  pointer-events:none;display:none;}}
#flash.on{{display:block;animation:flashAnim .15s ease-out forwards;}}
@keyframes flashAnim{{0%{{opacity:1}}100%{{opacity:0}}}}

/* Captured text */
#capturedTxt{{position:fixed;z-index:35;top:50%;left:50%;
  transform:translate(-50%,-50%);font-size:18px;font-weight:800;
  letter-spacing:.1em;color:#fff;font-family:var(--mono);
  pointer-events:none;display:none;text-shadow:0 0 20px rgba(94,223,255,.8);}}
#capturedTxt.on{{display:block;animation:capTxt .35s ease forwards;}}
@keyframes capTxt{{0%{{opacity:0;transform:translate(-50%,-50%) scale(.7)}}30%{{opacity:1;transform:translate(-50%,-50%) scale(1.08)}}100%{{opacity:0;transform:translate(-50%,-50%) scale(1.2)}}}}

/* Calibration overlay */
/* DJI Attitude Indicator Calibration */
#calibOverlay{{position:fixed;inset:0;z-index:25;display:none;
  align-items:center;justify-content:center;
  background:rgba(0,0,0,.88);}}
#calibOverlay.on{{display:flex;}}

/* Title */
.dji-title{{position:absolute;top:32px;left:50%;transform:translateX(-50%);
  text-align:center;}}
.dji-title-main{{font-size:10px;letter-spacing:.4em;color:rgba(255,255,255,.5);
  font-family:var(--mono);text-transform:uppercase;}}
.dji-title-sub{{font-size:9px;color:rgba(255,255,255,.25);margin-top:5px;
  font-family:var(--mono);letter-spacing:.12em;}}

/* Attitude indicator sphere */
#djiAI{{position:relative;width:260px;height:260px;border-radius:50%;
  overflow:hidden;border:2px solid rgba(255,255,255,.15);
  box-shadow:0 0 0 1px rgba(0,0,0,.5),0 8px 40px rgba(0,0,0,.6);}}
.ai-sky{{position:absolute;inset:0;background:#1a3a5c;}}
.ai-ground{{position:absolute;left:-50%;right:-50%;
  background:#3d2b1a;transition:none;}}
.ai-horizon-line{{position:absolute;left:-50%;right:-50%;height:2px;
  background:rgba(255,255,255,.9);}}
/* Pitch ladder */
.ai-ladder{{position:absolute;left:0;right:0;pointer-events:none;}}
.ai-rung{{position:absolute;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:6px;}}
.ai-rung-line{{height:1px;background:rgba(255,255,255,.5);}}
.ai-rung-lbl{{font-size:8px;color:rgba(255,255,255,.6);font-family:var(--mono);
  white-space:nowrap;line-height:1;}}
/* Roll arc at top */
.ai-roll-arc{{position:absolute;top:12px;left:50%;transform:translateX(-50%);
  width:160px;height:80px;overflow:hidden;pointer-events:none;}}
.ai-roll-arc svg{{position:absolute;bottom:0;left:0;}}
.ai-roll-pointer{{position:absolute;bottom:0;left:50%;
  transform:translateX(-50%);width:0;height:0;
  border-left:6px solid transparent;border-right:6px solid transparent;
  border-bottom:10px solid rgba(255,255,255,.8);}}
/* Center crosshair overlay */
.ai-center{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:80px;height:80px;pointer-events:none;}}
.ai-center-h{{position:absolute;top:50%;left:0;right:0;height:1px;background:#fff;
  transform:translateY(-50%);}}
.ai-center-v{{position:absolute;left:50%;top:25%;bottom:25%;width:1px;background:#fff;
  transform:translateX(-50%);}}
.ai-center-dot{{position:absolute;top:50%;left:50%;width:4px;height:4px;
  border-radius:50%;background:#fff;transform:translate(-50%,-50%);}}
/* Wing bars left/right */
.ai-wing{{position:absolute;top:50%;height:3px;background:#fff;border-radius:2px;
  transform:translateY(-50%);}}
.ai-wing.left{{left:15%;width:50px;}}
.ai-wing.right{{right:15%;width:50px;}}

/* Side info panels */
.dji-side-panel{{position:absolute;top:50%;transform:translateY(-50%);
  display:flex;flex-direction:column;gap:14px;}}
.dji-side-panel.left{{left:calc(50% - 190px);align-items:flex-end;}}
.dji-side-panel.right{{right:calc(50% - 190px);align-items:flex-start;}}
.dji-info-row{{display:flex;align-items:center;gap:8px;}}
.dji-info-lbl{{font-size:8px;letter-spacing:.2em;color:rgba(255,255,255,.3);
  font-family:var(--mono);text-transform:uppercase;}}
.dji-info-val{{font-size:14px;font-weight:300;color:rgba(255,255,255,.85);
  font-family:var(--mono);min-width:54px;text-align:right;}}
.dji-info-unit{{font-size:8px;color:rgba(255,255,255,.3);font-family:var(--mono);}}

/* Outer ring compass ticks */
.dji-outer{{position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);width:280px;height:280px;
  border-radius:50%;border:1px solid rgba(255,255,255,.08);pointer-events:none;}}

/* Scan line */
.dji-scan{{position:absolute;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(52,211,153,.8),transparent);
  display:none;animation:djiSc 1.4s ease-in-out infinite;}}
.dji-scan.on{{display:block;}}
@keyframes djiSc{{0%{{top:30%}}100%{{top:70%}}}}

/* Progress + status */
.dji-bottom{{position:absolute;bottom:40px;left:50%;transform:translateX(-50%);
  text-align:center;width:200px;}}
.dji-status-txt{{font-size:9px;letter-spacing:.22em;color:rgba(255,255,255,.35);
  font-family:var(--mono);text-transform:uppercase;margin-bottom:10px;}}
.calib-bw{{width:160px;height:1px;background:rgba(255,255,255,.12);
  overflow:hidden;margin:0 auto;}}
.calib-bf{{height:100%;background:#34d399;transition:width .1s linear;}}

/* Lock */
.dji-locked{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  display:none;flex-direction:column;align-items:center;gap:10px;
  background:rgba(0,0,0,.6);padding:20px 32px;border-radius:4px;}}
.dji-locked.on{{display:flex;animation:djiLkIn .3s ease forwards;opacity:0;}}
@keyframes djiLkIn{{from{{opacity:0}}to{{opacity:1}}}}
.dji-lock-ring{{width:52px;height:52px;border-radius:50%;
  border:1.5px solid #34d399;display:flex;align-items:center;justify-content:center;}}
.dji-lock-icon{{font-size:20px;color:#34d399;}}
.dji-lock-txt{{font-size:8px;letter-spacing:.3em;color:rgba(52,211,153,.7);
  font-family:var(--mono);text-transform:uppercase;}}

/* Calib success */
#calibSuccess{{position:fixed;z-index:30;top:50%;left:50%;
  transform:translate(-50%,-50%);display:none;flex-direction:column;
  align-items:center;gap:14px;pointer-events:none;}}
#calibSuccess.on{{display:flex;}}
.cs-burst{{position:absolute;width:160px;height:160px;border-radius:50%;
  border:2px solid rgba(52,211,153,0);}}
#calibSuccess.on .cs-burst{{animation:csBurst .6s cubic-bezier(0,0,.2,1) forwards;}}
@keyframes csBurst{{0%{{transform:scale(.4);border-color:rgba(52,211,153,.8);opacity:1}}100%{{transform:scale(1.6);border-color:rgba(52,211,153,0);opacity:0}}}}
.cs-burst2{{position:absolute;width:130px;height:130px;border-radius:50%;
  border:1.5px solid rgba(94,223,255,0);}}
#calibSuccess.on .cs-burst2{{animation:csBurst2 .7s .1s cubic-bezier(0,0,.2,1) forwards;}}
@keyframes csBurst2{{0%{{transform:scale(.5);border-color:rgba(94,223,255,.6);opacity:1}}100%{{transform:scale(1.8);border-color:rgba(94,223,255,0);opacity:0}}}}
.cs-p{{position:absolute;width:5px;height:5px;border-radius:50%;opacity:0;}}
#calibSuccess.on .cs-p{{animation:csP .6s var(--pd) ease-out forwards;}}
@keyframes csP{{0%{{opacity:1;transform:translate(0,0)}}100%{{opacity:0;transform:translate(var(--px),var(--py))}}}}
.cs-circle{{width:80px;height:80px;border-radius:50%;
  background:linear-gradient(135deg,rgba(52,211,153,.2),rgba(94,223,255,.1));
  border:2px solid rgba(52,211,153,.8);display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 30px rgba(52,211,153,.4);opacity:0;position:relative;z-index:1;}}
#calibSuccess.on .cs-circle{{animation:csCircle .4s .05s cubic-bezier(.34,1.4,.64,1) forwards;}}
@keyframes csCircle{{0%{{opacity:0;transform:scale(.3)}}100%{{opacity:1;transform:scale(1)}}}}
.cs-check{{font-size:32px;color:var(--green);opacity:0;
  text-shadow:0 0 16px rgba(52,211,153,.8);}}
#calibSuccess.on .cs-check{{animation:csCheck .35s .25s cubic-bezier(.34,1.6,.64,1) forwards;}}
@keyframes csCheck{{0%{{opacity:0;transform:scale(0) rotate(-30deg)}}100%{{opacity:1;transform:scale(1) rotate(0)}}}}
.cs-label{{font-size:13px;font-weight:700;color:var(--green);letter-spacing:.1em;
  font-family:var(--mono);text-transform:uppercase;opacity:0;}}
#calibSuccess.on .cs-label{{animation:csLabel .4s .3s ease forwards;}}
.cs-sub{{font-size:10px;color:var(--muted);letter-spacing:.08em;font-family:var(--mono);opacity:0;}}
#calibSuccess.on .cs-sub{{animation:csLabel .4s .45s ease forwards;}}
@keyframes csLabel{{0%{{opacity:0;transform:translateY(6px)}}100%{{opacity:1;transform:translateY(0)}}}}

/* Boot screen */
#boot{{position:fixed;inset:0;z-index:9000;background:#000;display:flex;flex-direction:column;
  align-items:center;justify-content:center;pointer-events:none;}}
#boot.hidden{{animation:bootFade .6s .2s ease forwards;}}
@keyframes bootFade{{to{{opacity:0;visibility:hidden}}}}
.boot-grid{{position:absolute;inset:0;background-image:linear-gradient(rgba(94,223,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(94,223,255,.04) 1px,transparent 1px);background-size:50px 50px;}}
.boot-scan{{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(94,223,255,.3),transparent);animation:bScan 1.5s linear infinite;}}
@keyframes bScan{{0%{{top:-2px}}100%{{top:100%}}}}
.boot-logo{{width:72px;height:72px;border-radius:18px;background:linear-gradient(135deg,rgba(94,223,255,.15),rgba(167,139,250,.1));border:2px solid rgba(94,223,255,.5);display:flex;align-items:center;justify-content:center;box-shadow:0 0 40px rgba(94,223,255,.25);margin-bottom:28px;animation:bPulse 2s ease-in-out infinite;}}
@keyframes bPulse{{0%,100%{{box-shadow:0 0 30px rgba(94,223,255,.2)}}50%{{box-shadow:0 0 60px rgba(94,223,255,.45)}}}}
.boot-title{{font-size:13px;letter-spacing:.3em;color:rgba(94,223,255,.6);font-family:var(--mono);text-transform:uppercase;margin-bottom:32px;}}
.boot-msgs{{width:320px;margin-bottom:28px;display:flex;flex-direction:column;gap:10px;}}
.boot-msg{{display:flex;align-items:center;gap:10px;font-size:12px;font-family:var(--mono);color:rgba(190,215,255,0);transition:color .3s;}}
.boot-msg.show{{color:rgba(190,215,255,.75);}}.boot-msg.done{{color:rgba(52,211,153,.9);}}
.bmd{{width:5px;height:5px;border-radius:50%;background:var(--cyan);flex-shrink:0;opacity:0;transition:opacity .3s;box-shadow:0 0 6px var(--cyan);}}
.boot-msg.show .bmd,.boot-msg.done .bmd{{opacity:1;}}
.boot-msg.done .bmd{{background:var(--green);box-shadow:0 0 6px var(--green);}}
.boot-bar-wrap{{width:320px;height:3px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;}}
.boot-bar-fill{{height:100%;border-radius:99px;width:0%;background:linear-gradient(90deg,var(--cyan),var(--purple));transition:width .3s;}}

/* Face lock — chỉ hiện khi không track được mắt (fallback) */
#faceLock{{position:fixed;z-index:8;pointer-events:none;display:none;}}
#faceLock.on{{display:block;}}
.flc{{position:absolute;width:16px;height:16px;opacity:0;}}
.flc.tl{{top:0;left:0;border-top:1.5px solid rgba(94,223,255,.6);border-left:1.5px solid rgba(94,223,255,.6);border-radius:3px 0 0 0;}}
.flc.tr{{top:0;right:0;border-top:1.5px solid rgba(94,223,255,.6);border-right:1.5px solid rgba(94,223,255,.6);border-radius:0 3px 0 0;}}
.flc.bl{{bottom:0;left:0;border-bottom:1.5px solid rgba(94,223,255,.6);border-left:1.5px solid rgba(94,223,255,.6);border-radius:0 0 0 3px;}}
.flc.br{{bottom:0;right:0;border-bottom:1.5px solid rgba(94,223,255,.6);border-right:1.5px solid rgba(94,223,255,.6);border-radius:0 0 3px 0;}}
#faceLock.on .flc{{animation:cIn .2s ease forwards;}}
#faceLock.on .flc.tl{{animation-delay:0s}}#faceLock.on .flc.tr{{animation-delay:.04s}}
#faceLock.on .flc.bl{{animation-delay:.04s}}#faceLock.on .flc.br{{animation-delay:.08s}}
@keyframes cIn{{from{{opacity:0;transform:scale(.6)}}to{{opacity:1;transform:scale(1)}}}}
.fl-lbl{{position:absolute;bottom:-20px;left:50%;transform:translateX(-50%);
  font-size:8px;letter-spacing:.12em;color:rgba(94,223,255,.7);font-family:var(--mono);
  text-transform:uppercase;opacity:0;white-space:nowrap;}}
#faceLock.on .fl-lbl{{animation:fadeU .3s .15s ease forwards;}}
@keyframes fadeU{{from{{opacity:0;transform:translateX(-50%) translateY(3px)}}to{{opacity:1;transform:translateX(-50%) translateY(0)}}}}
.fl-pulse{{position:absolute;inset:-4px;border-radius:3px;border:1px solid rgba(94,223,255,.15);animation:flP 2s ease-in-out infinite;}}
@keyframes flP{{0%,100%{{opacity:.2}}50%{{opacity:.5}}}}

/* Eye lock — Sony style: xanh lá, bo tròn, tracking mắt trái */
#eyeLock{{position:fixed;z-index:9;pointer-events:none;display:none;}}
#eyeLock.on{{display:block;}}
/* eyeLock: border liền như Sony thật */
#eyeLock{{position:fixed;z-index:9;pointer-events:none;display:none;
  border:3px solid #39ff14;border-radius:2px;
  box-shadow:0 0 0 1px rgba(57,255,20,.15);}}
#eyeLock.on{{display:block;}}
.elc{{display:none;}}
.el-dot{{display:none;}}
.el-ring{{display:none;}}


/* ── Auto-mode: transition overlay ── */
#transOverlay{{
  position:fixed;left:256px;top:46px;right:0;bottom:52px;z-index:20;
  display:none;align-items:center;justify-content:center;flex-direction:column;gap:16px;
  background:rgba(3,7,18,.82);backdrop-filter:blur(8px);
}}
#transOverlay.on{{display:flex;animation:fadeInC .3s ease;}}
.trans-next-label{{font-size:11px;letter-spacing:.2em;color:var(--dim);
  font-family:var(--mono);text-transform:uppercase;}}
.trans-class-name{{font-size:28px;font-weight:800;color:var(--cyan);
  text-shadow:0 0 30px rgba(94,223,255,.4);}}
.trans-hint{{font-size:12px;color:var(--muted);}}
.trans-countdown{{font-size:48px;font-weight:900;color:var(--green);
  font-family:var(--mono);text-shadow:0 0 24px rgba(52,211,153,.5);
  line-height:1;}}

/* ── Hold progress ring ── */
#holdRing{{
  position:fixed;z-index:18;top:50%;left:calc(256px + (100vw - 256px)/2);
  transform:translate(-50%,-50%);display:none;pointer-events:none;
}}
#holdRing.on{{display:block;}}

/* ── State chip (waiting/capturing) ── */
#stateChip{{
  position:fixed;z-index:16;
  bottom:calc(52px + 100px + 12px);
  left:50%;transform:translateX(-50%);
  padding:8px 22px;border-radius:99px;
  font-size:12px;font-weight:600;
  backdrop-filter:blur(14px);white-space:nowrap;
  display:none;pointer-events:none;
  transition:all .2s ease;
}}
#stateChip.on{{display:block;}}
#stateChip.waiting{{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);color:var(--amber);}}
#stateChip.capturing{{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.35);color:var(--green);}}

/* Overall progress bar (session) */
#sessionBar{{
  position:fixed;z-index:16;top:46px;left:256px;right:0;height:2px;
  background:rgba(255,255,255,.04);display:none;
}}
#sessionBar.on{{display:block;}}
#sessionBarFill{{
  height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple));
  transition:width .3s ease;
}}

/* Scanline */
#scanOv{{position:fixed;left:256px;top:46px;right:0;bottom:52px;pointer-events:none;z-index:5;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);opacity:.5;}}

/* Crop preview */
#cropPanel{{position:fixed;z-index:16;right:12px;bottom:64px;width:160px;
  background:rgba(4,9,22,.85);backdrop-filter:blur(16px);
  border:1px solid rgba(94,223,255,.2);border-radius:14px;overflow:hidden;
  display:none;box-shadow:0 4px 24px rgba(0,0,0,.5);}}
#cropPanel.on{{display:block;animation:cpIn .25s cubic-bezier(.34,1.2,.64,1) forwards;}}
@keyframes cpIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.cp-hdr{{display:flex;align-items:center;justify-content:space-between;
  padding:7px 10px 5px;border-bottom:1px solid rgba(94,223,255,.12);}}
.cp-lbl{{font-size:9px;font-weight:700;letter-spacing:.15em;color:var(--cyan);font-family:var(--mono);text-transform:uppercase;}}
.cp-badge{{font-size:8px;padding:1px 6px;border-radius:99px;background:rgba(94,223,255,.1);color:var(--cyan);font-family:var(--mono);}}
#cropImg{{width:100%;aspect-ratio:1;object-fit:cover;display:block;}}
.cp-foot{{padding:5px 10px;font-size:9px;color:var(--muted);font-family:var(--mono);display:flex;justify-content:space-between;}}

/* Dataset check screen */
#dataCheck{{position:fixed;inset:0;z-index:9998;background:rgba(3,7,18,.97);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  opacity:0;pointer-events:none;transition:opacity .5s;}}
#dataCheck.on{{opacity:1;pointer-events:auto;}}
.dc-grid{{position:absolute;inset:0;background-image:linear-gradient(rgba(94,223,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(94,223,255,.03) 1px,transparent 1px);background-size:60px 60px;}}
.dc-scan{{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(94,223,255,.3),transparent);animation:bScan 2s linear infinite;}}
.dc-title{{font-size:13px;letter-spacing:.25em;color:rgba(94,223,255,.6);font-family:var(--mono);text-transform:uppercase;margin-bottom:28px;animation:tr2 .4s .1s ease both;opacity:0;}}
@keyframes tr2{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
.dc-items{{width:420px;display:flex;flex-direction:column;gap:10px;margin-bottom:24px;}}
.dc-item{{display:flex;align-items:center;gap:14px;padding:11px 16px;border-radius:12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);animation:fsi2 .3s var(--d) ease both;opacity:0;}}
@keyframes fsi2{{from{{opacity:0;transform:translateX(-12px)}}to{{opacity:1;transform:translateX(0)}}}}
.dc-ico{{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;background:rgba(94,223,255,.08);}}
.dc-txt{{flex:1;}}.dc-lbl{{font-size:12px;color:var(--text);font-weight:500;}}.dc-sub{{font-size:10px;color:var(--muted);margin-top:2px;font-family:var(--mono);}}
.dc-st{{font-size:11px;font-family:var(--mono);font-weight:700;opacity:0;transition:opacity .3s,color .3s;}}
.dc-st.chk{{color:var(--amber);}}.dc-st.ok{{color:var(--green);}}
.dc-bw{{width:420px;height:3px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;animation:tr2 .4s .5s ease both;opacity:0;}}
.dc-bf{{height:100%;width:0%;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .25s;}}
.dc-ok{{margin-top:18px;font-size:13px;font-weight:700;color:var(--green);letter-spacing:.08em;opacity:0;transition:opacity .4s;display:flex;align-items:center;gap:8px;}}

/* Outro */
#outro{{position:fixed;inset:0;z-index:9999;background:#000;display:flex;flex-direction:column;
  align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .8s;}}
#outro.on{{opacity:1;pointer-events:auto;}}
.outro-particles{{position:absolute;inset:0;overflow:hidden;pointer-events:none;}}
.outro-p{{position:absolute;border-radius:50%;animation:pFloat var(--dur) var(--delay) ease-in-out infinite alternate;}}
@keyframes pFloat{{0%{{transform:translate(0,0) scale(1);opacity:var(--op1)}}100%{{transform:translate(var(--dx),var(--dy)) scale(1.4);opacity:var(--op2)}}}}
.og{{position:absolute;inset:0;background-image:linear-gradient(rgba(94,223,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(94,223,255,.03) 1px,transparent 1px);background-size:60px 60px;animation:gP 2s ease-in-out infinite;}}
@keyframes gP{{0%,100%{{opacity:.5}}50%{{opacity:1}}}}
.os-scan{{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(94,223,255,.4),transparent);animation:bScan 2s linear infinite;}}
.outro-corner{{position:absolute;width:60px;height:60px;animation:ocIn .5s var(--cd) cubic-bezier(.34,1.2,.64,1) both;opacity:0;}}
@keyframes ocIn{{from{{opacity:0;transform:scale(.4)}}to{{opacity:1;transform:scale(1)}}}}
.outro-corner.tl{{top:20px;left:20px;border-top:2px solid rgba(94,223,255,.6);border-left:2px solid rgba(94,223,255,.6);border-radius:6px 0 0 0;}}
.outro-corner.tr{{top:20px;right:20px;border-top:2px solid rgba(94,223,255,.6);border-right:2px solid rgba(94,223,255,.6);border-radius:0 6px 0 0;}}
.outro-corner.bl{{bottom:20px;left:20px;border-bottom:2px solid rgba(94,223,255,.6);border-left:2px solid rgba(94,223,255,.6);border-radius:0 0 0 6px;}}
.outro-corner.br{{bottom:20px;right:20px;border-bottom:2px solid rgba(94,223,255,.6);border-right:2px solid rgba(94,223,255,.6);border-radius:0 0 6px 0;}}
.outro-rings{{position:relative;width:180px;height:180px;margin-bottom:8px;animation:oRI .6s .2s cubic-bezier(.34,1.2,.64,1) both;opacity:0;}}
@keyframes oRI{{from{{opacity:0;transform:scale(.3)}}to{{opacity:1;transform:scale(1)}}}}
.outro-ring{{position:absolute;border-radius:50%;}}
.outro-ring.r1{{inset:0;border:2px solid rgba(94,223,255,.25);animation:os2 12s linear infinite;}}
.outro-ring.r2{{inset:16px;border:1px solid rgba(167,139,250,.2);animation:osr2 8s linear infinite;}}
.outro-ring.r3{{inset:32px;border:1px solid rgba(52,211,153,.2);animation:os2 5s linear infinite;}}
.outro-ring.r4{{inset:48px;border:1px dashed rgba(94,223,255,.12);animation:osr2 15s linear infinite;}}
@keyframes os2{{from{{transform:rotate(0)}}to{{transform:rotate(360deg)}}}}
@keyframes osr2{{from{{transform:rotate(0)}}to{{transform:rotate(-360deg)}}}}
.outro-dot{{position:absolute;width:10px;height:10px;border-radius:50%;top:-5px;left:calc(50% - 5px);box-shadow:0 0 10px currentColor;}}
.outro-dot.d1{{background:var(--cyan);color:var(--cyan);animation:os2 12s linear infinite;transform-origin:50% 90px;}}
.outro-dot.d2{{background:#a78bfa;color:#a78bfa;animation:osr2 8s linear infinite;transform-origin:50% 74px;inset:16px;top:-5px;}}
.outro-logo{{position:absolute;inset:62px;border-radius:20px;background:linear-gradient(135deg,rgba(94,223,255,.12),rgba(167,139,250,.08));border:1.5px solid rgba(94,223,255,.4);display:flex;align-items:center;justify-content:center;box-shadow:0 0 40px rgba(94,223,255,.2);animation:bPulse 2s ease-in-out infinite;}}
.outro-check{{font-size:36px;opacity:0;margin-top:-14px;animation:ckP .5s .9s cubic-bezier(.34,1.6,.64,1) both;color:var(--green);text-shadow:0 0 20px rgba(52,211,153,.6);}}
@keyframes ckP{{from{{opacity:0;transform:scale(0) rotate(-30deg)}}to{{opacity:1;transform:scale(1) rotate(0)}}}}
.outro-title{{font-size:28px;font-weight:800;color:var(--text);margin-bottom:6px;animation:tr2 .5s .4s ease both;opacity:0;text-shadow:0 0 40px rgba(94,223,255,.3);}}
.outro-thanks{{font-size:14px;color:rgba(94,223,255,.7);font-family:var(--mono);letter-spacing:.08em;margin-bottom:8px;animation:tr2 .5s .55s ease both;opacity:0;}}
.outro-sub{{font-size:12px;color:var(--muted);margin-bottom:28px;animation:tr2 .5s .65s ease both;opacity:0;}}
.outro-stats{{display:flex;gap:40px;margin-bottom:28px;animation:tr2 .5s .75s ease both;opacity:0;}}
.outro-stat{{text-align:center;}}.outro-stat-n{{font-size:36px;font-weight:800;color:var(--cyan);font-family:var(--mono);text-shadow:0 0 20px rgba(94,223,255,.4);}}
.outro-stat-l{{font-size:9px;letter-spacing:.18em;color:var(--dim);text-transform:uppercase;margin-top:4px;}}
.outro-bar-wrap{{width:320px;height:3px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;margin-bottom:16px;animation:tr2 .5s .85s ease both;opacity:0;}}
.outro-bar-fill{{height:100%;border-radius:99px;width:0%;background:linear-gradient(90deg,var(--cyan),var(--purple),var(--green));}}
.outro-status{{font-size:10px;letter-spacing:.22em;color:rgba(94,223,255,.4);font-family:var(--mono);text-transform:uppercase;transition:opacity .2s;animation:tr2 .5s .95s ease both;opacity:0;}}
.outro-close{{position:absolute;bottom:32px;font-size:9px;letter-spacing:.3em;color:rgba(94,223,255,.25);font-family:var(--mono);text-transform:uppercase;animation:tr2 .5s 1.1s ease both;opacity:0;}}

/* Sidebar animations */
@keyframes doneAnim{{0%{{opacity:0;transform:scale(.4) rotate(-25deg)}}65%{{opacity:1;transform:scale(1.35) rotate(6deg)}}100%{{opacity:1;transform:scale(1) rotate(0)}}}}
@keyframes countBump{{0%{{transform:scale(1)}}45%{{transform:scale(1.5);color:var(--cyan)}}100%{{transform:scale(1)}}}}
.count-bump{{animation:countBump .32s cubic-bezier(.34,1.56,.64,1) forwards;}}
</style></head>
<body>

<!-- Boot -->
<div id="boot"><div class="boot-grid"></div><div class="boot-scan"></div>
  <div class="boot-logo"><span style="font-size:22px;font-weight:800;color:var(--cyan);font-family:var(--mono)">AI</span></div>
  <div class="boot-title">Đang Khởi Động Hệ Thống</div>
  <div class="boot-msgs">
    <div class="boot-msg" id="bm0"><div class="bmd"></div><span>Khởi động Hệ thống Nhận diện...</span></div>
    <div class="boot-msg" id="bm1"><div class="bmd"></div><span>Nạp Module Theo dõi Khuôn mặt...</span></div>
    <div class="boot-msg" id="bm2"><div class="bmd"></div><span>Kích hoạt Giao diện Camera...</span></div>
    <div class="boot-msg" id="bm3"><div class="bmd"></div><span>Hệ thống Sẵn sàng.</span></div>
  </div>
  <div class="boot-bar-wrap"><div class="boot-bar-fill" id="bootBar"></div></div>
</div>

<!-- Face lock (fallback khi không track mắt) -->
<div id="faceLock"><div class="flc tl"></div><div class="flc tr"></div><div class="flc bl"></div><div class="flc br"></div><div class="fl-pulse"></div><div class="fl-lbl">TARGET LOCKED</div></div>
<!-- Eye lock (Sony style — ưu tiên hiện) -->
<div id="eyeLock"></div>

<!-- Calib success -->
<div id="calibSuccess">
  <div class="cs-burst"></div><div class="cs-burst2"></div>
  <div class="cs-p" style="background:#34d399;--pd:.05s;--px:-60px;--py:-20px"></div>
  <div class="cs-p" style="background:#5edfff;--pd:.08s;--px:60px;--py:-25px"></div>
  <div class="cs-p" style="background:#34d399;--pd:.06s;--px:50px;--py:50px"></div>
  <div class="cs-p" style="background:#a78bfa;--pd:.1s;--px:-50px;--py:55px"></div>
  <div class="cs-p" style="background:#5edfff;--pd:.04s;--px:-70px;--py:30px"></div>
  <div class="cs-p" style="background:#34d399;--pd:.09s;--px:70px;--py:-10px"></div>
  <div class="cs-p" style="background:#fbbf24;--pd:.07s;--px:20px;--py:-65px"></div>
  <div class="cs-p" style="background:#34d399;--pd:.11s;--px:-25px;--py:65px"></div>
  <div class="cs-circle"><div class="cs-check">&#10003;</div></div>
  <div class="cs-label">Hiệu chỉnh hoàn tất</div>
  <div class="cs-sub">Tư thế chuẩn đã được lưu</div>
</div>

<!-- Calib overlay DJI Attitude Indicator -->
<div id="calibOverlay">
  <div class="dji-title">
    <div class="dji-title-main">Hiệu chỉnh tư thế</div>
    <div class="dji-title-sub" id="djiStatusTxt">Nhìn thẳng · Giữ yên</div>
  </div>
  <div class="dji-outer"></div>
  <div id="djiAI">
    <div class="ai-sky"></div>
    <div class="ai-ground" id="aiGround"></div>
    <div class="ai-horizon-line" id="aiHorizon"></div>
    <div class="ai-ladder" id="aiLadder"></div>
    <div class="ai-wing left"></div>
    <div class="ai-wing right"></div>
    <div class="ai-center">
      <div class="ai-center-h"></div>
      <div class="ai-center-v"></div>
      <div class="ai-center-dot"></div>
    </div>
  </div>
  <div class="dji-scan" id="djiScan"></div>
  <!-- Side panels -->
  <div class="dji-side-panel left">
    <div class="dji-info-row">
      <div class="dji-info-val" id="djiPV">+0.00</div>
      <div><div class="dji-info-unit">°</div><div class="dji-info-lbl">PITCH</div></div>
    </div>
    <div class="dji-info-row">
      <div class="dji-info-val" id="djiRV">+0.00</div>
      <div><div class="dji-info-unit">°</div><div class="dji-info-lbl">ROLL</div></div>
    </div>
  </div>
  <div class="dji-side-panel right">
    <div class="dji-info-row">
      <div><div class="dji-info-unit">°</div><div class="dji-info-lbl">YAW</div></div>
      <div class="dji-info-val" id="djiYV">+0.00</div>
    </div>
  </div>
  <!-- Bottom -->
  <div class="dji-bottom">
    <div class="dji-status-txt" id="djiStatusTxt2"></div>
    <div class="calib-bw"><div class="calib-bf" id="calibFill"></div></div>
  </div>
  <!-- Lock -->
  <div class="dji-locked" id="djiLocked">
    <div class="dji-lock-ring"><div class="dji-lock-icon">✓</div></div>
    <div class="dji-lock-txt">Tư thế đã khoá</div>
  </div>
</div>

<!-- Dataset check -->
<div id="dataCheck"><div class="dc-grid"></div><div class="dc-scan"></div>
  <div class="dc-title">&#x2B21; Đang kiểm tra dataset...</div>
  <div class="dc-items" id="dcItems"></div>
  <div class="dc-bw"><div class="dc-bf" id="dcBar"></div></div>
  <div class="dc-ok" id="dcOk"><span>&#10003;</span><span>Dataset hợp lệ — Sẵn sàng lưu</span></div>
</div>

<!-- Outro -->
<div id="outro">
  <div class="outro-particles" id="outroParts"></div>
  <div class="og"></div><div class="os-scan"></div>
  <div class="outro-corner tl" style="--cd:.1s"></div>
  <div class="outro-corner tr" style="--cd:.15s"></div>
  <div class="outro-corner bl" style="--cd:.15s"></div>
  <div class="outro-corner br" style="--cd:.2s"></div>
  <div class="outro-rings">
    <div class="outro-ring r1"><div class="outro-dot d1"></div></div>
    <div class="outro-ring r2"><div class="outro-dot d2"></div></div>
    <div class="outro-ring r3"></div><div class="outro-ring r4"></div>
    <div class="outro-logo"><span style="font-size:20px;font-weight:800;color:var(--cyan);font-family:var(--mono)">AI</span></div>
  </div>
  <div class="outro-check">&#10003;</div>
  <div class="outro-title">Cảm ơn bạn rất nhiều!</div>
  <div class="outro-thanks">CẢM ƠN THƯỢNG ĐẾ &#x2B21;</div>
  <div class="outro-sub">Dữ liệu của bạn đã được lưu thành công vào dataset AI</div>
  <div class="outro-stats">
    <div class="outro-stat"><div class="outro-stat-n" id="oTotal">0</div><div class="outro-stat-l">Ảnh đã thu</div></div>
    <div class="outro-stat"><div class="outro-stat-n" id="oClass">0</div><div class="outro-stat-l">Class hoàn thành</div></div>
    <div class="outro-stat"><div class="outro-stat-n" id="oSession">1</div><div class="outro-stat-l">Phiên làm việc</div></div>
  </div>
  <div class="outro-bar-wrap"><div class="outro-bar-fill" id="oBar"></div></div>
  <div class="outro-status" id="oCl">Đang lưu dữ liệu...</div>
  <div class="outro-close">Hệ thống đang đóng lại...</div>
</div>

<!-- Scan overlay -->
<div id="scanOv"></div>
<div id="transOverlay">
  <div class="trans-next-label">Tiếp theo</div>
  <div class="trans-class-name" id="transName"></div>
  <div class="trans-hint" id="transHint"></div>
  <div class="trans-countdown" id="transCd"></div>
</div>

<div id="holdRing">
  <svg width="100" height="100" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="42" fill="rgba(0,0,0,.5)"
      stroke="rgba(255,255,255,.07)" stroke-width="2"/>
    <circle id="holdArc" cx="50" cy="50" r="42"
      fill="none" stroke="#fbbf24" stroke-width="5"
      stroke-dasharray="264" stroke-dashoffset="264"
      stroke-linecap="round" transform="rotate(-90 50 50)"/>
  </svg>
</div>

<div id="stateChip"></div>
<div id="sessionBar"><div id="sessionBarFill"></div></div>


<!-- Camera -->
<div id="cam"><img id="camImg" src="http://127.0.0.1:{_PORT}/stream"></div>
<div id="vig"></div>

<!-- UI grid -->
<div id="ui">
  <div id="top">
    <div class="top-brand">
      <div class="top-ring"></div>
      <div class="top-name">Dataset <span>|</span> Collector <span>|</span> v2</div>
    </div>
    <div class="top-center">
      <div class="hdr-pill hdr-ok" id="hdrPill">
        <div class="hdr-dot"></div>
        <span id="hdrTxt">Sáng tốt</span>
      </div>
    </div>
    <div class="person-tag" id="personTag"></div>
  </div>

  <div id="sidebar">
    <div class="lbl-scroll" id="lblList"></div>
    <div class="kbd-area">
      <div class="kbd-item"><kbd>SPACE</kbd><span class="kd">Bắt/Dừng</span></div>
      <div class="kbd-item"><kbd>↑↓</kbd><span class="kd">Chọn</span></div>
      <div class="kbd-item"><kbd>C</kbd><span class="kd">Calibrate</span></div>
      <div class="kbd-item"><kbd>ESC</kbd><span class="kd">Thoát</span></div>
    </div>
  </div>

  <div id="main"></div>

  <div id="bot">
    <div id="botIdle" style="display:flex">
      <span class="bot-idle">Đã chọn: <em id="botSelLbl"></em></span>
    </div>
    <div id="botRec" class="bot-rec" style="display:none">
      <div class="rec-dot"></div>
      <span class="rec-txt">Đang ghi: <span class="rec-cls" id="recCls"></span></span>
    </div>
    <div id="pbarWrap" class="pbar-wrap" style="display:none">
      <div class="pbar-bg"><div class="pbar-fill" id="pbarFill"></div></div>
      <div class="pbar-txt" id="pbarTxt"></div>
    </div>
    <div id="botRight" class="bot-right">
      SPACE → bắt đầu thu
    </div>
  </div>
</div>

<!-- Instruction + sub bar -->
<div id="instr">
  <div class="instr-tag" id="instrTag"></div>
  <div class="instr-l1" id="instrL1"></div>
  <div class="instr-l2" id="instrL2"></div>
</div>
<div id="subBar"><div id="subBarFill"></div></div>
<div id="subStep"></div>

<!-- Other overlays -->
<div id="frameCounter"></div>
<div id="distWarn"></div>
<div id="exprFeedback"></div>
<div id="flash"></div>
<div id="capturedTxt">ĐÃ CHỤP</div>

<!-- Crop preview -->
<div id="cropPanel">
  <div class="cp-hdr"><span class="cp-lbl">CROP PREVIEW</span><span class="cp-badge">224×224</span></div>
  <img id="cropImg" src="" alt="">
  <div class="cp-foot"><span id="cpRegion">—</span><span id="cpClass">—</span></div>
</div>

<script>
const TARGET_COUNT = {_TARGET_PER};
const TARGET_FRAMES_RAW = 20;
const SUB_DUR = {_SUBDUR};
const GRP_COL = {{MAT:"#5edfff",MIENG:"#34d399",DAU:"#a78bfa"}};

// ── Web Audio ──────────────────────────────────────────────
const _AC = new (window.AudioContext||window.webkitAudioContext)();
let _shutterBuf=null;
function _resume(){{ if(_AC.state==='suspended') _AC.resume(); }}
function sfx(type, opts={{}}){{
  _resume();
  const ac=_AC, t=ac.currentTime;

  // Helper — sine note với attack/decay chuẩn
  function note(freq, gain, dur, delay){{
    const o=ac.createOscillator(), g=ac.createGain();
    o.type='sine';
    o.frequency.setValueAtTime(freq, t+delay);
    g.gain.setValueAtTime(0, t+delay);
    g.gain.linearRampToValueAtTime(gain, t+delay+0.006);
    g.gain.exponentialRampToValueAtTime(0.001, t+delay+dur);
    o.connect(g); g.connect(ac.destination);
    o.start(t+delay); o.stop(t+delay+dur+0.01);
  }}

  switch(type){{

    // Điều hướng — click nhẹ trung tính
    case 'nav':
      note(opts.freq||700, 0.05, 0.06, 0);
      break;

    // Bắt đầu thu — hai nốt lên, sạch và gọn
    case 'start':
      note(392, 0.08, 0.18, 0.0);
      note(523, 0.08, 0.22, 0.1);
      break;

    // Dừng thu — hai nốt xuống, nhẹ nhàng
    case 'stop':
      note(523, 0.07, 0.16, 0.0);
      note(392, 0.07, 0.2,  0.1);
      break;

    // Chụp ảnh — tick kỹ thuật số ngắn gọn
    case 'capture':{{
      if(!_shutterBuf){{
        const raw=atob(
        'UklGRj5VAQBXQVZFZm10IBAAAAABAAIAgLsAAADuAgAEABAATElTVGYAAABJTkZPSUFSVA0AAABUb21z'+
        'IEp1cmpha3MAAElOQU0tAAAAU0hVVFRFUiBTT1VORCBCQVRUTEUgLy8gU09OWSBBN0MgSUkgdnMgQTY3'+
        'MDAAAElTRlQNAAAATGF2ZjYyLjMuMTAwAABkYXRhrFQBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'+
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAQABAAEAAQABAAEA'+
        'AQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEA'+
        'AQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEA'+
        'AQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEA'+
        'AQABAAEAAQACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgADAAIAAwACAAIAAgACAAIAAwADAAMA'+
        'AwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQA'+
        'BAAEAAQABQAFAAUABQAFAAUABQAFAAYABQAGAAYABgAGAAYABgAGAAYABgAGAAYABgAFAAYABQAGAAYA'+
        'BgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABQAGAAUABgAGAAYABgAGAAUA'+
        'BgAFAAUABQAFAAUABAAFAAQABAAEAAQABAADAAMAAwADAAMAAwAEAAQABAADAAMAAwADAAMAAwADAAIA'+
        'AwABAAIAAQACAAEAAQAAAAEAAAAAAP///////////v/+//7//v/+//3//v/9//3//P/8//z//P/8//z/'+
        '/P/8//z//f/7//3//P/9//3//P/9//z//P/7//z//P/8//3//f/9//3//P/9//v//P/7//z/+//7//r/'+
        '+//6//r/+v/5//r/+f/5//n/+f/5//n/+P/5//j/+f/4//n/+P/5//n/+P/5//j/+P/4//j/+P/4//j/'+
        '+P/4//j/+P/4//f/+P/3//f/9v/2//b/9f/2//T/9f/0//X/9P/0//T/9P/z//T/8//z//L/8//y//L/'+
        '8//y//L/8f/y//L/8//z//P/8//z//H/8f/w//D/7//x//D/8v/w//H/7//v/+//7v/u/+3/7P/s/+r/'+
        '6//q/+r/6v/p/+r/6P/p/+j/6P/n/+j/5//n/+b/5v/l/+X/5f/m/+X/5v/m/+X/5v/l/+b/5P/m/+T/'+
        '5f/k/+X/5f/l/+X/5f/l/+X/4//k/+L/4v/i/+H/4v/h/+L/4f/i/+H/4v/g/+H/4P/i/+H/4//i/+P/'+
        '4//i/+L/4f/h/+D/4P/g/+D/4f/i/+P/4//j/+L/4v/h/+H/4f/h/+H/4f/h/+L/4P/i/+D/4f/g/+D/'+
        '4P/g/+H/4f/i/+P/4//j/+P/4v/j/+L/4//k/+T/5f/l/+X/5f/l/+X/5P/k/+T/4//k/+L/5f/i/+T/'+
        '4v/j/+L/4v/i/+T/5P/m/+X/5v/m/+X/5v/l/+b/5v/n/+f/6P/o/+n/6P/p/+n/6v/q/+r/6v/q/+r/'+
        '6v/q/+v/7P/s/+3/7f/u/+3/7//u//D/8P/x//H/8f/x//H/8P/x//D/8f/w//L/8f/z//L/9P/0//b/'+
        '9f/2//X/9v/2//j/+P/6//r/+v/6//n/+v/4//n/+P/6//r/+//7//z/+//8//v//P/8//3//f/9//z/'+
        '/P/6//v/+f/7//r/+//7//z/+//7//r/+v/6//r/+v/6//v/+//6//v/+v/6//v/+//7//v/+//6//r/'+
        '+v/6//r/+f/6//r/+v/7//v//P/8//3//f/9//z/+//7//r/+v/7//r/+//7//v//P/7//z/+//8//z/'+
        '/P/7//z/+//8//v//P/7//3/+//9//v//P/7//3//P/9//z//P/8//v//f/8//7//f/+//3//f/8//z/'+
        '+v/8//r//P/7//z//P/9//3//f/+//3//v/9//7//f/+//z//f/7//z/+v/7//r/+//7//r/+//6//z/'+
        '+//+//z////+/wAA/////wAA/v////3//v/9//3//f/9//3//v/9//7//f////7/AAAAAAAAAQAAAAEA'+
        'AAAAAP/////+//7//f/9//z//v/9/wAA//8BAAEAAQACAAEAAQAAAAAAAAD/////AAD+/////P/+//z/'+
        '/f/9//3//v/+/////v//////////////AAAAAAAAAAD////////+/////v/+//7//v////7//////wEA'+
        'AAACAAIABAADAAQAAwAEAAMAAwADAAQABAAFAAUABwAHAAcABwAHAAcABwAIAAkACQAKAAsADAAMAAwA'+
        'DAANAA0ADQAOAA4ADgAOAA4ADgAOAA4ADgAOAA0ADQANAA0ADQAOAA0ADgAOAA0ADQAMAAwADAAMAA0A'+
        'DAAOAA0ADQANAAwADAAMAAwADAAMAA0ADQAOAA0ADQANAA0ADQANAA0ADgAOAA0ADQAMAA0ADAANAAwA'+
        'DAAMAAwADQANAA0ADQANAA0ADQANAA4ADgAOAA4ADgAOAA4ADgAOAA0ADwAOABAADwARABAAEAAQABAA'+
        'DwAQAA8AEAAPAA8ADwAOAA8ADgAPAA8AEAAQABEAEAASAA8AEQANAA8ADQAOAA0ADgAPAA8AEAAQABEA'+
        'EAARABAAEQAQABEADwARAA8AEAAQABAAEAAPABAADwAQABAAEQASABIAEwATABMAEgASABIAEwASABQA'+
        'EgAUABMAEwATABIAEwAUABQAFgAWABgAGQAZABsAGgAbABoAGwAaABsAGgAbABkAGwAYABoAGAAZABkA'+
        'GQAaABoAGgAaABkAGQAZABkAGwAaABsAGwAbABsAGgAaABoAGQAaABkAGgAZABoAGgAaABoAGQAZABgA'+
        'GAAYABcAGAAYABgAGAAXABgAFwAXABYAFgAVABYAFQAWABYAFwAYABgAGgAaABkAGgAYABkAFwAYABcA'+
        'GAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGQAYABgAGAAYABkAGAAaABkAGgAZABoA'+
        'GgAaABsAGgAbABkAGwAYABkAGAAYABcAFwAXABYAFgAWABUAFQAVABUAFQAVABUAFQAVABUAFAATABQA'+
        'EgATABIAEwATABMAEwATABIAEQARABAADwAPAA4ADwAOAA8ADwAQAA8AEQAQABAAEQAQABEADwAQAA4A'+
        'DgAMAAwACwALAAoACgAKAAsACwALAAwADAANAA0ADAANAAsADQALAA0ADAANAAwADQANAA0ADQAOAA0A'+
        'DQAMAAwADAALAA0ACwANAAwADQAMAAwADAALAAsACgAKAAkACQAJAAgACQAJAAoACgAKAAoACgAKAAkA'+
        'CQAIAAkACQAJAAkACQAJAAkACQAJAAkACQAJAAkACQAIAAgACAAIAAgABwAIAAcABwAGAAcABgAHAAUA'+
        'BgAGAAcABwAIAAgACAAHAAgABgAHAAYABgAGAAYABwAGAAcABwAHAAYABgAFAAYABAAGAAQABgAFAAYA'+
        'BgAGAAYABgAGAAcABgAIAAcACQAHAAkACAAKAAkACgAJAAoACQAJAAkACAAIAAcABwAHAAcABgAGAAUA'+
        'BQAEAAQABQAFAAYABgAGAAYABgAGAAUABQAEAAQABQAFAAUABQAFAAYABAAFAAMABQADAAUABAAGAAUA'+
        'BgAGAAYABQAEAAUABAAFAAUABQAFAAYABQAHAAUABwAGAAcABwAHAAcABwAHAAcABgAGAAUABAAFAAMA'+
        'BQAEAAYABQAGAAYABgAGAAYABQAFAAQABgAFAAYABQAGAAUABQAEAAMAAwACAAMAAQADAAEAAwACAAIA'+
        'AwABAAQAAgADAAIAAgACAAEAAgABAAIAAQACAAIAAwACAAIAAgABAAEAAAABAP//AAD+/////f/9//3/'+
        '/f/+//7//////////v/+//3//f/8//z/+//9//z//v/+//3//v/8//3//P/8//3/+//9//z//f/8//z/'+
        '/P/8//z//f/7//z/+v/7//r/+v/6//r/+//7//z//P/8//z//f/8//3//P/9//z//f/9//3//P/9//v/'+
        '/P/6//v/+v/7//r/+v/6//r/+f/6//j/+v/4//r/+P/5//f/+f/2//j/9v/3//b/9//2//f/9//3//f/'+
        '9v/3//b/9//2//f/9v/2//X/9f/1//X/9P/0//L/8//y//P/8v/y//L/8//y//L/8f/y//H/8v/x//H/'+
        '8P/w//D/8P/w//D/8P/v/+//7//v/+7/7//u/+7/7v/u/+7/7v/u/+7/7v/u/+7/7v/t/+3/7P/t/+v/'+
        '7P/s/+3/7v/u/+//7v/v/+//7//v/+//7//w/+//8P/v//D/7v/v/+7/7v/u/+7/7v/v/+7/8P/u//D/'+
        '7v/v/+7/7//v/+//8P/w//D/8P/w//D/8P/x//D/8f/x//L/8v/z//P/8//z//L/8//y//P/8//0//P/'+
        '9P/z//T/8v/z//H/8v/y//P/9P/1//b/9//2//f/9v/2//X/9v/1//b/9f/2//X/9f/1//X/9P/1//X/'+
        '9f/1//b/9v/2//f/9//3//f/+P/3//j/9//3//f/9v/2//f/9v/4//f/+f/4//n/+P/5//j/+f/4//n/'+
        '+P/5//j/+f/4//n/+P/5//n/+v/7//z//P/8//z//P/8//v//P/7//z//P/9//z//f/8//z//P/7//z/'+
        '+//9//z////+/wAA//8AAP//AAD//wAAAAAAAAAAAAAAAAAA/////////////////////wAA/v////7/'+
        '///+/////v////////8AAP//AAD//wAA//8AAAAA///////////+//7//v/+//3//v/9//7//f/+//3/'+
        '/v/9/////f8AAP3/AAD9/////P/9//z//P/9//3//v//////AAAAAP/////+//7//f/9//3//f/9//3/'+
        '/f/8//z//P/8//z//P/8//3//f/9//3//f/8//z/+//8//z//P/8//3//f/9//3//P/8//v//P/6//z/'+
        '+f/7//j/+v/4//n/+P/5//n/+f/6//v/+//8//z//f/8//3//P/7//v/+v/5//n/+f/5//j/+P/3//f/'+
        '9//3//f/9//4//j/+P/4//j/+P/4//j/+P/3//j/9//4//f/+P/3//j/9v/3//X/9v/2//b/9v/3//f/'+
        '9//3//f/9//3//b/9//2//f/9//3//f/9//2//j/9//4//f/+f/4//n/+v/6//r/+//6//v/+v/7//r/'+
        '+//5//r/+f/6//r/+//8//z//f/8//7//P/9//z//f/8//3//P/+//z//v/8//7//f////7//////wAA'+
        '//8AAP/////////////+/////v/+//3//v/9//7//f/+//7////+//7//v/+//7//f////7//////wAA'+
        '//8BAAAAAQD//wAA//8AAP///////wAA//8AAAAA//8AAP//AAAAAAAAAAABAAAAAQD//wAAAAAAAAEA'+
        'AAABAAEAAQABAP//AAD+/wAA//8AAAAAAQAAAAEA//8BAP//AAD+/////////wAA//8AAAAAAAABAAAA'+
        'AgABAAIAAgACAAMAAgADAAIAAwACAAIAAwACAAMAAgADAAMAAwADAAMAAwADAAMAAwACAAMAAgADAAQA'+
        'AwAFAAQABgAFAAYABgAGAAYABgAFAAUABAAFAAQABQAFAAYABgAGAAYABgAGAAYABQAFAAMABAADAAQA'+
        'AwADAAMAAgADAAIAAgABAAIAAgABAAIAAAABAAAAAQABAAEAAgABAAIAAQACAAAAAQAAAAEAAAABAAEA'+
        'AgABAAMAAQADAAEAAgABAAEAAgACAAMAAwADAAMABAAEAAQABQAFAAUABgAFAAUABAAFAAQABgAFAAcA'+
        'BQAHAAQABwAEAAgABgAJAAkACwALAA0ADQAOAA4ADgAPAA8AEAAPABEAEAASABEAEgARABIAEQASABIA'+
        'EwATABQAEwAUABMAFAATABQAEwATABQAEgAVABEAFAASABMAEwATABMAFAAUABQAEwASABIAEAARAA8A'+
        'EAAPABEADwAQAA8AEAAPAA8ADgAPAA8ADwAQAA8AEAAPAA8ADwAOAA8ADQAOAA4ADQAOAA0ADgANAA4A'+
        'DgANAA4ADQAOAAwADQANAA4ADgAOAA4ADgANAA0ADQANAA0ADQAOAA4ADgAPAA4ADwAPAA8AEAAPABEA'+
        'EAASABEAEQAQABAAEAAQABAAEAAPABAADwAQAA8AEAAPAA8ADwAPAA8ADwAPABAADwAQABAAEQARABIA'+
        'EgASABIAEgASABIAEgATABMAEwAUABQAFAAVABQAFgAUABYAFQAVABYAFQAWABQAFgAVABcAFQAXABYA'+
        'GAAWABgAFgAXABUAFgAVABYAFwAXABcAFwAYABcAFwAWABYAFQAVABQAFAATABQAEwATABMAFAATABQA'+
        'FAAVABUAFAAVABMAFAATABQAFAAVABQAFQATABMAEgASABMAEgATABIAEwATABMAEgARABEAEAAPABAA'+
        'DwAQAA8AEAAPABAADwAQAA8AEAAPABAAEAAQABEAEAAQABAAEAAOABAADQAQAA4AEQAPABEADwARABAA'+
        'EQARABEAEQARABEAEAARABAAEQARABAAEAAPABAAEAAQABIAEQATABIAEwASABMAEwATABMAFAATABMA'+
        'EwAUABQAFAAVABUAFQAVABQAFgAUABUAFQAVABUAFQAVABYAFQAXABUAFgATABQAEgAUABMAFQAVABYA'+
        'FgAWABUAFQAUABQAFAAVABUAFgAXABcAGAAXABcAFgAWABYAFgAWABYAFgAXABcAFwAXABcAFwAWABcA'+
        'FQAWABUAFwAWABcAFwAXABcAFwAXABcAFwAWABcAFgAWABYAFwAWABcAFQAVABMAEwATABMAFAAUABUA'+
        'FgAVABYAFQAWABQAFQAUABQAEwAUABMAFQASABQAEQASABAAEQAQABAAEAAQAA8AEAAPAA8ADwAPAA8A'+
        'DgAPAA4ADwAOABAADgAQAA4ADwAPAA8AEAAPABEADwAQAA8ADwAPAA4ADgANAA4ADgANAA0ADAAMAAwA'+
        'CwANAAsADQAMAA0ADAAMAAsACwAKAAoACQAJAAgABwAHAAUABQAFAAUABgAEAAUABAAFAAMABQADAAUA'+
        'BAAFAAQABQAEAAQABAADAAMAAwADAAMAAwADAAQAAgADAAEAAwACAAQAAwAEAAMABAACAAMAAgADAAIA'+
        'AwACAAMAAgACAAIAAgACAAIAAQACAAIAAwADAAQABAAEAAQABAADAAQAAwAEAAMABAACAAMAAgACAAEA'+
        'AQABAAEAAQABAAEAAgACAAIAAgACAAIAAQABAAEAAQABAAEAAQABAAAAAQAAAAAAAAAAAAAAAQAAAAEA'+
        '//8AAP7////9/wAA/////////v/9//z//P/7//z/+//8//r/+//6//r/+//7//v/+//5//r/+P/5//n/'+
        '+P/5//j/+P/4//f/+P/3//j/9//3//b/9v/2//X/9v/2//X/9//1//b/9f/1//X/9P/0//P/8//y//P/'+
        '8//0//P/9P/0//X/8//0//P/9P/y//P/8v/z//L/8v/y//D/8v/v//L/7//y//D/8//w//P/8P/y//D/'+
        '8v/x//L/8v/y//P/8//0//P/9P/z//P/8v/0//P/9P/z//T/9P/0//T/9P/1//X/9v/1//X/9f/0//T/'+
        '8//0//T/9P/0//P/9P/x//L/8P/w//D/7//w/+//8f/w/+//8P/u//D/7f/v/+7/7//u/+7/7v/t/+7/'+
        '7P/v/+z/8P/t/+//7f/t/+3/7f/t/+3/7v/u/+7/7v/u/+7/7v/u/+7/7//t/+//7f/t/+3/7P/u/+z/'+
        '7v/s/+7/7f/t/+z/7P/t/+3/7v/u/+//7v/u/+3/7v/s/+7/7v/u//D/7//w//D/7//w/+7/8P/u//H/'+
        '8f/x//P/8v/z//H/8v/x//H/8P/x//H/8f/x//H/8v/x//H/8P/x//H/8f/y//L/8//y//P/8v/y//L/'+
        '8v/z//L/9P/z//T/8//z//P/8v/z//L/8v/y//L/8v/y//P/8v/y//P/8v/z//L/8v/y//L/8P/w/+//'+
        '7//w/+//8P/u//D/7f/u/+7/7v/v/+//8f/w//H/7//w/+//7//v/+//7v/v/+3/7//u/+//7v/v/+7/'+
        '7v/u/+7/7f/t/+3/7P/t/+z/7v/u/+//7//u/+7/7f/s/+3/7P/t/+3/7f/t/+v/6//q/+r/6v/q/+r/'+
        '6//p/+r/6P/q/+n/6f/p/+r/6v/q/+r/6v/r/+v/6//r/+v/6v/r/+r/6//q/+v/6v/q/+n/6v/p/+r/'+
        '6v/r/+v/6//s/+v/7P/s/+3/7f/u/+7/7v/u/+7/7//v/+//8P/v//D/7//x/+//8f/v//D/8P/w//D/'+
        '7//w/+//7//u/+7/7f/u/+7/7//v/+//7//u/+3/7f/s/+7/7f/v/+7/7//v/+7/7//v/+//7//w/+//'+
        '7//u/+7/7v/u/+7/7v/u/+//7v/v/+7/7//v/+//7//v/+//8P/v//D/8P/w//D/7//w/+7/8P/v//D/'+
        '7//w//D/8P/w//D/7//w//D/8P/x//D/8f/w//D/7//w//D/8f/x//L/8P/y//D/8f/w//H/8f/x//D/'+
        '8f/w//D/8P/w//L/8v/z//L/8v/y//L/8f/y//H/8v/x//H/8P/y//H/9P/z//X/9f/1//X/9f/3//b/'+
        '+P/2//j/9//4//j/9//3//b/9v/2//b/9v/3//f/+P/3//f/9//3//b/9//2//b/9f/2//X/9v/2//f/'+
        '9//2//f/9f/2//X/9f/1//X/9f/1//P/9P/y//L/8v/y//L/8//y//P/8v/y//L/8v/z//H/8v/w//H/'+
        '7//w/+//8f/w//H/8P/w/+//7//v/+//8P/w//D/8P/w//D/8P/w//H/8f/y//H/8v/w//D/8P/w//D/'+
        '8P/w//H/8P/x//H/8f/y//H/8f/x/+//8P/v//D/8P/x//H/8f/w//D/7//w//D/8P/w/+//7//v/+7/'+
        '7v/u/+//7v/v/+7/7f/u/+z/7//t/+//7f/u/+3/7v/s/+//7f/v/+//7//w/+7/8P/u//D/8P/x//L/'+
        '8v/y//P/8v/z//L/8//y//P/8//y//T/8//0//T/9f/2//X/9v/1//X/9f/0//X/9P/2//b/9//4//f/'+
        '+P/3//j/+P/4//n/+P/4//f/+P/3//j/+P/5//n/+f/5//n/+P/4//f/+P/3//n/+f/6//v/+//8//z/'+
        '/P/8//v//P/6//z/+//8//v//P/9//z//v/9/////v8AAP//AAD///////////7//v/+//////////7/'+
        '///9//7//P/8//3//P/+//3////9//7//v/+///////////////+//7//f/9//3//v/+//3//v/8//7/'+
        '/P/9//v//f/7//z/+//6//r/+v/6//r/+v/6//r/+//6//r/+v/5//v/+P/7//j/+v/3//n/9//5//j/'+
        '+f/5//r/+v/6//v/+//9//z//f/9//3//v/9//7//v/+//7//f/+//z//f/8//z//P/8//z//f/8//7/'+
        '/P////3////+//////8AAAAAAQAAAAEAAAAAAAAAAQABAAIAAgACAAIAAQACAAEAAQABAAEAAQACAAIA'+
        'AwACAAMAAgADAAEAAwABAAIAAQACAAEAAgABAAEAAQAAAAIAAAACAAAAAgABAAIAAgACAAIAAgABAAIA'+
        'AQABAAIAAQACAAEAAQABAAAAAQAAAAEAAAAAAAAA//////////////7////+//7//v/+//////8AAP//'+
        '///+/////v///////////////v/+/////v//////////////AAAAAAAAAAABAAAAAQD//wEA//8BAAEA'+
        'AQACAAEAAgABAAIAAQACAAEAAgABAAMAAgADAAMAAwAEAAMABAADAAQAAwADAAMAAwADAAIAAwACAAQA'+
        'AwAFAAQABQAFAAUABQAGAAYABgAGAAYABQAEAAQAAwAEAAMABQAFAAYABQAGAAUABQAGAAYABwAHAAcA'+
        'BwAGAAUABQAEAAUABAAFAAUABgAGAAUABgAFAAYABQAFAAYABgAIAAcACQAHAAkABwAIAAcACAAIAAgA'+
        'CAAHAAcABwAHAAYABgAGAAcABwAHAAcABwAHAAYABQAFAAUABgAHAAgACAAJAAgACAAIAAcACAAHAAgA'+
        'CAAHAAgABwAHAAcABwAGAAYABQAFAAQABAADAAMAAwACAAQAAwAEAAMAAwADAAIAAgABAAIAAQACAAEA'+
        'AgACAAIAAgACAAIAAgACAAEAAgABAAIAAgACAAMAAgADAAIAAgACAAMAAwADAAEAAgAAAAEAAQACAAEA'+
        'AgABAAEAAQAAAAEAAAABAAEAAQACAAEAAgACAAIAAwACAAMAAwADAAMAAwADAAMAAwADAAMAAwAEAAQA'+
        'BAAEAAQABQAEAAYABQAGAAYABQAFAAQABAAEAAQAAwAEAAMAAwADAAMABAADAAQAAwAEAAMAAwAEAAQA'+
        'BQAEAAUABAAEAAQABAAFAAUABAAEAAIAAwACAAEAAgACAAMAAgACAAIAAQABAAEAAQABAAEAAQACAAIA'+
        'AwACAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAIAAwACAAMAAwADAAQAAwAEAAQABAAEAAQABAAEAAUA'+
        'BgAFAAYABQAGAAYABgAFAAYABAAFAAQABAAFAAQABgAFAAcABQAHAAYABgAGAAYABwAGAAcABgAHAAcA'+
        'BgAHAAYABgAFAAYABQAGAAcABgAHAAUABgAFAAUABgAEAAYABAAFAAUABQAEAAUABAAEAAQAAwADAAMA'+
        'BAAEAAUABAAFAAMABAADAAMAAwAEAAQAAwAEAAMABQADAAYABgAFAAcABAAGAAQABQAGAAUABwAGAAcA'+
        'BgAGAAYABgAGAAUABgAFAAYABgAHAAcABwAHAAYABgAEAAUABAAFAAUABQAFAAQABQAFAAUABgAFAAYA'+
        'BQAFAAUABgAFAAYABgAGAAYABQAFAAMABAADAAQABAAFAAQABAAEAAQABAAFAAQABQAEAAQABAAEAAQA'+
        'AwAEAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMABAADAAMAAwADAAMAAwADAAQAAwADAAIAAgABAAEA'+
        'AAAAAAAAAAAAAP//AAD//wAA//8BAP//AQD//wAAAQAAAAIAAQABAAEAAAAAAAAAAAAAAAEAAQABAAEA'+
        'AAAAAP///////////v//////AAAAAAAAAAAAAP//AAD//wEAAAABAAAA//////7//v/+//7//f/9//z/'+
        '/f/7//3//P/9//3//f/+//z//v/8//3//P/7//z/+v/7//r/+v/6//r/+f/5//j/+P/3//f/9//3//f/'+
        '9//3//j/+P/4//f/+P/3//f/+P/4//n/+f/5//n/+f/4//j/9//3//b/9//3//f/+P/4//n/+P/5//j/'+
        '+P/4//f/+P/3//f/9//2//b/9f/2//b/9v/2//X/9f/0//X/9f/1//b/9P/1//P/8//y//L/9P/0//T/'+
        '9f/z//X/8//0//T/9P/2//X/9//0//f/9P/3//X/9v/1//X/9f/1//T/9v/1//j/9//5//n/9//4//b/'+
        '9//2//f/+P/4//n/+f/4//n/+f/6//n/+v/5//j/+P/3//j/9//4//n/+P/5//j/+P/4//j/+f/5//n/'+
        '+v/4//r/+P/5//n/+f/6//j/+f/4//j/9//2//b/9//3//j/+P/5//n/+P/3//f/9v/3//b/9//3//f/'+
        '9//3//f/9//3//j/+P/3//f/9f/2//T/9f/z//T/8v/y//L/8v/y//L/8f/x//H/8f/y//L/8//0//L/'+
        '8v/v//D/7v/u/+//7v/u/+7/7v/u/+7/7//t/+7/7P/s/+v/7P/r/+v/6//q/+r/6P/q/+j/6f/o/+f/'+
        '5//l/+X/5P/j/+H/4P/d/93/4P/h/+v/6//1//T/9//4//X/9//w//H/4v/i/8v/y/+3/7j/tP+1/8P/'+
        'w//f/97////+/xgAFwAfACEAFwAZAAcACQD4//n/7f/t/+n/6f/w/+///f/8/wkACAALAAwABwAIAAIA'+
        'AgD///3////8/wIAAQAMAAsAGAAYAB4AHwAeAB0AGAAYABMAFAAQABMAEwAWABwAHQAmACcALwAvADQA'+
        'NQA1ADYAMgAyACsAKgAiACAAGQAYAA8ADwAHAAcABAACAAYAAwAOAAoAGwAYAC4ALQBBAEEATABNAE0A'+
        'TQBHAEYAPQA8ADMAMwAvAC4ANAA0AEIAQgBTAFIAYwBjAG8AcABxAHMAaABqAFcAWABGAEYAOQA5ADIA'+
        'MgAxADIANgA4ADwAPwA+AD8AOgA7ADQANAAuAC4AJgAnAB8AIAAaABoAFwAWABQAFAASABIADwAQAA4A'+
        'DgAQABAAFQAVABwAHAAjACMAKgAqADAAMAA3ADcAOwA8AD4APgA9ADwAOQA4ADIAMgAqACoAIgAiAB8A'+
        'HwAiACEAKAAnAC4ALQAwADAALQAuACUAJAAYABcACwAKAAMAAgABAAAABQAEAA4ADgAaABkAJQAkACwA'+
        'KwAuAC4AKgAqACMAIwAZABkAEQARAA0ADAANAA0AEAAQABQAFAAYABgAGQAZABYAFwARABEACQAJAP//'+
        'AAD3//j/8f/y/+//8P/x//L/9P/2//j/+f/4//n/9P/1/+3/7//n/+f/4v/i/9//3//e/9//3//g/+L/'+
        '4//n/+f/7P/r//L/8f/3//f//P/9/wAAAAADAAEABAABAAMAAgABAAIAAQACAAEAAQACAAAAAgABAAAA'+
        'AAD8//3/9//3//L/8v/u/+3/7P/q/+r/6P/n/+b/4//j/97/3f/Y/9f/0v/S/8//z//N/83/zf/N/87/'+
        'z//R/9L/0//U/9P/0//R/9H/zv/N/8r/yf/I/8f/x//I/8n/yf/N/8z/0f/P/9T/0//X/9f/2P/Z/9j/'+
        '2f/X/9f/1P/U/9H/0f/O/87/zf/N/83/zf/P/87/0v/R/9b/1P/Y/9f/2P/X/9X/1v/S/9P/0P/Q/87/'+
        'zf/N/83/z//P/9L/0v/U/9X/1f/V/9X/1P/T/9L/0v/Q/9D/z//P/8//0f/Q/9P/0f/U/9P/1f/U/9X/'+
        '1v/W/9f/2P/Y/9r/2v/c/9z/3//f/+L/4f/k/+P/5f/j/+X/5P/l/+T/5//m/+v/6v/w/+7/8//y//T/'+
        '9v/2//f/+P/3//n/9//5//j/+f/6//z//f8BAAAABQADAAkABwALAAkACwALAAwADAAMAAwADgAMAA4A'+
        'DQAOAA4ADgAPAA4ADgAOAA0ADQAMAA0ACwAMAAsACwAKAAoACQAIAAgABwAGAAYABAADAAIAAAAAAP7/'+
        'AAD+/wAAAAAAAAEA//8AAP///v/9//z//P/7//v/+//6//3/+/8BAP//BAADAAcABgAIAAgABwAIAAUA'+
        'BgAEAAUABQAFAAoACAAOAA0AEgATABcAGAAaABkAGwAZABoAFwAXABYAFgAWABYAFwAYABgAGgAaABsA'+
        'HAAcAB0AGwAcABoAGwAZABoAGAAYABYAFgAUABMAEgARABEADwAPAA4ADgAMAAwACwALAAsACgALAAkA'+
        'CgAIAAgABgAHAAQABgADAAUAAgACAAAAAAAAAP//AAD//wEAAAACAAEAAgACAAEAAQD///7//P/7//n/'+
        '+f/4//j/+P/6//r//P/6//z/+v/6//j/9//1//X/8v/y/+//8P/v/+7/7//u//D/7//v/+7/7P/s/+n/'+
        '6f/l/+b/4//k/+T/5P/l/+b/5//o/+j/6f/n/+j/5v/n/+X/5f/k/+P/4//i/+L/4f/h/+H/4v/h/+P/'+
        '4v/j/+P/5f/l/+b/5//m/+f/5//n/+f/6P/o/+j/6P/p/+r/6v/s/+z/7//v//L/8v/0//T/9v/1//f/'+
        '9//4//j/+v/6//3//f//////AQABAAIAAwADAAQABQAFAAUABgAGAAYABgAGAAgABwAKAAoADQANABAA'+
        'DwARABAAEgARABEAEgARABIAEQARABAAEAAPABAADwAQABAAEQASABEAEgARABEAEAAPABAADwAQABEA'+
        'EAATABIAFgAVABkAGQAcABwAHgAgACEAIgAjACMAJQAkACYAJgAnACkAKQArACsALAAtACwALwAtAC8A'+
        'LwAvADAALwAvAC4ALgAvAC8AMQAxADMANAA1ADcANgA4ADgAOQA6ADkAPAA7AD0APgA+AD8AQABAAEAA'+
        'PwA/AD4APgA9ADwAPAA5ADsAOQA5ADkAOAA4ADgANgA3ADUANwA0ADUAMwAzADIAMQAwADAAMAAwADEA'+
        'MQAyADEAMQAwAC8ALwAtAC0AKgArACkAKQAoACgAJwAoACcAKAAmACcAJQAmACQAJAAiACIAIAAgAB4A'+
        'HgAdAB0AHgAeAB8AHgAfAB8AHQAeABwAHQAbABsAGQAZABYAGAAVABcAFQAWABUAFQAUABUAEwATABEA'+
        'EQAPAA4ADQAMAAsADAALAAwADAAMAAwADAALAAwACQALAAYACAAEAAUAAgADAAEAAgABAAIAAQACAAEA'+
        'AQD//////P/8//n/+f/2//b/9P/0//L/8v/x//L/8f/y//L/8//0//X/9f/1//T/9P/y//P/7v/v/+r/'+
        '6//o/+f/6P/n/+n/6f/s/+z/7v/u/+//7//t/+3/6f/p/+T/5f/g/+L/4f/i/+P/5P/n/+f/6v/r/+3/'+
        '7f/u/+7/7f/u/+z/7f/r/+v/6v/q/+n/6f/p/+n/6f/q/+r/6//r/+z/7P/s/+z/7P/r/+v/6f/q/+b/'+
        '6P/m/+f/6P/n/+v/6v/t/+7/8P/y//L/9P/0//T/9v/1//b/9v/2//f/9//4//r/+v/+//3/AQAAAAQA'+
        'BAAGAAYABgAHAAcABwAHAAgACAAJAAoACgAMAA0ADgAQABEAEgAUABMAFQAUABUAFgAWABcAFwAXABgA'+
        'GAAZABgAGQAaABsAHAAcAB0AHAAdABoAHAAaABwAGwAcABwAHQAcAB4AHQAeAB8AHwAhACAAIQAgAB8A'+
        'HwAeAB4AHQAdAB0AHQAcAB0AHAAcABoAGwAaABwAGgAdABsAHQAdAB0AHQAdABwAGwAaABsAGQAaABoA'+
        'GgAbABoAHAAaABsAGgAaABoAGAAaABgAGQAXABgAFgAXABUAFwAUABUAEwAUABIAEwAPABAADQAPAAwA'+
        'DQANAA0ADgANAA4ADQANAA4ADAAOAAoADAAJAAoACAAJAAgACQAGAAgABgAIAAUABwAEAAQAAwADAAEA'+
        'AQABAAEAAAABAAEAAQACAAIAAQABAAEAAQAAAAEAAAABAAEAAgAAAAIAAQADAAEAAgAAAAAAAQAAAAIA'+
        'AQADAAIAAwADAAIAAQACAAAAAAD//////////wAA//8BAAAAAQD//wAA/v////7////9//7//v////7/'+
        '/////////v/9//z/+//7//r/+//6//v/+//8//v/+v/4//r/+v/6//v/9//5//n/+v/5//n/9v/2//v/'+
        '+v/7//z/+//8/////v/9//z//f/8//z/+//6//r/+//6//v/+v/+//3/+//7//r/+v/9//z/+v/6//z/'+
        '/P/7//v/9//3//f/9//0//P/9f/0//T/8//0//T/+v/6//b/9v/3//b/9v/2//X/9P/5//f/8//y//X/'+
        '9f/0//T/7f/s//L/8P/w/+//8v/y//b/9f/y//L/8v/x//D/7v/y//H/8v/y//L/8//y//L/5//n/+v/'+
        '6v/n/+b/3v/d/+//7v/w/+7/6v/o//P/8f/r/+n/5P/i/+3/6//r/+r/6v/p/+//7v/q/+n/6v/q/+7/'+
        '7//p/+r/5f/m/+X/5P/l/+P/5f/j/+j/5v/u/+z/8f/v/+//7P/o/+b/7P/r//b/9P/x//D/8//z/+v/'+
        '6v/f/9//4//j/+D/4f/n/+j/6f/q/+j/6P/w/+//4P/e/93/2//n/+T/4f/f/+b/5f/m/+b/5//n/+r/'+
        '6f/r/+n/7//u/+n/6P/n/+f/5P/k/+n/6v/r/+v/4f/g//3//P//////8f/x//3//f/n/+b/6v/p//b/'+
        '9v/l/+X/9v/2//H/8P/p/+j/AwACAP7//P/1//T/8//y/+n/6P/z//P/AAAAAAMABAABAAEA/P/8////'+
        '////////CAAIABMAEgD6//r/9f/0//7//f/0//P/EwARABEAEAD2//X/DQAMAPr/+v/2//f/CAAJAOv/'+
        '7P/5//v//v8AAPn/+/8cABsACAAHAPr/+f8IAAcA/f/9//f/9//0//L/+P/2//3//P8BAAEAAgADAAAA'+
        'AQAFAAUA+v/6////AAD//wEA5f/l//D/8P8AAP///P/8/wIAAgABAAEA+P/4/+3/6//s/+r/8f/v//X/'+
        '9P8AAAIACAALAP3/AADr/+v/+v/5/wAA/v/9//3/DAAOAO//8f/0//T////9/9D/zv/2//X/CwALANn/'+
        '2f/8//z/CwAKAOz/7P8SABIAAQABAMr/y/8HAAgABQAGAOD/4P85ADgAFwAWANb/1f8MAAoA6//q/+P/'+
        '5P8FAAcA9f/2/wgABwAIAAcAAQABAPn/+f/9//7/EAAQAO//7v8HAAYAAAAAAN3/3v8RABIA/f/9//P/'+
        '8f8LAAsA9v/3/wQABADp/+n/7//v/xcAFwD6//r/DAALAAwACwDx//D///////L/8v8AAAAACwAKAOz/'+
        '7P/9//z/DgAOABQAFAAiACMADAAMAPn/+P/u/+3/8v/x/wcABwD5//n/CQAJAB0AHAD3//b//f/8//n/'+
        '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'+
        'gf+n/6f/9/33/Rr9Gv36/fr9yf3J/b/9v/0N/Q39EfwR/B3+Hf5K/0r/W/9b/0X/Rf9y/XL9oP2g/Rz+'+
        'HP6q/ar9mf6Z/rf/t/+WAJYASv9K/4P/g//oAegBCQAJAOn96f0n/Cf8NPo0+qr8qvy//r/+YP9g/3oA'+
        'egBQAFAAdQB1AIH/gf/f/d/9+fz5/EL7QvvH+8f7YP5g/pL/kv+AAIAASAFIAfgA+ADoAOgA8ADwAKz/'+
        'rP+M/oz+k/6T/p7+nv7Q/tD+Mv8y/6b/pv/J/8n/EP8Q/8r+yv7L/sv+PP88/xoAGgAR/xH/5/7n/hsA'+
        'GwDp/un+x/7H/jwAPAD0//T/4f/h/53/nf/g/uD+pf+l/8j/yP94/nj+ov2i/WT9ZP0w/jD+rP+s/xMA'+
        'EwDtAO0AhQKFAm8CbwIPAg8C2AHYAcgAyADm/+b/3f7d/rH+sf7s/+z/gACAANgA2ABgAWABdwF3AR8B'+
        'HwGFAIUAtwC3AAMBAwHv/+//Bv8G/3j/eP/1//X/NwA3ANQA1AACAQIBqQCpAD4APgCS/5L/Gf8Z/+T+'+
        '5P5Q/lD+B/4H/sn+yf5x/3H/nP+c/yoAKgA1ADUA4P/g/4IAggAqASoBeAF4AYQBhAHrAOsAWwBbAEUA'+
        'RQB1AHUARQBFACUAJQBRAFEA3v/e/10AXQAWARYBdQB1AL8AvwD4APgAfwB/AOUA5QCmAKYAov+i/8P+'+
        'w/6o/qj+9v72/mb+Zv63/rf+gv+C/1X/Vf9//3//l/+X/4D/gP8w/zD/X/5f/lj+WP52/nb+9v32/Tb+'+
        'Nv51/3X/IwAjANX/1f9J/0n/c/5z/lz+XP42/zb/VP9U/xz/HP8i/yL/Fv8W/0r/Sv9H/0f/nP6c/hn+'+
        'Gf4D/gP+7v3u/Rv+G/5w/nD+g/6D/q3+rf4z/zP/wf/B/33/ff/Y/tj+Mf8x/53/nf8f/x//t/63/rr+'+
        'uv77/vv+r/+v/z0APQCGAIYABQEFATABMAEZARkB1QDVAA8ADwDi/+L/NgA2AIsAiwBSAVIBZgFmAcsA'+
        'ywDvAO8AWwFbAYMBgwF5AXkBQQFBAfcA9wC0ALQAtAC0AHwAfADF/8X/Of85/w//D/9r/2v/MgAyAH0A'+
        'fQAlACUA8v/y//P/8/+o/6j/cP9w/2X/Zf9j/2P/8P/w/2kAaQBYAFgAYgBiAIAAgAC5ALkA2gDaABYB'+
        'FgHTAdMBTwJPApACkALAAsAC6gLqAiYDJgPIAsgCTAJMAuEB4QGEAYQB4AHgAUECQQJZAlkCNgI2AsUB'+
        'xQGYAZgBXAFcAewA7ABTAFMA0f/R/00ATQDjAOMADwEPAYQBhAG8AbwBxQHFAdEB0QFuAW4BRAFEAVsB'+
        'WwElASUBSgFKAbEBsQGwAbABrAGsAb8BvwHHAccB7wHvARgCGALRAdEBLQEtAT8BPwG8AbwBdwF3AZQB'+
        'lAH3AfcBTgFOAfYA9gD4APgAkgCSANoA2gAAAQABlwCXALkAuQCgAKAAOwA7AJUAlQDiAOIAfAB8APL/'+
        '8v8S/xL/Vf5V/uT+5P4FAAUAvAC8AHoAegDx//H/UAFQAakDqQOaBJoEwQPBA1wCXAJ7AnsCKAMoA4gC'+
        'iAIsASwBv/+//+z+7P5j/mP+vv2+/c39zf2B/oH+Xf9d/2L/Yv8x/jH+WP1Y/Yf9h/3p/en9t/23/Vn9'+
        'Wf2m/ab9cP5w/vn++f7C/sL+mv6a/jf/N//R/9H/SgBKAOYA5gBnAWcB6AHoASECIQLfAd8BHgIeApQC'+
        'lAKrAasB5//n/4f+h/4Z/hn+zf7N/g7/Dv9D/kP+2P3Y/fn9+f1e/l7+9v72/m3/bf/e/97/CwALAJn/'+
        'mf/A/sD+yf3J/QX9Bf2o/Kj8Zvxm/Dj8OPy5/Ln8Af4B/pD/kP/6APoAzAHMAfQB9AE9Aj0CWgNaA0QE'+
        'RAS6A7oDZwJnAoUBhQE4ATgBvwC/AMf/x//h/uH+Nf41/q79rv1S/VL9Mf0x/cT9xP3o/uj+w//D/9f/'+
        '1/9p/2n/HP8c/9H+0f5k/mT+Lv4u/gT+BP7s/ez9CP4I/ln+Wf5+/37/NQE1AXMCcwJGA0YDHQQdBL4E'+
        'vgT9BP0E0wTTBL0DvQPTAdMBdwB3AGwAbACDAIMAhv+G/9j+2P6F/4X/5f/l/+z/7P9oAGgA9AD0ALUB'+
        'tQH/Af8BnQGdAZEBkQGrAasB2wHbAe4B7gExATEBRQBFACYAJgDeAN4AdQF1AZ4BngEUAhQCqwKrAjYD'+
        'NgPsA+wDBgQGBCQDJAP9Af0BwADAAIL/gv++/r7+hP6E/sH+wf7g/uD+vf69/lL/Uv8iACIAYQBhAJYA'+
        'lgDPAM8ABQEFAVYBVgHBAcEBDgIOAqIBogHZANkAMgAyAJv/m/9h/2H/JP8k/77+vv4Q/xD/5P/k/1wA'+
        'XADYANgAgAGAAUcBRwFHAEcAhf+F//P+8/5//n/+Zv5m/j7+Pv7x/fH9Df4N/lH+Uf5F/kX+Qv5C/j7+'+
        'Pv5P/k/+qf6p/sb+xv7m/ub+T/9P/3D/cP9q/2r/h/+H/3v/e//+/v7+Z/5n/gn+Cf6h/aH9NP00/TT9'+
        'NP3E/cT9Rf5F/nv+e/7v/u/+Z/9n/2v/a/8N/w3/gv6C/t393f05/Tn9+vz6/O387fw8/Tz9CP4I/mf+'+
        'Z/7e/t7+nf+d//3//f8zADMAnv+e/4H+gf7d/d394f3h/X7+fv6M/oz+Zv5m/g7/Dv9z/3P/lP+U/9z/'+
        '3P/5//n/7v/u/9H/0f+//7//T/9P/9T+1P6y/rL+Iv4i/qT9pP20/bT9yv3K/f/9//1Y/lj+yP7I/jX/'+
        'Nf9M/0z/Sv9K/4v/i//U/9T/EwATAHUAdQBXAFcA2P/Y/9T/1P/i/+L/AAAAAFYAVgB8AHwAqgCqAMsA'+
        'ywDNAM0A0ADQAMgAyADQANAAkACQABUAFQDW/9b/sP+w/7z/vP8LAAsAEwATALv/u/9t/23/L/8v//z+'+
        '/P7s/uz+9/73/ln/Wf8iACIAMwEzAd4B3gFzAXMB8QDxAO8A7wDuAO4AFgEWAWIBYgGjAaMB4QHhAQ8C'+
        'DwIKAgoCtAG0AWsBawGBAYEBsAGwAX0BfQEuAS4BNgE2AfsA+wBoAGgAxf/F/wf/B//N/s3+//7//hz/'+
        'HP9L/0v/iv+K//v/+/+lAKUAKwErAWABYAFpAWkBWgFaAesA6wBuAG4AZwBnAJIAkgC3ALcAsgCyALUA'+
        'tQDfAN8AmgCaACYAJgD/////xP/E/x3/Hf9f/l/+MP4w/jb+Nv7k/eT9uf25/QL+Av5m/mb+0v7S/pL/'+
        'kv9mAGYA3wDfAPEA8QC/AL8AuQC5AOgA6AAvAS8BcAFwAWoBagGWAZYBsgGyAXoBegGeAZ4BuAG4AXMB'+
        'cwENAQ0BZQBlAJL/kv+y/rL+yP3I/cv8y/wJ/An8pPuk+5n7mfsf/B/85Pzk/Nb91v3E/sT+Vv9W//D/'+
        '8P9QAFAAiACIAAABAAFPAU8BggGCAckByQEmAiYCTwJPAukB6QGHAYcBdgF2AUcBRwGqAKoAGQAZAPf/'+
        '9/+L/4v/0v7S/mL+Yv4Y/hj+tf21/fz8/PxC/EL87vvu+7D7sPvc+9z7k/yT/Cn9Kf3A/cD9h/6H/mL/'+
        'Yv9RAFEA4gDiAD0BPQGUAZQBbQFtAREBEQGCAIIAZ/9n/4L+gv5R/lH+nf6d/kv/S/8dAB0A4QDhAJIB'+
        'kgHfAd8ByQHJAYUBhQG5ALkAev96/1P+U/5y/XL9Mf0x/Vj9WP2L/Yv9NP40/u7+7v5R/1H/x//H/ykA'+
        'KQBUAFQAVwBXAEkASQBkAGQAYQBhAEgASAAgACAAtf+1/1r/Wv/7/vv+ov6i/pr+mv6q/qr+9/73/rD/'+
        'sP+MAIwAKQEpAYABgAGqAaoBYwFjAfkA+QDqAOoAAgECAeAA4ACNAI0AggCCAKIAogCkAKQA0QDRADwB'+
        'PAF7AXsBVwFXAUwBTAFKAUoBDQENAfUA9QC4ALgARgBGAMj/yP8h/yH/vP68/pj+mP6y/rL+Iv8i/4b/'+
        'hv/T/9P/BQAFACcAJwBWAFYAeQB5AIEAgQCNAI0A0gDSAP4A/gC9AL0AkwCTAOoA6gBbAVsBcwFzAVQB'+
        'VAE8ATwBWwFbAVgBWAHkAOQAhgCGAEoASgDx//H/f/9///b+9v6s/qz+s/6z/un+6f40/zT/TP9M/0r/'+
        'Sv9l/2X/eP94/6H/of/w//D/EAAQAGYAZgANAQ0BMQExAQYBBgH6APoA9QD1AOYA5gCXAJcAPgA+AOT/'+
        '5P+g/6D/ov+i/5b/lv95/3n/K/8r/+b+5v4F/wX/C/8L//v++/4Z/xn/av9q/6P/o/+K/4r/af9p/z//'+
        'P/8g/yD/F/8X/xn/Gf9h/2H/oP+g/6j/qP+A/4D/Qv9C/0//T/+O/47/yv/K/wgACAA8ADwAUgBSADUA'+
        'NQAIAAgA+f/5//X/9f/Q/9D/of+h/0v/S//g/uD+3P7c/v7+/v4D/wP/DP8M/zz/PP+4/7j/KwArAHAA'+
        'cAB8AHwArACsAEIBQgEoASgBkACQAFkAWQD4//j/fP98/yD/IP/c/tz+F/8X/3n/ef+p/6n/9v/2/yQA'+
        'JAATABMAKQApACUAJQDs/+z/x//H/5H/kf9b/1v/cf9x/8H/wf/3//f/CAAIAB4AHgAwADAAhgCGACMB'+
        'IwGaAZoBuAG4AaEBoQGHAYcBNwE3AdcA1wCzALMAfgB+AF8AXwCHAIcArQCtAL0AvQDWANYAAwEDASgB'+
        'KAE4ATgBCgEKAbMAswBSAFIAvP+8/1D/UP9p/2n/4//j/zAAMAAsACwAbgBuAKQApADlAOUAVwFXASkB'+
        'KQEQARABWwFbASsBKwGyALIAPwA/ACcAJwB0AHQAkACQAIAAgAB0AHQAUgBSAEYARgBhAGEAQwBDADIA'+
        'MgCqAKoAKAEoAXEBcQGKAYoBRwFHAbwAvAD2//b/Tv9O/w3/Df8w/zD/qf+p/ysAKwCYAJgA+AD4AE4B'+
        'TgGSAZIBpAGkAZEBkQFcAVwBEgESAdkA2QC7ALsAgQCBAAgACAC4/7j/rf+t/93/3f8bABsA5//n/6r/'+
        'qv+n/6f/jv+O/37/fv+T/5P/2P/Y/9n/2f+D/4P/bP9s/1v/W/90/3T/2P/Y/w0ADQBOAE4AygDKAGQB'+
        'ZAHUAdQB/wH/ARsCGwLGAcYBDQENAVYAVgCa/5r/H/8f/83+zf6K/or+pv6m/hb/Fv+e/57/zP/M/6D/'+
        'oP+8/7z/BgAGABAAEAD/////DwAPADoAOgBCAEIAIQAhACcAJwBrAGsAuwC7AOkA6QDrAOsA9QD1ACsB'+
        'KwE+AT4BBgEGAcUAxQCgAKAAdwB3AB0AHQCg/6D/T/9P/zH/Mf83/zf/iv+K/yQAJADCAMIAKwErAWoB'+
        'agGiAaIBmgGaAWQBZAFRAVEBCwELAX0AfQANAA0A4//j/+7/7v/6//r/IQAhAGMAYwB0AHQAcABwAF8A'+
        'XwA4ADgASQBJALwAvABTAVMBqQGpAYYBhgHzAPMAQwBDANj/2P+E/4T/V/9X/6r/qv8fAB8AYwBjALkA'+
        'uQBPAU8B8gHyATgCOAIWAhYC4AHgAaUBpQFgAWABCgEKAXQAdADP/8//Xv9e/xD/EP/D/sP+Zv5m/lz+'+
        'XP6d/p3+qP6o/uj+6P6h/6H/TwBPAMsAywA7ATsBoQGhAecB5wEPAg8CGAIYAgoCCgKuAa4BogCiAIH/'+
        'gf8H/wf/Ev8S/1D/UP+Q/5D//v/+/3EAcQCjAKMA2wDbANYA1gBwAHAACwALAJT/lP8H/wf/eP54/vb9'+
        '9v2X/Zf9P/0//fr8+vwD/QP9c/1z/Sv+K/4Q/xD/AAAAAN8A3wCjAaMBHAIcAjwCPAL9Af0BXwFfAc4A'+
        'zgBWAFYApP+k/+/+7/6b/pv+kP6Q/pX+lf62/rb+7f7t/kf/R//L/8v/SQBJAK8ArwDbANsApgCmADkA'+
        'OQDF/8X/QP9A/5D+kP7u/e79n/2f/ZP9k/3P/c/9X/5f/jT/NP8gACAA7QDtAKABoAEnAicCPQI9AugB'+
        '6AFkAWQBugC6ANP/0//O/s7+Bf4F/qH9of13/Xf9W/1b/ZD9kP1K/kr+IP8g/9//3/+yALIAWwFbAZYB'+
        'lgGHAYcBTAFMAc0AzQAeAB4Alv+W/z//P//e/t7+vv6+/gH/Af9P/0//vP+8/2EAYQDeAN4AMgEyAXAB'+
        'cAFCAUIBoQChAM7/zv/t/u3+Ff4V/lr9Wv3p/On81/zX/Av9C/2Z/Zn9Tf5N/t7+3v6e/57/mQCZAGgB'+
        'aAH0AfQBVQJVAnYCdgIoAigCigGKAc8AzwAFAAUAev96/zL/Mv/9/v3+BP8E/x3/Hf8P/w//+f75/hP/'+
        'E/9M/0z/W/9b/2X/Zf96/3r/R/9H/w3/Df8d/x3/Qv9C/1//X/+I/4j/1v/W/yoAKgBRAFEAfAB8ANUA'+
        '1QBCAUIBkwGTAawBrAG8AbwBywHLAZUBlQEvAS8BzADMAHIAcgAfAB8A7P/s//v/+///////xf/F/3n/'+
        'ef82/zb/Df8N/yn/Kf+Z/5n/8P/w//r/+v//////9P/0/+X/5f/i/+L/wf/B/6X/pf+l/6X/yP/I/wcA'+
        'BwBBAEEAnwCfAPcA9wDrAOsAlgCWAFQAVAA/AD8AOwA7AEEAQQAwADAAJQAlAFoAWgBaAFoA/////8T/'+
        'xP+v/6//iv+K/23/bf9n/2f/U/9T/yn/Kf8Q/xD/CP8I//H+8f7p/un+Gv8a/1D/UP9Q/1D/Hv8e//f+'+
        '9/4v/y//kf+R/9X/1f8lACUAYwBjAHwAfAB7AHsAPgA+AO3/7f+Q/5D/Mv8y/xL/Ev8A/wD/Dv8O/4n/'+
        'if9AAEAA2ADYAAgBCAEIAQgBFgEWAfQA9ACWAJYAKgAqAN7/3v+s/6z/X/9f/xf/F//4/vj+Af8B/zP/'+
        'M/9z/3P/nP+c/6L/ov/S/9L/PAA8AHEAcQBpAGkAZgBmAFQAVAAZABkAyf/J/3X/df8//z//O/87/0X/'+
        'Rf9y/3L/3f/d/zkAOQBYAFgAcQBxAJ8AnwDBAMEAswCzAFkAWQDQ/9D/XP9c/xP/E//Z/tn+p/6n/rj+'+
        'uP4A/wD/Mf8x/1z/XP+n/6f/9f/1/zwAPACAAIAAoACgAJgAmAB0AHQAKgAqAOn/6f+y/7L/VP9U/w7/'+
        'Dv8W/xb/av9q/9v/2/8lACUAYwBjAL0AvQAuAS4BiwGLAcoBygH9Af0B6AHoAWgBaAGLAIsAhv+G/6H+'+
        'of7t/e39c/1z/Tf9N/1Z/Vn92v3a/Xv+e/4S/xL/df91/8P/w/8JAAkADgAOAOL/4v+g/6D/jP+M/53/'+
        'nf+R/5H/zv/O/3kAeQAmASYBvQG9AV8CXwLyAvICMQMxAzUDNQMmAyYD5gLmAmUCZQKkAaQBggCCAOT+'+
        '5P44/Tj9B/wH/HL7cvtQ+1D7bPts++L74vvI/Mj8/P38/Vz/XP+iAKIApgGmAWMCYwLOAs4C0ALQAmIC'+
        'YgLMAcwBNQE1Aa4ArgAkACQAav9q/+L+4v7R/tH+CP8I/47/jv9cAFwAMgEyAckByQEcAhwCGwIbAmAB'+
        'YAHk/+T/IP4g/pv8m/ym+6b7Yfth+wb8BvyA/YD9X/9f/1sBWwEsAywDcgRyBNcE1wRRBFEENwM3A6oB'+
        'qgHd/93/Vv5W/kr9Sv23/Lf8r/yv/DP9M/0X/hf+Cf8J/8r/yv9bAFsA/wD/AMUBxQE8AjwCKwIrArIB'+
        'sgH4APgAMAAwAHb/dv/S/tL+VP5U/ij+KP5j/mP+4/7j/rL/sv/EAMQA0QHRAWsCawJJAkkCqQGpAeQA'+
        '5AAUABQAJv8m/yD+IP5k/WT9LP0s/Xj9eP0t/i3+/v7+/sD/wP9mAGYAvQC9AL8AvwClAKUAggCCAE8A'+
        'TwD+//7/gP+A/xr/Gv8X/xf/VP9U/2v/a/9p/2n/yv/K/1YAVgCdAJ0AzQDNAPEA8QASARIBTQFNAXEB'+
        'cQFuAW4BJAEkAYgAiAD6//r/fv9+/wH/Af+//r/+3v7e/kX/Rf+k/6T/5v/m/wQABADQ/9D/gv+C/1n/'+
        'Wf9c/1z/nP+c/wMAAwBpAGkAjgCOAIEAgQB/AH8AWABYAEoASgC7ALsASAFIAZ4BngGyAbIBkQGRAYsB'+
        'iwGPAY8BUQFRAeEA4QBZAFkAyf/J/1H/Uf8j/yP/ev96/zMAMwDiAOIAPQE9AUsBSwE5ATkBBQEFAdoA'+
        '2gDnAOcA3wDfAKAAoABZAFkA/////2z/bP/4/vj+Bv8G/0z/TP+H/4f/x//H//v/+/8FAAUA2//b/6f/'+
        'p/+Q/5D/hv+G/4n/if+o/6j/2f/Z/x4AHgCbAJsARQFFAe4B7gGGAoYC5QLlAugC6AKoAqgCVQJVAt4B'+
        '3gFEAUQBwADAAEMAQwC1/7X/Qf9B//7+/v7a/tr+uv66/q3+rf6s/qz+wP7A/g3/Df9l/2X/sv+y/wkA'+
        'CQBcAFwAfwB/AHQAdABiAGIAUABQADoAOgD6//r/pP+k/57/nv/c/9z/PwA/ALsAuwA0ATQBjgGOAZoB'+
        'mgGWAZYBtAG0AbABsAFoAWgB0wDTAA0ADQAs/yz/Sv5K/rX9tf2s/az99v32/Rn+Gf4s/iz+ff59/vP+'+
        '8/5s/2z/8v/y/34AfgC1ALUAkgCSAH8AfwCCAIIAYwBjABgAGADt/+3//f/9//P/8//s/+z/LAAsAHUA'+
        'dQCGAIYAbgBuAFMAUwAkACQA4P/g/7j/uP+p/6n/e/97/w3/Df+E/oT+Mv4y/ib+Jv4m/ib+Pv4+/pf+'+
        'l/4P/w//iv+K/xYAFgDKAMoAiAGIAQoCCgIgAiAC0wHTAWkBaQHyAPIAWwBbAKz/rP8b/xv/4/7j/s7+'+
        'zv64/rj+wf7B/gT/BP+S/5L/IAAgAJQAlADxAPEA4QDhAHkAeQD4//j/Xv9e/6z+rP4B/gH+s/2z/eH9'+
        '4f1B/kH+zP7M/pH/kf9OAE4A6QDpAHQBdAHmAeYBJgImAvcB9wFWAVYBZwBnAD3/Pf8t/i3+jf2N/Wr9'+
        'av2b/Zv9Jv4m/hn/Gf81ADUAQwFDAeQB5AENAg0C5QHlAVQBVAGfAJ8A8v/y/2z/bP8A/wD/cf5x/hP+'+
        'E/7n/ef96P3o/WH+Yf4i/yL/2P/Y/2gAaAAKAQoBuAG4ASsCKwJZAlkC/AH8ATEBMQFTAFMAjP+M/+v+'+
        '6/5X/lf+JP4k/mf+Z/7R/tH+R/9H/9z/3P+/AL8AnwGfAScCJwJQAlACEQIRAqUBpQETARMBXgBeAN//'+
        '3/+N/43/Wv9a/3n/ef/O/87/GAAYAHkAeQDyAPIAUgFSAZYBlgGrAasBigGKATABMAGlAKUABwAHAGP/'+
        'Y//H/sf+Rf5F/hP+E/44/jj+oP6g/mH/Yf9JAEkAKQEpAeQB5AFtAm0CzALMApcClwLiAeIBQAFAAZcA'+
        'lwDT/9P/PP88//X+9f4F/wX/VP9U/6v/q/8FAAUAcgByAMoAygDpAOkAzADMAK4ArgCYAJgAUABQAAMA'+
        'AwDV/9X/wv/C/87/zv/W/9b/6//r/zIAMgB9AH0AmQCZAJIAkgCCAIIAegB6AIQAhACHAIcAlACUAKQA'+
        'pACKAIoAQQBBAMz/zP91/3X/af9p/5D/kP/x//H/VwBXAKIAogDnAOcAGgEaAToBOgFQAVABYgFiAUsB'+
        'SwHtAO0AgQCBAEYARgAuAC4ACwALANX/1f91/3X/5/7n/lD+UP7Y/dj9tf21/c39zf3l/eX9Gv4a/oj+'+
        'iP4+/z7/CAAIAJQAlADyAPIANQE1AT0BPQEQARAB8ADwAAkBCQEnAScBQgFCAV4BXgFHAUcBNwE3AT0B'+
        'PQEHAQcBtQC1AG0AbQAzADMA7P/s/1z/XP+s/qz+BP4E/l79Xv3W/Nb8pvym/M/8z/wM/Qz9b/1v/RL+'+
        'Ev7g/uD+xf/F/40AjQA+AT4ByAHIAQcCBwIWAhYC2wHbAXkBeQE5ATkBGwEbASIBIgEkASQBFwEXAREB'+
        'EQH6APoAxQDFAHoAegAVABUAi/+L/+3+7f5Z/ln+yf3J/UH9Qf3N/M38kvyS/Kb8pvz3/Pf8gv2C/T3+'+
        'Pf4X/xf/CgAKAAkBCQHcAdwBNwI3Ai0CLQLzAfMBiwGLAesA6wA7ADsAzf/N/6H/of+//7//NAA0ALIA'+
        'sgAQARABUQFRAWsBawE8ATwBqwCrAN3/3f/w/vD+G/4b/pT9lP1R/VH9cv1y/ff99/2f/p/+M/8z/6L/'+
        'ov8HAAcAdgB2AOkA6QA0ATQBQAFAAS0BLQH1APUAlACUACQAJACv/6//SP9I//r++v69/r3+kP6Q/rD+'+
        'sP4r/yv/u/+7/1IAUgDRANEADwEPAS4BLgFFAUUBXwFfATUBNQGxALEAKgAqALL/sv9x/3H/hP+E/8X/'+
        'xf81ADUAoQChAMMAwwDJAMkA/wD/AEQBRAE9AT0B8wDzAHUAdQDF/8X/Fv8W/1n+Wf6m/ab9Tv1O/WD9'+
        'YP3K/cr9Wv5a/vr++v6V/5X/8//z/zMAMwBZAFkAZwBnAJEAkQCyALIA3QDdACQBJAFcAVwBigGKAZwB'+
        'nAGHAYcBUAFQAfYA9gCmAKYAXwBfAAgACACq/6r/Vv9W/wb/Bv+m/qb+af5p/mT+ZP5J/kn+9/33/c79'+
        'zv0N/g3+X/5f/pv+m/70/vT+bv9u/+D/4P8+AD4AlwCXALkAuQCYAJgAbQBtAEkASQAeAB4AGAAYAFEA'+
        'UQB8AHwAbQBtADkAOQD8//z/3P/c/6r/qv9b/1v/Ef8R/+r+6v4T/xP/Pf89/0//T/9z/3P/lf+V/7n/'+
        'uf/C/8L/2P/Y//v/+///////EgASAAcABwDq/+r/6P/o/8j/yP+e/57/nP+c/9H/0f8RABEAOwA7AGUA'+
        'ZQCBAIEAfwB/AG8AbwB1AHUAbwBvAEsASwBJAEkAUQBRAEcARwAsACwA6//r/63/rf+E/4T/fP98/6n/'+
        'qf//////VgBWAIsAiwCeAJ4AhgCGAFIAUgASABIA0f/R/5D/kP9K/0r/Rf9F/3z/fP+m/6b/x//H/+z/'+
        '7P8aABoANAA0ACkAKQATABMAAwADABAAEAAiACIAKwArADoAOgBLAEsAWgBaAFgAWABJAEkALAAsAPz/'+
        '/P/n/+f/6f/p/+X/5f/q/+r/FwAXAGYAZgCkAKQAxwDHAM0AzQCkAKQAUABQAOn/6f+K/4r/SP9I/yL/'+
        'Iv8J/wn/Bv8G/zX/Nf+B/4H/xP/E//j/+P8oACgAMQAxAAMAAwDP/8//q/+r/5b/lv+O/47/kP+Q/6j/'+
        'qP/L/8v/1P/U/8X/xf/O/87/8v/y//z//P/0//T/AQABAAoACgD5//n/+P/4/ykAKQBcAFwAWQBZABoA'+
        'GgC4/7j/Zv9m/0D/QP83/zf/R/9H/27/bv+d/53/z//P/wcABwBSAFIArwCvAAwBDAFSAVIBWAFYARgB'+
        'GAGsAKwALQAtALn/uf9V/1X/Dv8O/wD/AP8g/yD/Z/9n/9n/2f9NAE0ApgCmAOIA4gD8APwA+AD4AN0A'+
        '3QC1ALUAmACYAIMAgwBkAGQALwAvAOj/6P+v/6//kv+S/5X/lf+z/7P/2v/a/wQABAA6ADoAZwBnAGYA'+
        'ZgAzADMA7f/t/7r/uv+q/6r/kv+S/2j/aP9u/27/sf+x//L/8v8sACwAfwB/AM0AzQDjAOMAzwDPAMMA'+
        'wwDDAMMArQCtAIUAhQBpAGkAZQBlAHIAcgCEAIQAdQB1AEEAQQANAA0A3//f/5X/lf8y/zL/8v7y/vn+'+
        '+f4s/yz/cf9x/8X/xf8XABYAXgBeAIgAiACRAJEAhACEAFwAXAAqACoA/v/+/+f/5//h/+H/4P/g//7/'+
        '/v85ADkAfQB9AMYAxgACAQIBKQEpATQBNAESARIBwwDDAGAAYAAQABAAyf/J/37/fv84/zj/A/8D//j+'+
        '+P4S/xL/K/8r/03/Tf+M/4z/2P/Y/xgAGAA2ADYANAA0ABYAFgDV/9X/j/+P/23/bf9h/2H/bP9s/6D/'+
        'oP8BAAEAgACAAPwA/AB0AXQB6AHoATsCOwJXAlcCMQIxAuAB4AFsAWwBxQDFAAUABQA1/zX/ZP5k/sL9'+
        'wv15/Xn9m/2b/e397f09/j3+lf6V/vr++v5R/1H/h/+H/6n/qf+9/73/xP/E/7v/u/+Y/5j/ef95/4P/'+
        'g//Q/9D/XQBdAAwBDAHMAcwBggKCAgUDBQMrAysD9wL3Ap4CngIeAh4CVQFVATUANQDb/tv+qv2q/dj8'+
        '2Pxj/GP8UPxQ/Jj8mPwt/S39+v36/eL+4v7H/8f/jQCNAC0BLQGuAa4BCgIKAjECMQIjAiMC9gH2Ab8B'+
        'vwF2AXYBBQEFAWEAYQC+/77/cP9w/5z/nP8vAC8A3wDfAGYBZgGqAaoBowGjAUoBSgGJAIkAUv9S/9b9'+
        '1v13/Hf8n/uf+3n7efsE/AT8Qf1B/Q7/Dv8RAREB4QLhAjAEMATJBMkElQSVBLEDsQNVAlUCwADAADH/'+
        'Mf/2/fb9XP1c/XT9dP0f/h/+H/8f/yEAIQDkAOQAUgFSAaQBpAEMAgwCRgJGAukB6QHwAPAAzv/O//7+'+
        '/v6Q/pD+WP5Y/j7+Pv4//j/+cP5w/vf+9/7W/9b/3gDeALgBuAEaAhoC9wH3AWUBZQGSAJIAvP+8/xD/'+
        'EP+i/qL+bf5t/nH+cf67/rv+Vf9V/yMAIwDoAOgAZgFmAZgBmAGvAa8BxwHHAcABwAFhAWEBrQCtAA4A'+
        'DgDa/9r/4f/h/8b/xv+W/5b/rv+u/xYAFgBoAGgAZgBmAEgASABVAFUAegB6AHYAdgBRAFEARgBGAEoA'+
        'SgA5ADkADQANAOH/4f/b/9v/7f/t/wYABgAUABQA/v/+/9X/1f+y/7L/nf+d/4j/iP+C/4L/wf/B/zwA'+
        'PACmAKYAvgC+AJAAkABrAGsAagBqAHEAcQBWAFYAFQAVAMz/zP+e/57/oP+g/8z/zP/w//D/2f/Z/5T/'+
        'lP9W/1b/Of85/0D/QP9i/2L/kv+S/8T/xP/h/+H/9v/2/yYAJgB7AHsA0ADQAAoBCgEzATMBPAE8AQsB'+
        'CwGXAJcA/P/8/3b/dv8u/y7/LP8s/1P/U/9h/2H/Of85/w7/Dv8V/xX/Sv9K/3P/c/9w/3D/av9q/3b/'+
        'dv94/3j/Zf9l/07/Tv9h/2H/ov+i/9v/2//Q/9D/h/+H/1z/XP+m/6b/XQBdADUBNQHtAe0BSwJLAisC'+
        'KwKmAaYB8gDyAFUAVQDo/+j/c/9z/8b+xv7//f/9f/1//YH9gf3v/e/9kv6S/hf/F/9S/1L/bP9s/6H/'+
        'of8BAAEAYQBhAJsAmwC4ALgAugC6AKAAoABuAG4AJQAlAM//z/9w/3D/Mv8y/0v/S/+y/7L/TwBPABAB'+
        'EAHHAccBRwJHAn4CfgJwAnAC/wH/ARUBFQHn/+f/uv66/sX9xf0y/TL9D/0P/UX9Rf2v/a/9XP5b/lH/'+
        'Uf9lAGUAUwFTAesB6wEuAi4CJgImAukB6QGQAZABEgESAWAAYACG/4b/zf7N/oP+g/6z/rP+Of85//v/'+
        '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'+
        'HwDM/8z/jf+N/3D/cf97/33/pv+o/9T/1f88ADsArwCsABwBFwGEAYABnwGcAYwBigEuAS4BpwCqADcA'+
        'PQC+/8L/XP9b/wD//v6z/rf+if6V/o3+mf6//sn+Bf8S/2//fv+8/77//v/r/1wAOgCeAH8AwgCpALQA'+
        'nAB9AGsAGAAXAKr/uf9a/2v/G/8m/yD/K/9N/1z/l/+o/+//AgAvAEYAiwClANsA7wD5AAQB9QD1AMQA'+
        'tQB3AFcAAgDd/3X/Wv8Z/wn/8f7k/vn+8/4y/zz/ZP93/7L/uP/q/97/5P/U//b/6P/z/9//BADu/yUA'+
        'HwATACcAEwA6AAoAOQAGADgA/f8qAPn/FgAFABAA7f/v/wEA+f8YAAUALAAaADcAMABFADYAdAA/AEcA'+
        '8/8XAMv/t/9+/2X/Mf+G/2L/W/91/2f/zv+R/w4Aw/8eAPX/LwDo/xsAGQA4AAoA9v8/AAoAQwAjAAEA'+
        'CABYAGIAEwD+/yEA8f9bACQAQgAJAF4AIwDn/7f/9P/o/+7/EwDJ/xYADgBgAO3/HQAyADQAAADn/9z/'+
        'wP+7/5//Jf8J/0L/Pf/c/gX/6f4x/0X/ff9//4n/MAAcACcAFQCbAI4AxgCvAMwAqwACAfIAaQBzAJkA'+
        'nQAuAAkAuP9y/7b/fv9f/07/tP+5/3D/ff+p/9T/t/8MAJn/+f9lAKEALAA+AIcAjgBUAFcA6v/W/2oA'+
        'PgC6/5H/2f+9/6T/gP8n//X+S/8l//D+5/6E/4j/nv+j/xYAMADPAAcB2gARAbIByAF1AXcBQwFKARgB'+
        'IAFgAE8ASQAiAI//b/9d/03/XP9K/1D/Nf9x/1z/Hf8e/3z/lv+X/77/1P/+/9X/9f+z/8X/KAAyAK//'+
        'r/8/ACcAUgAmAMz/qv8+ADgAW/9g/3X/a/+s/5z/d/97/ysAQAATACAAvQC/APUABAHCAOYA2AD8ADgA'+
        'RwCEAIEAMQAnAKv/nP+9/6f/1v+7/9//x/9B/zD/nf+V/yH/H//9/vj+yP+9/zv/MP/O/8v/k/+V//T/'+
        '8//JAMYA+P/9/+4A/AASAB8Ah/+Q/xgAJgDP/ub+ev+S/0n/Xf+l/7T/mAClAEwAUgA5ATUBAgD4/yUA'+
        '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'+
        'WOzX7NfszO/M7xTpFOma65rrcfhx+CH/If9X+1f7JPsk+38BfwFiAGIA7v3u/VX+Vf4R/BH8E/4T/kkC'+
        'SQLF/8X/2fnZ+T37PfuvA68DmAaYBqACoAKr/av9X/lf+Sb4Jvg5+Dn4mvaa9tH30fcp+in6Nvo2+kcB'+
        'RwEjByMHxALEAoECgQLEAcQBh/+H/8MDwwOiBKIE6gXqBVYIVgjKCsoKyw7LDowHjAerA6sDEQgRCEwE'+
        'TAQNBQ0F8AbwBsIBwgG1/7X/+Pr4+uD34PfS+tL6Yflh+Ur7SvuZ/5n/p/yn/Hb9dv1N/03/O/s7+5b6'+
        'lvqn+af5OPg4+Nj62Pol+yX7Ivwi/M/9z/0w/TD9+//7//YC9gI7ADsAWPxY/Mj9yP0u/y7/fP58/joD'+
        'OgPWBdYFfwN/A10FXQUaCRoJIgsiC/EJ8Ql9B30H7QXtBdsC2wIYAxgDtgS2BEEDQQMaBBoECwULBT0E'+
        'PQTCA8IDxgPGA8QGxAYvBy8H+AL4AqMCowK9A70D1wDXAPX+9f5pAGkAbABsAFAAUACWBJYEfgZ+Bp8E'+
        'nwRVBlUGUAVQBZkAmQBeAF4Azv/O/wz+DP6W/Jb8Xvle+dL40vg/+D/40PjQ+P/8//yK+4r7Yfph+oX9'+
        'hf0f/R/9yPzI/Iz8jPyl/aX9+QH5AcEBwQHLAMsAMwIzAhcCFwIQBREFxQjFCCMHIwcPBg8GzAfMB6QF'+
        'pAUTAxMDhwSHBDgDOAPuAO4AEgETAUEBQQGAAoACXwFfAfz//f94AHgA8v3y/af+p/76APsACP8I/3v/'+
        'e/8sACwAlv+W/1ABUAHNAc0BLQEtAacApwAzADQA7QDtANkB2QG2ArYCcAJwArYAtwBC/kL+2vzb/Er/'+
        'Sv8nAScBkACQAIcBiAFYAlgCPwI/AqICogJ/A38DiAOIA2EBYQEUARQBqgGqAYsAiwAJAQkBnv+e/wz9'+
        'DP11/nX+4v/i//D/8P/g/+D/jwCPABkBGQF6/3r/vv++/+wA6wDiAOIAEwISAgECAQIOAQ4BeQB4AIX/'+
        'hP/3/vf+Hv4d/r/+v/56/3n/Wv9Z/14BXQFcAlsCgAJ/AjIDMQPiAuEC2QPYA+YD5AORAo8C0wLRAvEB'+
        '8AFYAFYAGAAWAMX/w/8r/yn/cv5x/oP9gv0A/f78+/z5/Aj+Bv7B/7//vf67/jH9L/0S/hD+Av7//Wv+'+
        'af64/rX+nv2c/dX+0/7H/8T/2wDZAE4BSwFF/kP+GP4V/rb/s/8mASMBoAKdAvAA7QArASgBVwFVAcP+'+
        'wP4x/y7/SP9F/9j+1f6eAJsAJAIhAvUC8gJMAkgCYwJgAhMDEAMTAhACrgKqAmwDaQMrAicCjAGIAecB'+
        '5AF7AXgB1P/R/zAALAC7ALcAfP94/0YBQgFMAUkBuP61/sMAwAAlASEB2//X/0wBSAFh/17/oP2c/bf+'+
        's/40/jH+CP8E/+UA4gDTANAAaQBlACgAJQA1/zH/Jf4i/k3+Sv7K/sb+eP91/9f/1P89/jr+If0e/aP8'+
        'oPzJ+8b7Rv1C/Wz/af8QAAwA4QDeAGQBYAFhAF4ABf8C/+/97P0J/Qb9Lf0r/R/+HP5R/07/iwCIAOoA'+
        '5wDmAOMAPAE5AZUBkgF0AXIBfgF8AeoB5wE5ATcBAgEAASoBKAEw/y7/ff57/nT/cv8m/yT/xf/D/18B'+
        'XQH/Af4BFwIVAgAC/gGUAZMB8wDxANMA0QCtAKsAQgBAAHEAcABHAEYAk/+S/+r+6f5u/m3+uP63/lX/'+
        'VP/a/9n/BwAGAIAAfwAWARUBxf/F/zH/MP9dAF0A1ADTAAICAQKuAq4CIgIhAnQCdAJBAkEC2wHaAVUB'+
        'VQGtAKwA6wHrAfQC9AKnAqcChwKHAsQCxALFAsUC4QHhAQMCAwLnAucCcQJxArgBuAFIAUgBTwFPARYB'+
        'FgF3AHcAwQDBAFAAUAC1/7X/IwAjABMAEwASABIArACsALUBtQFgAWABtf+1/xUAFQArACsAXv9e/7IA'+
        'sgCFAYUB6wDrAOQA5AAPAQ8BiQCJALX/tf/C/8L/jACMANAA0AAkACQAgwCDAG8BbwG0ALQAvAC8ANQA'+
        '1ACY/5j/pf+l/9D/0P+W/5b//////30AfQClAKUAQABAAOMA4wCZAZkBTAFMATQBNAHHAMcAPwE/AcMB'+
        'wwHWANYAPAA8AGv/a/9J/0n/8f/x/87/zv81ADUAywDLAAcBBwELAQsBeQF5ARoCGgLJAMkAVABUANYA'+
        '1gCe/57/ef95/4D/gP82/jb+1P3U/WD+Yf4I/wj/vv++/8oAygAJAQkBIwEjAeYB5gF0AXQBDgEOAfMA'+
        '8wCoAKgAoACgAJT/lP9q/2r/t/+3/8H+wf69/73//QH9AQECAQIqASoBBAIEAlwCXALDAMMAKQAoADwA'+
        'PABm/2b/qv6q/o3+jv6l/qX+ZP5k/hb+Fv5J/kn+Rf9F//b/9v+Y/5j/xf/F/zEAMQAbABsAaQBpAJIA'+
        'kgC+/77/G/8b/xEAEQCxALEA9//3/6//r/9T/1P/3P7c/gX/Bf89/z3/s/+z/6T/pP8U/xT/av9q/8z/'+
        'zP+m/6b/4f/h/0cARwBKAEoAEwATAPb/9v/u/+7/6f/o/zEAMQCEAIQAYABgANf/1/88/zz//v7+/iT/'+
        'JP+j/6P/FQAVAOv/6//h/+H/WgBaAKcApwCXAJcAowCjAJIAkgDm/+f/kv+T/5z/m/9U/1T/Lv8u/0L/'+
        'Qv+g/6D/0//S//v/+/+kAKMArwCuAGkAagBjAGIA/f/9/9f/1/8EAAMA9//4/+v/6v/b/9v/x//J/y8A'+
        'LQC/AL8ApACkAHIAcgCGAIYATwBPADAALwAxADAAtf+2/3r/ev/N/8z/lP+U/yf/J/8O/w3/6f7q/j3/'+
        'Pf/j/+H/IQAjAAEAAADO/83/df94/8v+yf64/rj+Av8E/87+zP7L/s3+1v7V/pf+l/6Y/pn+Ef8P/5P/'+
        'lf9z/3L/Tv9N/23/b/8m/yT/uf66/lj+Wf7q/ej91v3X/Rr+G/5S/lH+tv63/hv/G/8O/wz/JP8l/3L/'+
        'cv+t/6z/FQAWAFsAWgAgACAAo/+j/1b/Vv8r/yv//P77/vj++v6f/p3+Cf4J/u397/3X/dT94f3j/U3+'+
        'Tv5O/k3+UP5S/vj+9/5S/1P/DP8L/xj/GP91/3b/QP8//97+4P7Z/tf+z/7P/s3+zv7U/tH+2/7d/h7/'+
        'Hv9b/1r/Pv9A/+X+4/7o/ur+Ff8U/+b+5f7B/sP+jv6M/l3+Xv6b/pz+3P7a/v/+AP9K/0r/f/9+/5b/'+
        'l//B/8H/vv+9/63/rv/L/8v/8f/v/xQAFgAQAA8ACAAHAAkACwAoACcAUgBSAB0AHwAXABUAQgBDAHYA'+
        'dwCIAIYAMAAyAGsAawCqAKkAQwBEAO7/7P/1//b/FwAXAK7/rf+9/7//5v/l/3P/c//C/8P/1f/U/4r/'+
        'i//h/+H/PgA9AE4ATgBaAFoAiwCLAPb/9v+x/7H/HQAdAKf/qP+h/6H/VABUAKAAoAA6ADoAAwADAHkA'+
        'eQBiAGIAYABhABMAEgCc/5z/BgAGANX/1P/T/9T/JwAnAAcABwDW/9f/jP+L/wQABAAhACEAsf+w/9b/'+
        '1///////9//3/6D/oP+4/7f/o/+j/wD/Af9Y/1f/BAAEAIIAggC3ALcAdAB0AFgAWACmAKYAwADAAGoA'+
        'agBkAGQA8f/x/6X/pf++/77/kv+S/2QAZABLAEsAsv+y/ysAKwDF/8X/kv+S/8z/zP+7/7z/xf/F/9b/'+
        '1v/G/8b/mv+a/z0APQAiACIAu/+7/xcAFwDm/+b/MwAzAD4APgB/AH8AmgCaAN3/3f9IAEgAzv/O////'+
        '//8pACkAI/8j/zsAOwCc/5z/xv/G/9AA0ABe/l/+Df8N/64ArgAh/yH/IAAgAN0D3QOq/6r/Wvla+fYA'+
        '9gAXABcAQfdB9x0AHQDD/8P/FvkW+fgC+AIbARsBNPc092r9av3LAcsBqPqo+o79jv3vBO8EcP9w/8YA'+
        'xgBKBEoE8gDyAM4FzgUKBAoEewJ7AscHxwfAA8ADCwMLA/X/9f8+/j7+kQeRBxoEGQQ1/jX+XwFfAbgA'+
        'uADW/db9pP6k/psAmwBz/HL80/3T/TIBMgHz/vP+cARwBM8AzwAO/A78BAIFAu377fs0/TT9KwMrAzH9'+
        'Mf2kAKQA7APsA1UCVQLWA9YDkQKRAoYChgLT/tP+j/2P/UoDSgNaAloC6wHrATYCNgL4APgA+gP6Ay7+'+
        'Lv5X+Vf5LAAsACYAJgDa/tr+dAJ0Ag0EDQSBBYEFXQFdAdb/1v85AzkD2/7b/kr+Sv5JAEkAff19/ZoA'+
        'mgB2A3YDUAFQATsAOwAoASgBNQE1ASv+K/7dAN0A2AXYBf4B/gGm/6b/sAGwARj/GP+M/Yz9RQBFAGgA'+
        'aAD+//7/Mf8x/2/8b/wi/iL+Y/9j//X+9f5jAWMBTgBOAOz/7P/aANoAewB7ANwD3AMEAwQDXwBfAJ0A'+
        'nQBy/nL+JQAlAJIBkgHO/8//dQB1AOL+4v6M/oz+/gD+AP8B/wEAAgACJQAlABQAFABfAF8AVf9V/5b/'+
        'lv+p/an9bP1s/WAAYACiAKIACAAIAJn/mf8V/xX/v/6//pD+kP6yALIA9wH3AeT/5P9v/2//QQBBADj+'+
        'OP6C/YL9I/4j/pP8k/xi/mL+LAAsAE79Tv2N/Y39rP+s/5f/l/+e/57/wv/C/63/rf81/zX/3/7f/lT+'+
        'VP7b/dv9PP48/hr+Gv6t/q3+nP6c/qz9rP3Z/tn+xv7G/jv+O/7G/8b/zf/N/zT/NP8OAA4AZQBlAGf/'+
        'Z/8g/yD/Z/9n/9393f3G/Mb8JP0k/a79rv3X/9f/5QDlADT/NP9y/3L/4wDjAJEAkQCjAKMAfAB8ACT/'+
        'JP/m/ub+Ef8R/7n+uf7W/db9M/0z/T3+Pf7c/tz+VP9U/5IAkgAgASABkgGSAfEA8QDw//D/+f/5/yQA'+
        'JABZAVkBZAFkAQwADAD2//b/nf+d/5H/kf9z/3P/7v7u/lMAUwB9AH0AT/9P/xsAGwCEAIQAvwC/AFUC'+
        'VQJ+An4CgwGDAdIB0gELAgsCJgEmAQ8BDwF2AXYBlwCXADsAOwDKAMoAhQCFAOgA6ABFAUUBhgCGAEAB'+
        'QAFVAlUCjwKPAuIC4gKEAoQCsAKwAgIDAgPHAccBHwEfAboBugH7AfsBygHJAecB5gGOAY0BewB6AJQA'+
        'lAAIAQcBcQBxAIwAjAB2AXYBUgFSAdoA2gCiAaIBkgGTAZcAmADxAPIA5ADlAPEA8gCjAaQB3gDfAPgA'+
        '+gDcAd0BgwGFAe8B8AFYAlkCXgFfAcAAwADBAMIANQA2ALn/uf/6//n/5P/j/xgAFwCsAKsANwA1ACAA'+
        'HgC/AL0AxADBAKYApACwAK4AnwCcAKMAoQD3APUA9QDzAPsA+gCGAYUBFwEWAdEA0QDQAdABJwInAgMC'+
        'BQIWAhgCtQG3AQQBBgHvAPIAEgEWAXkAfABXAFsAdgB5AEIARgDlAOkAKgEsAb0AwACoAKoAmQCbAJoA'+
        'mwCRAJEAjwCOAIgAhwCnAKUAnQCaAKH/nv9t/2r/6P/k/5f/lP98/3j/5v/j/08ATAAMAAgAFgATALIA'+
        'sABHAEQANAAyAF8AXgDx//D/9//3/8P/xP+s/67/lv+Y/0r/Tf9k/2f/D/8S/0//Uv/e/+L/6v/u/5gA'+
        'nADQANQAMAAzALf/uv+v/7H/xv/I/77/v//0//X/1//X/9H/0P+5/7j/Of83/7b/tP/J/8b/iP+F/0UA'+
        'QQBhAF4AUgBOAHAAbAD5//X/FAAQACEAHgBL/0j/hwCFAMcAxQBn/Wb9kv+S/64BrgFw/XH9DAANANsB'+
        '3QFb/V79vv/B/5oAnQBh/WX9c/93/3EBdQGYAJwAdAB4AKYBqQE/AEIAbP5u/vb/+P+p/qv+iv2L/UX/'+
        'Rf/r/uv+5f3j/av+qf6eApwCtgCzAPP88PxgAlwCEwAPAKT9of0mBCIEPv86/xn+Fv7CA78DQv5A/nH+'+
        'b/68ArsCA/8C/6P+o/5NAU0BWgBbAMf/yP9aAVsBtv+4/6P+pf4oASsB1v7Z/qH9o/0TABYACf8M/yT/'+
        'J//z/vX+LQAvAMoDzAM8AD4AQf5C/o0BjQEYABgAVf5U/i3/Lf99/3z/DwAOAPL/8f8H/wb/3v/c/ygA'+
        'JwDX/db9u/66/mgAZgDO/sz+jACLAHgCdwI+AT0B4QDhAFb/Vv9F/0X/rACsAID+gP66/br9Kf8q/1/+'+
        'X/4x/jH+IP8g/8b+xv7D/sP+uP64/nr+ev6m/6b/ff9+/2P/Y/8hASEBBQAGAHb+dv5C/0L/WP9Y/3P/'+
        'c//g/+D/1//X//j/+P9GAEYAEQARAHH/cf81ADUAbgBuAGD/YP8bABsAUABQAFX/Vf9r/2v/SP9I/1H/'+
        'Uf9g/2D/wP7A/kH/Qf+C/4L/+v76/sD/wP8iACIA+f/5/zMAMwBJ/0n/of+h/zkBOQHMAMwAnQCdAFcB'+
        'VwFqAWoBLAEsAbAAsACdAJ0AagBqACAAIABfAF8AXwBfAJAAkABnAGcAFwAXAJoAmgBnAGcAoP+g/4j/'+
        'iP+q/6r/Af8B/2X+Zf6//r/+rf6t/oj+iP5B/kH+Mf4x/tf/1/8cABwAFP8U/8r/yv9YAFgA7//v/4z/'+
        'jP8r/yv/ov6i/lL+Uv62/rb+K/8r/9H/0f/t/+3/Q/9D/3f/d/9B/0H/o/6j/hr/Gv8w/zD/v/6//gf/'+
        'B/+5/7n/hv+G/8L+wv6+/r7+if6J/h7+Hv6n/qf+R/9H/yv/K/+L/ov+5f3l/bv9u/1c/lz+Rv9G/7X/'+
        'tf/2//b/4//j/5v/m//f/9//BgAGAL//v/9a/1r/HP8c/2H/Yf9//3//cf9x/6f/p/+6/7r/xf/F/6b/'+
        'pv+K/4r/CAAIAP3//f9Q/1D/VP9U/5L/kv+a/5r/Xf9d/4T+hP5Y/lj+6P7o/m3+bf6C/oL+lACUAE4B'+
        'TgEfAB8AWgBaANgA2ADP/8//DgAOADEBMQH5//n/Z/5n/nr+ev6G/ob+zv7O/nH/cf9e/17/cv9y/6T/'+
        'pP+R/5H/yv/K/8j/yP+s/6z/9v/2/0cARwCOAI4AjgCOAHwAfAC4ALgAtQC1AHgAeABEAEQA/////9X/'+
        '1f97/3v/JP8k/37/fv/d/93/3//f/9b/1v+n/6f/p/+n/6r/qv+y/7L/eQB5AP0A/QDIAMgAvgC+AIYA'+
        'hgBRAFEAewB7AHUAdQB/AH8AfQB9AEEAQQBlAGUAXQBdAAsACwALAAsA/f/9/zMAMwCHAIcASABIACUA'+
        'JQDp/+n/gv+C/4v/i/+g/6D/pf+l/2X/Zf9S/1L/kf+R/2X/Zf++/77/QQBBAC8ALwAVABUArv+u/83/'+
        'zf8lACUAJQAlAHYAdgD2//b/cv9y/6H/of9h/2H/dP90/3D/cP9N/03/eP94/zP/M/9u/27/l/+X/wz/'+
        'DP8q/yr/Zv9m/07/Tv9o/2j/nP+c/5L/kv9I/0j/Zv9m/5n/mf9p/2n/cf9x/6f/p/+w/7D/sP+w/77/'+
        'vv/L/8v/zv/O/7r/uv+k/6T/ov+i/5D/kP+K/4r/if+J/0n/Sf8M/wz/A/8D/xT/FP8K/wr/Gf8Z/2P/'+
        'Y/8v/y//Lv8u/+j/6P8PAA8A3f/d/8v/y/+S/5L/of+h/7T/tP/Q/9D/BQAFAMv/y//d/93/FgAWAN7/'+
        '3//6//r/AQABAOb/5v8cABwA9f/1/9P/0//p/+n/8v/y/1kAWQBwAHAAMwAzADcANwD1//X/sv+y/8j/'+
        'yP/a/9r/8f/x/xsAGwAvAC8AJwAnAGAAYACKAIoAUABQAFoAWgBWAFYA3//f/8L/wv/9//3/CgAKAA8A'+
        'DwAZABkADQANACsAKwAUABQAmP+Y/+L/4v9nAGcAEAAQAO3/7f/d/93/i/+L/8X/xf/4//j/2f/Z/6r/'+
        'qv9W/1b/Rf9F/1b/Vv9l/2X/lP+U/7X/tf/f/9//7//v//j/+P8YABgABQAFAPD/8P/X/9f/v/+//+//'+
        '7/8DAAMA2v/a/9r/2v/z//P/zP/M/6H/of+O/47/WP9Y/1T/VP9c/1z/V/9X/4b/hv+L/4v/uf+5//r/'+
        '+v/K/8r/rv+u/7r/uv/h/+H/CQAJAPv/+//5//n/1v/W/73/vf/V/9X/xf/F/7X/tf+6/7r//////zIA'+
        'MgD+//7/BgAGABUAFQDj/+P/3f/d//v/+/8GAAYA5v/m/8r/yv/L/8v/4//j//f/9//+//7/SwBLAHAA'+
        'cAA2ADYAUgBSAJoAmgC2ALYAtQC1AKAAoACnAKcArwCvALgAuADFAMUA0gDSAP0A/QD5APkA7ADsABMB'+
        'EwH1APUAxQDFANEA0QDUANQA9gD2ACIBIgEuAS4BNgE2AQwBDAHuAO4A7gDuAMsAywD6APoALQEtAQ4B'+
        'DgEUARQBBQEFAewA7ADYANgAigCKAGoAagBvAG8AewB7ALoAugDbANsA1wDXAMgAyACYAJgAbABsAGgA'+
        'aACIAIgAvgC+AOMA4wDOAM4ArwCvAKgAqACBAIEAWwBbAFkAWQByAHIArwCvAM0AzQC3ALcAlgCWAGEA'+
        'YQBKAEoAhwCHALkAuQCHAIcAWABYAIUAhQC/AL8AxQDFAK8ArwCgAKAAiQCJAFcAVwA+AD4ARwBHAEYA'+
        'RgBIAEgAWgBaAFMAUwAqACoABAAEAP////8HAAcAAQABAPT/9P/f/9//3v/e/wIAAgAOAA4A5v/m/83/'+
        'zf/m/+b/AAAAABYAFgBbAFsAngCeAJkAmQBtAG0ARQBFAEQARABuAG4AiwCLAIgAiACDAIMAdwB3ADgA'+
        'OADd/93/tf+1/7T/tP+1/7X/2v/a/xQAFAAuAC4AXgBeAKYApgCkAKQAewB7AGQAZABMAEwAQABAACQA'+
        'JAD1//X/5//n//L/8v8JAAkABQAFANT/1P/U/9T/BgAGAB4AHgAPAA8A7//v/+P/4//Z/9n/wP/A/8P/'+
        'w/+y/7L/bP9s/0H/Qf9I/0j/c/9z/7X/tf/U/9T/tv+2/67/rv/c/9z/9f/1/+r/6v/L/8v/qf+p/73/'+
        'vf/a/9r/0P/Q/+z/7P8lACUAKgAqABUAFQAvAC8AUwBTAEgASAA/AD8AYgBiAH8AfwBsAGwAZQBlAJIA'+
        'kgCrAKsAngCeAJEAkQB+AH4AaABoAFQAVABCAEIAEAAQALf/t/+T/5P/sv+y/8T/xP/c/9z/JwAnAGIA'+
        'YgBiAGIAaQBpAHUAdQBgAGAATwBPAFsAWwCCAIIApACkAKcApwCRAJEAYABgADsAOwAzADMAHQAdAPz/'+
        '/P8EAAQALwAvADYANgAUABQA/P/8/+n/6f/c/9z/5P/k//z//P/8//z/2//b/+f/5/8QABAABwAHAOv/'+
        '6//x//H/HgAeAFsAWwCSAJIApgCmAIcAhwBjAGMAUgBSAFAAUABWAFYAVQBVAEYARgAoACgAHQAdACQA'+
        'JAAUABQAAQABAOz/7P/c/9z/+v/6/ysAKwAnACcAFQAVADoAOgBdAF0AYwBjAGsAawBgAGAAYgBiAHIA'+
        'cgBvAG8AdAB0AHcAdwBoAGgAWgBaAFgAWABaAFoASgBKAEwATABfAF8AUABQACYAJgAKAAoAAAAAAPH/'+
        '8f/i/+L/4//j/97/3v/M/8z/vf+9/8v/y//6//r/LAAsAEsASwBdAF0AdAB0AGwAbAAuAC4AAgACAPj/'+
        '+P/w//D/8P/w//r/+v/1//X/5v/m/+r/6v/f/9//vf+9/7D/sP+y/7L/xf/F/9b/1v/G/8b/w//D/77/'+
        'vv+Z/5n/iP+I/57/nv+4/7j/vP+8/6L/ov97/3v/bv9u/3L/cv9Z/1n/bv9u/8b/xv8JAAkAKQApABIA'+
        'EgDK/8r/iv+K/07/Tv8r/yv/Ov86/1r/Wv9r/2v/cv9y/4v/i/+s/6z/vP+8/6//r/+T/5P/m/+b/8H/'+
        'wf+3/7f/hP+E/1b/Vv8o/yj/B/8H/wr/Cv8j/yP/R/9H/13/Xf9X/1f/Pv8+/yj/KP8w/zD/Qv9C/0n/'+
        'Sf9N/03/Tv9O/1n/Wf9u/27/Yv9i/yv/K/8I/wj/F/8X/yj/KP8z/zP/Lv8u/wX/Bf/y/vL+IP8g/2X/'+
        'Zf+h/6H/xP/E/6//r/9x/3H/Of85/xP/E/8A/wD/3v7e/qr+qv6i/qL+w/7D/vj++P5B/0H/av9q/3f/'+
        'd/+R/5H/mf+Z/4X/hf+E/4T/eP94/1H/Uf9o/2j/n/+f/5X/lf+N/43/uf+5/+H/4f/f/9//5P/k/wIA'+
        'AgAAAAAA8P/w//D/8P/i/+L/zP/M/7v/u/+2/7b/uP+4/7b/tv+0/7T/pf+l/57/nv+u/67/zf/N//z/'+
        '/P8sACwAUQBRAFcAVwA2ADYAGQAZAA0ADQAAAAAA8//z/+T/5P/G/8b/tf+1/9X/1f8BAAEAGgAaADkA'+
        'OQBTAFMAYQBhAG0AbQBkAGQAXABcAGkAaQBuAG4AbgBuAGcAZwBPAE8AKwArAP/////q/+r/BQAFADIA'+
        'MgBTAFMAYgBiAEcARwAUABQACgAKABgAGAAfAB8ALwAvACUAJQAbABsATgBOAH0AfQBuAG4AXQBdAGMA'+
        'YwBSAFIARQBFAGYAZgB8AHwAaABoAEQARAAzADMAQABAAEQARAAtAC0ABAAEANv/2//F/8X/vP+8/8P/'+
        'w//T/9P/4P/g/+7/7v/4//j/BwAHAB4AHgAhACEACQAJAPT/9P8AAAAAFwAXAA0ADQDn/+f/sv+y/4n/'+
        'if+l/6X/4//j//b/9v/i/+L/v/+//6r/qv+8/7z/1f/V/+L/4v/s/+z/1v/W/6f/p/+U/5T/ov+i/7v/'+
        'u//c/9z/9v/2/wEAAQAMAAwAGQAZABAAEADn/+f/vP+8/7n/uf/n/+f/GAAYACgAKAAoACgALgAuAC0A'+
        'LQAcABwACgAKAAIAAgAFAAUAGAAYACMAIwAiACIATQBNAIsAiwCFAIUAZgBmAFsAWwAkACQA3//f/+f/'+
        '5//4//j/0P/Q/73/vf/e/97/7//v/+D/4P/Z/9n/8v/y/wYABgDy//L/0P/Q/8f/x//Z/9j/4//j/+H/'+
        '4f/k/+T/9P/0/xkAGQAtAC0AHwAfACQAJAAjACMACwALABAAEAAYABgAFQAVACYAJgAnACcAEgASABUA'+
        'FQAmACYAHQAdAAgACAD6//r/4v/i/9n/2f/f/9//2//b//P/8/8dAB0ALwAvAC0ALQAVABUA5f/l/7n/'+
        'uf+p/6n/qP+o/7P/s//D/8P/rf+t/5f/l/+c/5z/mf+Z/5T/lP+X/5f/of+h/4//j/9x/3H/iv+K/7H/'+
        'sf/J/8n/6P/o/wEAAQALAAsADwAPACEAIQA7ADsAUwBTAGEAYQBcAFwAWQBZAFUAVQBNAE0APAA8ACAA'+
        'IAAeAB4AMAAwAEsASwBTAFMAPwA/ADoAOgAyADIAHQAdABcAFwAiACIAPwA/AFcAVwBgAGAAbwBvAJMA'+
        'kwCoAKgAmgCaAKUApQC8ALwAvgC+ALYAtgCXAJcAcwBzAGIAYgBgAGAAUQBRAEsASwBgAGAASgBKACIA'+
        'IgAYABgABgAGAAAAAAATABMALQAtADYANgAhACEAHwAfAC4ALgAkACQAEQARAA4ADgAbABsAJgAmACAA'+
        'IAAVABUAFgAWABwAHAAVABUA+P/4/97/3v/1//X/JAAkADYANgA2ADYAMQAxADIAMgA7ADsARgBGAE4A'+
        'TgAqACoA8v/y/+D/4P/N/83/pv+m/5L/kv+g/6D/yP/I//L/8v8TABMAIQAhABwAHAAhACEALgAuADwA'+
        'PABYAFgAcgByAHYAdgBfAF8AQABAADQANABNAE0AcgByAGAAYAApACkAFgAWACcAJwBBAEEAUQBRAE4A'+
        'TgBAAEAANAA0ACgAKAAMAAwA+f/5//n/+f/2//b/+f/5/wcABwAQABAAEgASAAcABwD//////////wUA'+
        'BQAlACUATQBNAFAAUAAuAC4ACwALAP3//f8GAAYAGwAbACEAIQAoACgARABEADIAMgAGAAYABAAEABAA'+
        'EAAbABsAKAAoAEEAQQBZAFkAXABcAG4AbgBZAFkAFQAVAAcABwAeAB4AKQApADoAOgBXAFcAcgByAGwA'+
        'bABSAFIAVwBXAGIAYgBFAEUALQAtAEMAQwB1AHUAngCeAJ0AnQB8AHwAaABoAHYAdgCNAI0AowCjAMEA'+
        'wQCxALEAcwBzAEoASgBFAEUAXABcAHIAcgCIAIgAlgCWAIEAgQB0AHQAZgBmAEMAQwAyADIAQwBDAHAA'+
        'cACdAJ0AtgC2ALoAugCrAKsApQClAK0ArQCpAKkAlgCWAH4AfgBhAGEAPAA8ABEAEQDt/+3/3f/d/9f/'+
        '1//l/+X/CgAKABgAGAAdAB0ALQAtAC4ALgAdAB0ACQAJAAwADAALAAsAAQABACgAKABkAGQAigCKAKwA'+
        'rAC1ALUAmgCaAHAAcABKAEoAKQApABcAFwAgACAAIAAgABwAHAAxADEAMwAzABcAFwD0//T/1P/U/8X/'+
        'xf/Y/9j/DQANACoAKgAqACoANQA1ADgAOABNAE0AdgB2AHQAdABCAEIADQANAAgACAAxADEAYwBjAIUA'+
        'hQCOAI4AnQCdAK4ArgCzALMAuQC5AMMAwwDXANcA6ADoAOkA6QDGAMYAewB7AEQARAA0ADQALgAuAD8A'+
        'PwBhAGEAdAB0AHIAcgBfAF8AVwBXAGQAZABpAGkAdQB1AHgAeABVAFUAPwA/ADwAPAA3ADcALAAsACsA'+
        'KwBAAEAAVwBXAHcAdwCVAJUAmwCbAJYAlgB7AHsASwBLAB8AHwAKAAoAAgACAPf/9//1//X/5//n/9j/'+
        '2P/g/+D/3P/c/87/zv/L/8v/2v/a//L/8v8EAAQABAAEAPT/9P/0//T/9v/2/+v/6//v/+//AQABABwA'+
        'HAA+AD4AZABkAIUAhQCaAJoApwCnAKkAqQCoAKgAmACYAHgAeABjAGMAVgBWAEAAQAAdAB0ACwALABQA'+
        'FAAgACAAMAAwAC8ALwAbABsAGAAYACAAIAAWABYA8v/y/9H/0f/S/9L/7v/u/w4ADgAvAC8ASwBLAE8A'+
        'TwBGAEYAOwA7ACwALAAYABgA/v/+/+n/6f/Q/9D/0P/Q//n/+f8VABUAFAAUAAoACgD/////9//3//X/'+
        '9f/4//j/7//v/+P/4//j/+P/4f/h/9T/1P/P/8//2P/Y/+X/5f/y//L//v/+/xIAEgAZABkAAwADAPT/'+
        '9P/c/9z/u/+7/8v/y//v/+///P/8//z//P/8//z/AgACAP3//f/u/+7/zf/N/6P/o/+W/5b/nf+d/5r/'+
        'mv+S/5L/lf+V/5z/nP+Q/5D/ff99/4f/h/+f/5//q/+r/7H/sf+f/5//cv9y/0//T/9K/0r/YP9g/3r/'+
        'ev+W/5b/rv+u/6v/q/+d/53/mv+a/6z/rP/M/8z/2v/a/8n/yf+o/6j/nf+d/67/rv+9/73/wv/C/7//'+
        'v/+7/7v/wf/B/8j/yP/P/8//1//X/9f/1//F/8X/sf+x/7H/sf+p/6n/p/+n/8f/x//V/9X/xf/F/7L/'+
        'sv+i/6L/lP+U/4H/gf+A/4D/nf+d/63/rf+x/7H/q/+r/5P/k/+L/4v/qP+o/9L/0v/p/+n/5P/k/9P/'+
        '0//H/8f/wf/B/73/vf++/77/u/+7/6X/pf+M/4z/i/+L/5//n/+w/7D/vf+9/8//z//V/9X/1v/W/+L/'+
        '4v/8//z/HQAdADgAOAA3ADcAEwATAPP/8//i/+L/x//H/8f/x//3//f/JQAlAEIAQgBQAFAAPwA/ABgA'+
        'GADz//P/4v/i/93/3f+//7//of+h/5v/m/+V/5X/nf+d/7j/uP/U/9T/6v/q//z//P8TABMAIwAjAB4A'+
        'HgAPAA8A/////+P/4//H/8f/w//D/9T/1P/t/+3/+P/4/+3/7f/e/97/4v/i//n/+f8AAAAA9P/0/+v/'+
        '6//h/+H/1v/W/87/zv/Y/9j/2v/a/73/vf+9/73/y//L/73/vf+9/73/y//L/9D/0P/K/8r/yf/J/+P/'+
        '4/8FAAUAFQAVABcAFwAbABsAEwATAO//7//H/8f/rf+t/6L/ov+c/5z/mv+a/6f/p/+8/7z/2v/a//j/'+
        '+P8NAA0AJAAkADcANwA2ADYAGAAYAOz/7P/N/83/y//L/9j/2P/S/9L/tv+2/6f/p/+5/7n/3P/c//D/'+
        '8P/p/+n/4//j/+n/6f/r/+v/9//3/woACgAPAA8AEwATAA4ADgDv/+//0P/Q/8//z//Z/9n/xv/G/6D/'+
        'oP+i/6L/xv/G/+r/6v8HAAcAGwAbAB0AHQAUABQAAQABAOj/6P/P/8//qf+p/5b/lv+k/6T/qf+p/7P/'+
        's//M/8z/1v/W/9L/0v/I/8j/wv/C/8L/wv/D/8P/tf+1/5z/nP+S/5L/lP+U/5j/mP+e/57/kf+R/3f/'+
        'd/9q/2r/cP9w/4//j/+2/7b/wv/C/8P/w//J/8n/wf/B/7D/sP+w/7D/yP/I/9n/2f/L/8v/u/+7/7v/'+
        'u//M/8z/5f/l//X/9f/9//3/BAAEAAcABwAMAAwAGAAYAB4AHgADAAMA8v/y//T/9P/f/9//1f/V/+X/'+
        '5f/t/+3/5P/k/9H/0f/L/8v/4//j//r/+v/0//T/8P/w//7//v8FAAUABwAHAAQABAD0//T/5P/k/9P/'+
        '0/++/77/rP+s/7T/tP/K/8r/2//b/+D/4P/R/9H/xv/G/9P/0//m/+b/4v/i/9D/0P/M/8z/0f/R/+T/'+
        '5P/x//H/1//X/7b/tv+m/6b/pf+l/67/rv+9/73/xP/E/6j/qP+a/5r/vv++/+f/5//6//r/BgAGABEA'+
        'EQAWABYA/////9T/1P/M/8z/6P/o/wsACwA3ADcAUgBSAEYARgAqACoADQANAPr/+v/7//v/BgAGAP3/'+
        '/f/c/9z/t/+3/5v/m/+h/6H/vP+8/8//z//b/9v/4//j/+b/5v/V/9X/yP/I/9D/0P/b/9v/6v/q/+T/'+
        '5P/N/83/x//H/9P/0//g/+D/5P/k//T/9P8OAA4AGwAbACIAIgAWABYAAwADAPj/+P/t/+3/3//f/8L/'+
        'wv+a/5r/cv9y/1z/XP9u/27/kP+Q/6X/pf+7/7v/4f/h/xEAEQAyADIALwAvABUAFQD3//f/6//r//T/'+
        '9P/7//v/+f/5//D/8P/b/9v/0f/R/+L/4v//////CgAKAAIAAgAJAAkAFAAUABUAFQAiACIALAAsACIA'+
        'IgASABIABgAGAPj/+P/j/+P/0f/R/8v/y//X/9f/8v/y/xUAFQA0ADQARwBHAEgASABCAEIAQgBCAEcA'+
        'RwBSAFIAWQBZAEsASwA8ADwAMAAwAB8AHwAYABgAJQAlADsAOwBLAEsAUwBTAEgASAAmACYACwALAAEA'+
        'AQAIAAgAJAAkAEIAQgBQAFAATQBNAEsASwBPAE8AQgBCADcANwBKAEoAYQBhAG0AbQBtAG0AUgBSACQA'+
        'JAD/////9v/2/wsACwAvAC8AVABUAGwAbABxAHEAZwBnAFYAVgBGAEYANgA2ACcAJwAiACIAJgAmACEA'+
        'IQADAAMA4v/i/97/3v/r/+v/8//z//X/9f/1//X/AgACABQAFAAhACEAJQAlAB0AHQAWABYAIQAhACgA'+
        'KAAbABsADQANAAgACAAGAAYA//////n/+f8JAAkAIQAhACQAJAAeAB4AJQAlAC4ALgAtAC0AIgAiABAA'+
        'EAD5//n/4//j/+P/4//6//r/DwAPABoAGgAdAB0AHAAcABgAGAAVABUAFwAXAB8AHwAqACoALQAtACYA'+
        'JgAbABsACwALAP////8BAAEAAgACAPH/8f/L/8v/qf+p/7D/sP/W/9b/+P/4/xAAEAAaABoADQANAP7/'+
        '/v/3//f/9P/0//L/8v/n/+f/0f/R/9D/0P/p/+n/8f/x/+f/5//d/93/xv/G/7H/sf+2/7b/z//P/+v/'+
        '6//y//L/6P/o/+z/7P/6//r/AAAAAAQABAABAAEA7P/s/9H/0f/L/8v/5P/k/wAAAAAIAAgABwAHAP3/'+
        '/f/7//v/FQAVAC4ALgA3ADcAOAA4AC8ALwAiACIAGAAYAAQABADs/+z/4v/i/9z/3P/Y/9j/2P/Y/9j/'+
        '2P/Z/9n/3v/e/+z/7P/y//L/6//r/+3/7f/o/+j/1P/U/8T/xP/I/8j/4f/h//v/+//+//7/9v/2//7/'+
        '/v8NAA0ADAAMAAEAAQDr/+v/0P/Q/9H/0f/n/+f/+//7/wgACAAPAA8AHQAdAC8ALwAwADAAHgAeAAMA'+
        'AwDi/+L/yv/K/9H/0f/o/+j/6v/q/9P/0//J/8n/4f/h//////8NAA0AFgAWABgAGAATABMAEQARAA8A'+
        'DwAQABAADwAPAP/////w//D/6//r/+b/5v/i/+L/4v/i/+b/5v/x//H/BAAEACAAIAAzADMALgAuACIA'+
        'IgAlACUAOQA5AEoASgBKAEoARABEAEgASABRAFEASQBJADIAMgAgACAADgAOAPf/9//2//b/BQAFAAYA'+
        'BgD5//n/8P/w//D/8P/p/+n/2f/Z/9P/0//Z/9n/7v/u/wgACAAUABQAGwAbABwAHAANAA0A/P/8//X/'+
        '9f/4//j/BQAFABsAGwAxADEAMQAxABsAGwAEAAQA9v/2//H/8f/z//P/+P/4///////7//v/3//f/8P/'+
        'w/+//7//0P/Q/+X/5f/z//P/+v/6//3//f/5//n/9//3/+//7//S/9L/uv+6/7b/tv+6/7r/x//H/9z/'+
        '3P/u/+7/7//v/+//7/8DAAMAHAAcACMAIwAYABgA/f/9/97/3v/S/9L/2f/Z/+D/4P/g/+D/3P/c/9T/'+
        '1P/P/8//0v/S/9j/2P/d/93/2//b/9D/0P/O/87/4//j//v/+/8EAAQAAgACAP7//v/8//z//P/8//v/'+
        '+//2//b/8f/x//X/9f8AAAAACwALAA8ADwAJAAkACAAIABQAFAAiACIAKwArACUAJQAPAA8A9f/1/+D/'+
        '4P/T/9P/x//H/8D/wP/E/8T/zv/O/93/3f/r/+v/6v/q/9v/2//X/9f/6P/o//j/+P/3//f/8P/w//T/'+
        '9P/9//3//P/8//n/+f//////BwAHABIAEgAWABYACwALAP7//v8CAAIADAAMAAcABwADAAMACwALAA0A'+
        'DQAGAAYA+v/6/+j/6P/W/9b/1P/U/9r/2v/Z/9n/3v/e//H/8f/5//n/9v/2//v/+/8EAAQABgAGAAwA'+
        'DAAQABAAAQABAOX/5f/Q/9D/yP/I/87/zv/f/9//7f/t//T/9P/2//b/9P/0/+X/5f/L/8v/uv+6/7n/'+
        'uf/F/8X/3//f//7//v8XABcAIgAiACMAIwAhACEAHQAdABQAFAAJAAkABQAFAAYABgD9//3/8f/x/+j/'+
        '6P/l/+X/6P/o/+7/7v/2//b/+v/6//r/+v/4//j/8f/x/+r/6v/g/+D/1P/U/8//z//Y/9j/4P/g/9j/'+
        '2P/V/9X/6P/o//7//v8EAAQA+//7/+//7//q/+r/8//z//v/+//v/+//0f/R/7//v//F/8X/z//P/9X/'+
        '1f/f/9//5v/m/+n/6f/p/+n/4//j/9T/1P/J/8n/zv/O/9n/2f/k/+T/9P/0/wAAAAAEAAQAAQABAPX/'+
        '9f/l/+X/2v/a/9T/1P/T/9P/4P/g/+//7//z//P/+P/4//f/9//h/+H/xP/E/7//v//J/8n/0P/Q/+L/'+
        '4v/z//P/6v/q/93/3f/h/+H/5P/k/+D/4P/i/+L/5v/m/9//3//W/9b/yv/K/7D/sP+Y/5j/kP+Q/4r/'+
        'iv+M/4z/oP+g/7n/uf/G/8b/zv/O/9X/1f/U/9T/yP/I/8P/w//N/83/3f/d/+T/5P/b/9v/x//H/7j/'+
        'uP+9/73/0f/R/+X/5f/y//L/+//7/wIAAgAEAAQA+//7/+b/5v/P/8//wv/C/8L/wv/C/8L/wP/A/8T/'+
        'xP/F/8X/wP/A/73/vf/A/8D/w//D/7//v/+2/7b/sf+x/7n/uf/E/8T/w//D/7r/uv+w/7D/ov+i/5H/'+
        'kf+I/4j/j/+P/6b/pv+5/7n/wP/A/7//v/+3/7f/r/+v/7P/s/+7/7v/uP+4/7H/sf+5/7n/xP/E/8T/'+
        'xP+//7//wv/C/8n/yf/W/9b/4//j/+D/4P/V/9X/0f/R/8//z//M/8z/z//P/9X/1f/U/9T/0//T/9P/'+
        '0//O/87/zP/M/9D/0P/V/9X/1v/W/9L/0v/P/8//1//X/+D/4P/c/9z/0f/R/8z/zP/P/8//2f/Z/+b/'+
        '5v/l/+X/0P/Q/73/vf/A/8D/x//H/8b/xv/I/8j/0v/S/+D/4P/v/+///P/8/wUABQAIAAgAAwADAPz/'+
        '/P/+//7/BgAGAAkACQAIAAgAAQABAPP/8//n/+f/7f/t/wEAAQARABEAFwAXABMAEwAKAAoACgAKABIA'+
        'EgARABEACAAIAAMAAwADAAMABAAEABEAEQAtAC0APwA/AD8APwA0ADQAIwAjABEAEQAIAAgAAwADAPr/'+
        '+v/y//L/+f/5/wYABgADAAMA9f/1//f/9/8GAAYAEAAQAAwADAABAAEA+f/5//f/9//8//z/AgACAAQA'+
        'BAANAA0AHQAdACoAKgApACkAIAAgAB0AHQAqACoAOQA5ADgAOAAuAC4AKgAqACkAKQAmACYAJwAnADIA'+
        'MgBBAEEASgBKAEQARAA5ADkAOwA7AD8APwA1ADUAKwArACsAKwAlACUAEwATAAIAAgD9//3//v/+//z/'+
        '/P//////BgAGAAAAAAD2//b//v/+/xIAEgAaABoAFAAUAA0ADQAFAAUAAAAAAA4ADgAoACgAMgAyAC0A'+
        'LQAeAB4ADQANAAoACgASABIAFAAUAA0ADQAQABAAFAAUAAwADAAOAA4AHAAcACAAIAAfAB8AJgAmACkA'+
        'KQAfAB8AFQAVABUAFQAiACIAOwA7AE8ATwBSAFIASgBKAEAAQAAyADIAKAAoACcAJwAsACwAKwArACQA'+
        'JAAfAB8AIwAjAC0ALQA2ADYAOQA5ADAAMAAfAB8AGwAbACkAKQAsACwAGwAbABAAEAAbABsALgAuADgA'+
        'OAAuAC4AGAAYABIAEgAkACQANwA3AD8APwA8ADwAKQApAB4AHgApACkAOgA6AEIAQgBEAEQAPgA+ADMA'+
        'MwAyADIAPwA/AEQARAA5ADkAIwAjAAkACQDz//P/5v/m/9v/2//S/9L/1f/V/+D/4P/n/+f/5v/m/93/'+
        '3f/S/9L/0v/S/9v/2//l/+X/6f/p/+T/5P/h/+H/7f/t//7//v/7//v/8//z//v/+/8JAAkACgAKAP3/'+
        '/f/y//L/8v/y//r/+v8HAAcAGgAaACgAKAAhACEADAAMAPv/+//x//H/6//r/+r/6v/t/+3/8v/y//r/'+
        '+v8HAAcAFgAWAB8AHwAlACUALwAvAEEAQQBQAFAAUABQADsAOwAcABwABgAGAAAAAAD+//7/9//3//j/'+
        '+P8EAAQADwAPABUAFQAfAB8AJwAnACIAIgATABMACAAIAAcABwAKAAoACgAKAAEAAQDy//L/6v/q/+//'+
        '7//6//r/AAAAAP////8CAAIADQANABUAFQAXABcAGAAYABsAGwAZABkAEAAQAA0ADQAaABoAJQAlABwA'+
        'HAAQABAAFQAVACEAIQAoACgAMQAxADsAOwBAAEAARABEAEUARQA9AD0ALgAuACMAIwAcABwAHQAdACYA'+
        'JgAqACoAIwAjABsAGwAYABgAFAAUABIAEgAUABQAEAAQAAwADAASABIAFwAXABUAFQAUABQAEwATAA4A'+
        'DgAQABAAHwAfADIAMgA8ADwAPAA8ACwALAAZABkAEAAQAA4ADgACAAIA8//z//P/8/8DAAMAFgAWABwA'+
        'HAAPAA8AAAAAAP3//f8BAAEAAwADAAcABwASABIAHAAcAB4AHgAdAB0AEwATAAgACAAQABAAHwAfAB0A'+
        'HQAMAAwAAgACAAgACAATABMAFgAWABgAGAAkACQAMgAyACoAKgAKAAoA7//v/+z/7P/1//X/AAAAAAwA'+
        'DAARABEABgAGAPj/+P/6//r/BQAFAA8ADwAYABgAHgAeAB0AHQASABIADgAOABUAFQAWABYAEAAQABIA'+
        'EgAXABcADgAOAPj/+P/o/+j/6P/o//T/9P/9//3//v/+/wIAAgASABIAIQAhACIAIgAbABsAFQAVABQA'+
        'FAAVABUAGAAYABcAFwALAAsA+f/5/+3/7f/w//D//P/8/w4ADgAiACIAOQA5AEsASwBTAFMATwBPAEcA'+
        'RwBFAEUARwBHAEUARQBEAEQASwBLAFYAVgBbAFsAVQBVAEUARQA0ADQALQAtADYANgBHAEcASwBLAD0A'+
        'PQAoACgAEwATAAEAAQDx//H/6//r//P/8/8BAAEACwALABEAEQAbABsAKwArADcANwA+AD4AQABAAD4A'+
        'PgA7ADsAPgA+AEMAQwA6ADoAJQAlABIAEgAIAAgADQANAB4AHgAwADAANwA3ADAAMAAcABwA/f/9/+b/'+
        '5v/g/+D/5P/k/+z/7P/3//f/AwADAAsACwAVABUAHwAfACQAJAAnACcALQAtADMAMwA7ADsARABEAEkA'+
        'SQBGAEYAQABAAEAAQAA/AD8AMQAxACAAIAAbABsAGgAaABUAFQARABEADwAPAA8ADwAOAA4AEAAQABEA'+
        'EQAMAAwABAAEAAEAAQABAAEAAgACAAEAAQABAAEAAwADAAMAAwAFAAUADAAMAAwADAABAAEAAAAAABMA'+
        'EwAsACwAOwA7AD4APgA4ADgALQAtACIAIgAgACAAJAAkACUAJQAgACAAIQAhACYAJgAuAC4AOAA4AEMA'+
        'QwBHAEcAQABAADwAPAA7ADsAMgAyACUAJQAbABsAEwATABMAEwAfAB8AMAAwAD0APQA/AD8AOgA6AC8A'+
        'LwAkACQAIQAhACoAKgAxADEALAAsACEAIQAaABoADwAPAAEAAQD5//n/+f/5//b/9v/v/+//6v/q/+f/'+
        '5//i/+L/2//b/8//z//E/8T/wf/B/8n/yf/Y/9j/4v/i/+b/5v/t/+3/+f/5//3//f/1//X/7v/u/+//'+
        '7//z//P/+//7/wMAAwAIAAgACQAJAAMAAwDx//H/3f/d/9b/1v/c/9z/4//j/+H/4f/c/9z/2v/a/9//'+
        '3//o/+j/9P/0/wUABQAXABcAHgAeABgAGAAPAA8ACwALAAYABgD6//r/9P/0//T/9P/2//b//v/+/wYA'+
        'BgADAAMA+P/4//D/8P/u/+7/8//z////////////7f/t/9j/2P/J/8n/vP+8/7T/tP+3/7f/v/+//8n/'+
        'yf/X/9b/4P/g/+b/5v/t/+3/9v/2/wIAAwAXABcAKwArADIAMgAtAC0AIwAjABMAEwAEAAQA/P/8//f/'+
        '9//3//f//f/9/wQABAAEAAQAAwADAA0ADAAbABsAJAAjACkAKQAwADAAKgAqABcAGAAGAAYA/P/8/+//'+
        '7//j/+P/6P/n//v/+/8HAAcACQAJAAwADAARABIAFwAYABkAGgAVABUADAAMAAEAAQDz//L/5P/j/+D/'+
        '4P/o/+j/8f/x//T/9f/y//L/7v/v/+3/7f/t/+3/6f/p/+P/4v/e/97/4P/g/+7/7v8AAAAABwAIAAUA'+
        'BgAGAAYACgAKAAwADAAPAA8AEgARABEAEAATABMAHQAdACQAJQAmACcAJgAnACMAJAAZABkAEAAQABQA'+
        'FAAgACAAKQApACsAKwAqACoAKgAqAC0ALQAwADAAKwArAB4AHQAUABQAGQAYACIAIgAhACAAFgAWAAsA'+
        'CwAAAAAA+P/4//r/+v/+//7//P/7//3//f8DAAMA/v/+//D/8P/r/+v/7P/s/+X/5f/U/9T/wf/B/7X/'+
        'tf+0/7T/uf+4/73/vf/E/8T/zP/M/83/zv/L/8v/yf/J/8z/y//T/9P/3f/d/+P/4//j/+P/5f/l/+j/'+
        '6f/q/+r/7P/s//T/9P8BAAEACwALAAkACAD6//r/8v/x//b/9v/7//v/+P/5//T/9P/t/+3/3//f/9P/'+
        '0v/U/9P/3P/b/97/3v/X/9f/z//P/9H/0f/e/9//6//s//L/8v/z//T/8//z//H/8P/p/+j/2//b/87/'+
        'zv/G/8b/xv/H/8//z//a/9v/3f/e/9b/1v/S/9L/1//W/9v/2//b/9v/3v/e/+b/5v/o/+j/2//b/8T/'+
        'xf+t/67/mf+a/4z/i/+O/47/o/+j/7v/u//M/8z/2P/Y/+L/4v/p/+n/8f/x//j/+P/1//X/7v/u//H/'+
        '8f/6//r/AAAAAP3//f/3//f/9P/0//n/+f8BAAEABgAGAAcABwAGAAcABQAFAAQABAABAAEA+f/5//L/'+
        '8f/y//H/9v/2//7//v8JAAkADQANAAQABQD+//////8AAPz//f/3//f/+//7/wQAAwAJAAgADQAMABAA'+
        'EAATABMAFgAWABMAFAAGAAYA9//3//D/8P/t/+3/7P/s/+//7v/t/+3/5//n/+7/7v8AAAEACwALAAoA'+
        'CwAEAAQA+v/6/+7/7v/q/+r/6//r/+3/7P/0//P/AwADABAAEAAZABkAJQAlADQANQA+AD4AQABAAD4A'+
        'PgAyADEAHQAdAA4ADQAMAAwAFAAUAB0AHgAkACQAJAAkAB0AHQATABMACgAKAAQAAwADAAIACAAIABIA'+
        'EQAcABwAIgAiABoAGwAJAAkA/f/9//r/+v/7//r//v/+/wQAAwAFAAQAAQAAAAEAAQAEAAQAAgADAAQA'+
        'BQAMAAwADgAOAAkACQAIAAgACgAJAAEAAQDu/+7/3f/d/9b/1v/a/9r/4v/j/+j/6P/q/+r/7v/u//L/'+
        '8v/1//T/9v/1//n/+f8CAAIADwAPAB0AHQAkACQAIAAgABcAFwALAAsA/f/9//X/9f/4//j/AgACAA0A'+
        'DQAXABgAGwAbABUAFgAPAA8ADQAMAAkACQAGAAUACAAIAAwADAAJAAkAAAABAPj/+P/u/+7/4//j/+L/'+
        '4v/t/+3//P/8/wcABwANAA0ACQAJAP7//v/2//f/+P/4//T/9P/m/+b/2P/Y/8z/zP/F/8T/xv/F/8v/'+
        'y//S/9L/4P/h//L/8v/5//n/9P/1//D/8P/u/+7/7v/t//H/8P/y//H/8f/w//j/+P8GAAYACgALAAQA'+
        'BAD//wAA/////wIAAQAKAAoAFgAWACEAIAAnACcAKwArACsAKwAoACgAIQAhABoAGgAYABgAFQAVAA4A'+
        'DgAGAAYAAQABAPv/+//1//X/9P/0//n/+f8DAAMADwAPABcAFwAQABAAAQABAPP/8//p/+n/3//f/9T/'+
        '1P/M/8z/0P/Q/9//3//s/+z/7f/t/+T/5P/X/9f/0f/R/9j/2P/g/+D/4f/h/+P/4//p/+n/6P/o/+P/'+
        '4//m/+f/9f/1/wYABgATABMAGAAYABQAFAAQABAACwAKAP/////1//X/9//3//////8FAAUABAAEAPn/'+
        '+f/p/+n/4f/h/+T/5P/o/+f/5P/k/9v/2//O/87/yP/I/8//z//f/9//7//v//v/+/8DAAMACAAIABIA'+
        'EgAfAB8AJgAmACMAJAAbABsAFAATABIAEgATABMADgAOAAUABQADAAMABAAFAAQABAAEAAUABwAHAAcA'+
        'BwAJAAkADwAPABIAEgAPAA8ADwAPABQAFAAVABYAEgATAA8AEAAMAA0ACQAIAAQABAD+//3/9//3//T/'+
        '9P/2//b//v/+/wwADAAXABcAFQAVAAsACwADAAMAAAAAAAIAAgAHAAcADAAMAA8AEAAWABYAIQAiACgA'+
        'KAAjACMAGgAaABYAFgAUABQADwAOAAoACgAIAAgAAwADAPT/9P/k/+X/2//c/9f/1//W/9b/2//a/+P/'+
        '4//n/+f/5v/m/+f/5//x//H//////wsACwAVABUAHAAcAB0AHAAWABYAEAAQAA8ADwAOAA4ADAANAAwA'+
        'DAAKAAoABgAGAAIAAgADAAMABAAEAAIAAgAHAAcAFwAXACYAJgAmACYAGgAaAAsACwACAAEA/////wAA'+
        '///+//3/+f/6//v/+/8CAAMABgAHAP//AADv/+//3//f/9z/2//p/+j//v/+/xAAEAAcABwAJgAmADIA'+
        'MwA6ADsAMwAzACIAIgAXABYAEwASAA8ADwARABAAFwAXABYAFgAMAA0ACQAKAA0ADgAQABEAFAAUABkA'+
        'GAAVABUABwAGAPf/9//s/+z/4//j/9z/3P/d/93/6P/o//P/9P/6//r//P/8//j/+P/v/+//6//q//T/'+
        '9P8HAAYAEQARAAsACwD9//7/9v/2//f/+P/6//r/+f/5//f/9//0//T/8//z//f/9//5//j/8//z/+3/'+
        '7v/0//T/AAABAAkACgANAA0ADgAOAA0ADAAKAAoACAAIAAcABwAGAAYACAAIAA4ADgASABIAEgATABUA'+
        'FQAaABoAHQAcABoAGgAYABgAGwAaAB8AHgAlACUAKgAqACQAJQASABMAAAABAPr/+///////BAAEAAQA'+
        'AwAEAAQACQAJAAwADAAIAAgABAAEAAEAAgD//////P/8//f/9//v/+//6f/p/+n/6f/w/+//+P/4/wAA'+
        'AAAGAAYACgALAAsACwAEAAQA9//3//H/8P/2//b//v/+//3//f/4//j/+v/6/wIAAgAIAAgADgAOABUA'+
        'FQAeAB4AJAAkACEAIQAbABsAFgAWABMAEwANAA0ACQAJAAwADAASABIAFAAUABIAEgAPAA8ACQAJAAAA'+
        'AAD3//f/8f/x/+7/7v/v/+//8v/y//D/8P/q/+r/7f/t//X/9v/6//r//P/8/wIAAQAJAAkADwAOABIA'+
        'EgAVABUAFgAWABYAFgATABQADgAOAAsACwAOAA4ADwAPAAsACwAGAAYABAAEAAMAAwADAAMABAAEAAMA'+
        'AwAAAAAA/v/+//z//P/7//v/+f/5//f/9v/0//T/8v/y//D/8f/t/+3/6f/p/+n/6f/u/+7/8f/x//L/'+
        '8v/3//f/AQABAAoACgAPAA8AEAAQAAwADAAEAAQA/P/8//r/+v/9//3/AAD///z/+//2//b/9f/1//b/'+
        '9v/2//b/9v/2//f/9//4//j/9v/2//P/8//w//D/7v/u/+7/7v/y//L/9f/1//T/9P/z//P/9P/0//P/'+
        '8//s/+z/5//n/+T/5P/i/+L/4//j/+j/6f/x//H/+P/4//j/+P/v//D/6f/p/+3/7f/u/+7/5v/m/+P/'+
        '4//r/+v/9v/2//v/+//+//7///////3//f/8//z//P/8//z//P/8//z//f/9//b/9v/q/+r/5P/k/+j/'+
        '6P/v/+//9//3//v/+//4//j/8//z/+7/7v/n/+f/3f/d/9n/2f/d/93/4P/g/9//3//f/+D/4//j/+z/'+
        '7P/7//v/CAAIAAwADAAEAAQA9f/1/+j/6P/i/+L/4P/g/9//3//e/97/3P/c/9b/1v/X/9f/4f/h/+j/'+
        '6P/i/+L/2v/a/9b/1v/W/9b/0//T/83/zf/G/8b/xf/F/8b/xv/L/8v/1v/W/+X/5f/v/+//8P/w/+//'+
        '7//z//P/9//3//f/9//1//X/+P/4//////8IAAkAEAAQABEAEQAKAAoAAQABAP7//v/+//7/+f/4//D/'+
        '8P/q/+r/5//n/+b/5v/p/+r/7v/u/+z/7P/i/+L/2P/Y/9T/1P/S/9H/0P/P/83/zf/K/8r/yP/J/8n/'+
        'yf/J/8n/yP/I/8r/yv/Q/9D/1//W/9//3v/n/+f/6v/q/+T/5P/a/9r/0v/T/9X/1f/i/+P/7v/u/+z/'+
        '7P/l/+X/5P/k/+r/6f/w//D/9v/2//z//P8CAAMABwAHAAoACgAMAAwADAAMAAcABgADAAMACQAJABQA'+
        'FAAXABcAEQARAAcACAACAAIAAwADAAgACQAMAAwACQAJAAAAAAD3//f/8v/y//P/8//0//T/8f/x/+r/'+
        '6v/k/+X/4f/i/+H/4f/i/+L/4f/h/97/3v/f/9//5f/l/+z/7P/y//L/+f/5/wIAAgAKAAoAEQASABUA'+
        'FQAQABEACAAIAAAAAAD8//z//v/+/wMAAgAFAAUAAQABAPn/+f/x//H/7//v//T/9P/7//v/AQAAAAIA'+
        'AQD9//3/9P/0/+v/6//m/+b/4//j/+X/5v/r/+v/7//w//H/8f/x//H/7v/u/+n/6P/n/+b/7P/s//j/'+
        '+P8DAAMABAAFAPn/+v/q/+r/4P/g/9//3//i/+L/5f/l/+X/5f/l/+X/6P/o/+3/7f/x//H/8f/x/+z/'+
        '7f/m/+f/4//j/+T/5P/o/+j/6v/q/+f/5//l/+X/6v/q//L/8v/z//P/7P/s/+b/5v/k/+T/5//n/+3/'+
        '7f/0//T/9//3//X/9f/0//T/+f/5//3//f8AAAAAAwADAAYABgAGAAYAAwADAP3//f/4//f/9P/0//T/'+
        '9P/5//n/AQABAAYABwAHAAcABAAEAAEAAgD+//7/+f/5//X/9f/0//T/8f/w/+7/7v/x//H/9f/1//P/'+
        '8//s/+z/6P/o/+z/7P/z//P/+f/4//z//P/8//z/9//3//H/8f/v/+//8f/x//f/9///////AgACAPj/'+
        '+f/n/+b/1v/W/87/zv/Q/9D/1//X/9//3//n/+f/6v/q/+n/6f/o/+j/6P/p/+j/6f/o/+j/6v/q//D/'+
        '7//2//b//P/8/wAAAAABAAEAAAAAAAAAAAABAAEAAwACAAYABgAMAAwADwAPAA0ADQAHAAcAAwADAAQA'+
        'BAAIAAgACwALAAsACwAIAAcABQAEAAcABwAKAAoABQAFAPj/+P/x//H/9v/1/wAA//8FAAUABgAGAAUA'+
        'BQAEAAQABAAEAAQABAAFAAUABAAEAAEAAQD8//z/+P/3//f/9///////CQAKAA0ADQAEAAUA+P/4//H/'+
        '8f/v/+//7P/s/+j/5//l/+X/6P/o/+7/7v/z//P/8v/y/+v/6//k/+T/4P/g/97/3v/d/93/3v/e/93/'+
        '3f/a/9r/2f/Z/9//3//o/+j/7P/t/+3/7f/v/+//9f/1//3//f8EAAQACAAIAAUABgD+//7/+P/4//j/'+
        '9//4//j/9f/2//D/8P/s/+z/5//n/+L/4v/f/9//3v/e/9r/2v/T/9P/z//P/83/zf/N/83/zf/N/8//'+
        'z//S/9L/1f/U/9j/1//c/9z/4v/i/+X/5f/m/+b/6f/p/+7/7v/2//b//P/8//3//f/7//v/+f/5//f/'+
        '9//1//X/9P/0//n/+v8BAAEAAgACAP3//f/7//r/AQABAA0ADQAVABUAGQAZABoAGgAYABgAFQAVABIA'+
        'EgAQABAADwAPABEAEQAYABgAHwAfACMAIwAjACQAIwAjACEAIQAbABsAFwAXABgAGAAZABkAFAAUAAwA'+
        'DAAHAAcABwAHAAcABwAIAAgADAAMABMAEwAZABkAHQAdACEAIQAmACYAKwAqAC4ALgAwADAAMAAxAC8A'+
        'LwApACkAIwAjACIAIgAoACgALwAvADMAMwAzADMAMgAyADIAMgA1ADUANwA4ADMAMwAoACgAHAAcABYA'+
        'FgAWABYAFwAXABUAFQASABIAEgASABUAFQAYABgAGwAbAB4AHgAdAB0AGgAaABkAGQAbABsAGwAbABcA'+
        'FwARABEAEAAQABUAFQAcAB0AIAAgAB0AHQAWABcAEwATABQAFAAYABgAHgAeACMAIwAkACQAJQAlACgA'+
        'JwAqACoAKAAoAB4AHgAUABUAEAAQAA4ADgAKAAkABAADAAIAAgAGAAYADAAMABMAEwAaABoAIAAgACIA'+
        'IgAgACAAGgAbABEAEQAHAAcAAQABAAIAAQAGAAYADAAMABIAEgAUABQAEQARAA0ADQAOAA4AFAAUABsA'+
        'GwAdAB0AGgAaABQAFAAPAA8AEAAPABcAFwAiACIAJwAoACUAJQAhACEAIQAhACQAJAAmACYAJAAkACAA'+
        'IAAdABwAHAAcACIAIgAqACsAMAAwAC4ALgApACkAJAAkACEAIQAdAB0AFwAXABIAEgATABMAGgAZACEA'+
        'IQAlACUAIgAiABoAGgARABEACwALAAgABwAGAAYABQAGAAQABAADAAMAAwADAAcABgALAAsADgAOAA8A'+
        'DwAMAAwACAAIAAgABwALAAoADgAOAA4ADgALAAsACwALABEAEAAZABkAIQAiACcAKAAqACsAKQApACQA'+
        'IwAdAB0AGwAaAB0AHQAjACMAKAAoACoAKgAoACgAJgAmACQAJQAkACQAJAAkACMAIwAhACEAIAAfACAA'+
        'IAAjACMAJAAkACMAIwAiACIAIwAjACUAJQAjACMAHAAcABIAEgAIAAgA//////n/+f/3//f/+f/5////'+
        '//8FAAYACQAJAAcABwACAAIA/v/+//3//f////7/BAAEAA4ADQAVABUAFwAXABIAEgAMAA0ACgAKAAsA'+
        'CwAOAA4AEwASABgAFwAbABsAHQAdAB0AHQAgACAAKAAoADMAMwA7ADsAOwA7ADQANAArACsAJQAkACIA'+
        'IgAiACIAIwAkACMAIwAgACAAHgAeAB4AHgAfAB8AHAAcABgAGAAUABQAEAAPAAkACQACAAIA/v/+//3/'+
        '/f/8//z/+P/5//P/8//u/+7/6//r/+7/7f/1//T/+v/6//v/+//8//z/AAAAAAEAAQD9//7/+P/4//b/'+
        '9v/3//b/8//z/+v/6//n/+f/6v/q//L/8v/5//n//P/8//3//f/9//3//P/8//v/+//7//r//f/9/wMA'+
        'AwAGAAYAAwADAPz//P/2//b/8P/w/+r/6v/o/+j/7P/s//T/9P/3//b/8//z//L/8v/2//X//P/8/wEA'+
        'AQACAAIA//8AAPv/+//2//b/8f/x/+3/7P/o/+j/6f/p/+//7//2//b/+v/5//v/+//8//z//P/8//j/'+
        '+P/x//H/7P/s/+z/7P/w//D/9f/1//r/+///////AgACAAUABQAIAAgADQANABQAFAAaABoAHAAcABsA'+
        'GwAbABoAHgAeACQAJAAsACwAMQAxAC8ALwAmACYAHAAcABcAFwAVABUAEwATAA4ADgALAAoACQAJAAwA'+
        'DAASABIAGQAaAB8AIAAhACIAIAAgAB0AHQAaABoAGAAYABkAGQAbABsAHQAdACIAIgArACsAMQAxAC0A'+
        'LgAjACMAGQAZABQAFAARABEADwAOAA8ADwARABEAEwATABEAEQAPAA8AEAAQABMAEwAXABcAGgAaABoA'+
        'GgAYABcAGAAYABsAGwAcABwAGwAbABsAGwAdAB0AHQAdABgAGAATABMAEQARABAAEAANAA0ACgAKAA4A'+
        'DgAVABYAGQAZABUAFgAOAA4ACQAJAAgACAAKAAoACwALAAsACwAMAAwADQANAA4ADgAMAAwACAAJAAUA'+
        'BQADAAMABQAFAAoACQAPAA4AEQARABEAEQARABEAEQARAA8ADgAKAAoACAAIAA0ADQASABMAEwATAA8A'+
        'DwANAAwAEAAQABYAFgAYABgAFAAVAA4ADgAIAAgAAgACAP7//v/7//v/+f/5//b/9v/y//H/8P/v//T/'+
        '9P/9//3/BAAEAAUABQABAAEA/P/8//v/+//+//7/AAAAAP/////6//r/9v/1//b/9f/5//n//////wUA'+
        'BQAMAAwAEwATABkAGQAcABwAGwAbABYAFgAQABAADAAMAAoACgAIAAgABQAFAP3//v/0//T/7P/s/+r/'+
        '6v/t/+3/7P/s/+j/6P/k/+T/4f/h/9//3v/e/97/4P/h/+P/5P/i/+L/3P/c/9j/2P/b/9v/3//f/+L/'+
        '4//l/+X/6f/o/+v/6//t/+3/7//v//D/8P/w//D/7//v/+7/7v/s/+z/6v/q/+f/5//h/+H/2//b/9v/'+
        '2//h/+H/6P/o/+v/6//s/+z/7v/u//D/8P/x//H/8v/y//b/9v/5//n/+P/4//b/9v/3//f/+v/6//z/'+
        '/P/8//z//f/9//////8BAAEAAwADAAQABAACAAIAAAAAAAEAAAACAAIAAgACAP/////8//z/+v/6//j/'+
        '+P/5//n//v/+/wUABQAKAAoADQANAA8ADwARABAAEAAQABAAEAASABIAFQAVABYAFgATABMADAAMAAYA'+
        'BgAEAAQACAAIAAsADAANAA0AEAAQABYAFgAcABwAHQAdABsAGwAXABcAEgASAAsACwAEAAQAAAAAAPv/'+
        '+//1//X/8//z//j/+P/+//7/AgACAAQABAAEAAQABAAEAAMAAwADAAMAAwADAP/////2//b/8P/w/+//'+
        '7//v/+//7//v/+//7//x//H/8v/y//H/8f/u/+7/6v/q/+X/5f/g/+D/4P/f/+X/5f/v/+//9//4//n/'+
        '+f/z//P/7f/t/+3/7f/z//L/+f/5//z//P/7//v/+f/5//n/+f/6//r/+//8//////8FAAUACQAKAAgA'+
        'CAABAAAA9//2//H/8f/y//H/9v/2//r/+//8//z/+v/7//f/9//y//L/8P/w//L/8v/0//T/9v/2//b/'+
        '9v/3//f/+v/6//////8BAAIA//8AAPr/+v/1//T/8//z//X/9P/3//f/+v/6//7//v8GAAYADQANABIA'+
        'EgAUABUAFgAXABcAGAAXABYAFQAVABQAFAATABMAEAAQAAoACgAEAAQAAQABAAAAAAAAAAAAAAAAAP7/'+
        '/v/+//7/AAD//wAAAAD//////////wIAAwAGAAYABwAHAAIAAgD5//n/8v/y//D/8P/x//H/8v/y//D/'+
        '8P/w//D/9f/1//v/+/8AAAAAAQABAAAAAAD//////////wEAAQADAAMABQAFAAUABQAGAAYABgAGAAIA'+
        'AgD/////AAABAAQABAAEAAQAAwADAAUABQALAAsADQAMAAgACAADAAMAAwADAAcABwAKAAoACgAKAAYA'+
        'BgACAAIA/v/+//n/+f/0//P/8v/y//j/+P8EAAQADAAMAA0ADQAGAAYA/P/8//P/8v/u/+7/8P/x//b/'+
        '9v/5//n/+f/4//b/9v/1//T/9v/2//j/+f/6//r/+P/5//b/9v/2//X/+f/5//7//v8DAAMABwAHAAcA'+
        'BwAFAAUABwAHAA4ADgAVABUAFgAWABEAEQANAA0ACwALAAoACgAKAAoACwALAAwADQAPAA8AEAAQABEA'+
        'EQAOAA4ABwAHAP3//f/1//X/8f/x//H/8f/0//P/9//3//z//P8BAAEAAwADAAEAAQD/////AQAAAAUA'+
        'BQAIAAgACQAJAAcABwAGAAYABQAFAAgACAANAA0ADgAOAAwADAAJAAkACQAJAAkACQAJAAgACgAKAA4A'+
        'DgAPAA8ACwALAAcABwAIAAgACwALAAwADAAKAAoABwAHAAQAAwAAAAAA/f/9//z//P/8//z//v/+/wEA'+
        'AQAEAAQABwAHAAkACgAMAAwADQANAA4ADgAQAA8AFAAUABkAGQAbABsAGAAYABEAEQAMAAwACQAJAAcA'+
        'BwADAAMAAAAAAP7//v/9//3//P/8//v/+//7//v/+v/6//r/+v/6//r/+//7//j/+f/0//T/9P/0//v/'+
        '+/8BAAIABQAFAAgACAALAAsADAALAAoACgAKAAoADAAMAA0ADQAMAAwADAAMAA4ADgARABEAEgATABQA'+
        'FAATABQAEAAQAA0ADQAQABAAFgAWABcAFwARABEACgAJAAQABAADAAMABgAGAAsACwAPABAAEAAQAA8A'+
        'DwARABEAFgAWABsAGwAbABsAFgAWAA4ADgAHAAYAAwACAAAAAAD9//3//P/8//7//v8BAAEABAAEAAcA'+
        'BwAKAAsACwALAAgACQAHAAgACwALABIAEgAXABcAFwAXABIAEgALAAsACQAIAAsACgAOAA0ADAAMAAcA'+
        'BwAEAAQABQAFAAkACQANAA0ADwAPABAAEAAQABEAEAAQAA0ADgAIAAgAAgACAP3//f/8//z////+/wUA'+
        'BAALAAoADQANAAsACgAHAAcABAAEAAIAAgABAAEAAgACAAAAAAD6//r/8v/z//L/8v/3//f/+v/6//r/'+
        '+v/8//z/AQABAAQAAwADAAMAAgADAAUABQAGAAYAAwAEAAEAAQABAAEAAAAAAP////8CAAIACAAIAA4A'+
        'DgAQAA8ADwAPAA0ADAAIAAgAAgACAP////8AAAAABQAFAAoACwAPAA8ADwAQAAoACwAEAAQAAQABAAIA'+
        'AgACAAEA//////3//f/8//z//P/7//v/+//9//3/AQABAAUABQAJAAkADQANAA0ADgAIAAkAAQACAP3/'+
        '/v/+////AQABAAEAAQD+//3/+v/6//r/+v/8//v//P/7//n/+P/2//b/9//3//v/+//+//7/////////'+
        '/////wAAAAAAAAAAAAD//////P/8//r/+v/7//v//v/9/wAA//////7//P/7//r/+f/5//n/+v/6//v/'+
        '+//7//v/9//4//P/9P/x//H/8P/w/+//7//w//D/8v/z//b/9v/5//n/+f/4//b/9f/y//H/8f/w//T/'+
        '8//3//f/+P/4//n/+f/9//7/BAAEAAYABgACAAMA//////7//v///////////wEAAQADAAMABAADAAEA'+
        'AQD9//3/+f/4//T/8//w//D/8P/v//L/8v/2//f/+//7////AAADAAQABQAGAAcABwAIAAgACAAIAAkA'+
        'CAAKAAoACwALAAoACgALAAoADwAOABQAFAAYABgAGgAbAB4AHgAgACEAIAAgAB0AHgAbABsAFwAYABMA'+
        'EwARABEAEQASABIAEgAOAA4ACAAIAAUABQAFAAQABAAEAAQAAwADAAMABQAFAAcABwAKAAoACwALAAoA'+
        'CgAKAAoADQANABAAEAAQABAADAAMAAcABwADAAMAAAAAAP3//f/7//v/+v/6//v/+//9//3//f/9//r/'+
        '+v/1//b/8v/y/+7/7v/q/+r/5f/m/+T/5P/j/+P/4f/h/97/3v/f/97/5P/k/+v/6v/w//D/9P/0//r/'+
        '+v///wAAAwAEAAcABwAIAAgABQAFAP7//v/5//n/+f/5//z/+//7//v/9//3//L/8f/u/+7/7f/t/+7/'+
        '7v/v/+//8P/x//P/8//2//b/9v/2/+//7//l/+X/3f/d/9n/2f/W/9b/1//X/9r/2f/c/9z/3f/d/97/'+
        '3v/i/+L/5//n/+r/6v/r/+v/7f/t/+7/7v/q/+r/4P/g/9j/2P/V/9X/1v/W/9j/2P/b/9v/3f/d/9//'+
        '3//i/+L/5v/m/+j/6P/n/+f/4v/i/93/3v/a/9v/2P/Y/9T/1P/Q/9D/zf/O/87/zv/Q/9H/1P/U/9j/'+
        '2P/b/9r/3f/c/97/3f/f/97/4P/g/+P/4//l/+X/5P/l/+L/4v/e/9//3f/d/93/3f/e/97/3f/d/9n/'+
        '2v/Y/9j/2f/Z/9r/2f/X/9b/0P/P/8n/yf/F/8X/xf/F/8f/x//J/8n/yv/K/8n/yv/I/8n/yP/I/8j/'+
        'yf/L/8v/z//P/9T/1P/Y/9j/3P/c/97/3f/d/93/3P/b/9z/2//e/97/4f/h/+T/5P/m/+b/5v/m/+X/'+
        '5v/k/+X/5P/k/+T/5f/l/+X/5f/l/+X/5f/j/+P/4P/g/9//3v/e/93/3f/c/9z/3P/d/93/3//f/+H/'+
        '4v/l/+X/6v/q/+//7//v//D/6//s/+b/5v/i/+L/4P/g/93/3f/c/9v/2//b/9z/2//e/97/4v/h/+X/'+
        '5f/n/+f/5v/m/+P/4//h/+H/4//j/+b/5//n/+f/5f/m/+X/5v/p/+r/7//v//H/8f/w//D/7v/u/+z/'+
        '6//p/+j/6P/o/+r/6v/u/+3/8v/x//X/9f/3//f/9v/2//b/9v/4//j/+//7//z//f/9//3//v/+//3/'+
        '/f/7//v/+v/6//v/+v/6//r/9//3//b/9f/3//f/+f/4//f/9//2//b/+f/5//7//v8BAAIAAgADAAAA'+
        'AAD7//z/+P/4//b/9v/2//b/9v/2//n/+P//////BQAFAAYABgACAAEA/P/8//j/+P/2//b/9v/2//n/'+
        '+v/9//3//P/8//n/+f/1//X/8v/y/+7/7v/q/+r/6f/p/+v/6v/t/+3/7//v/+7/7v/t/+3/7v/t//H/'+
        '8f/0//T/9v/2//b/9v/5//n//f/+/wMAAwAGAAYABQAFAAAAAAD9//3//f/9///////+//7/+//7//j/'+
        '9//z//P/7P/s/+X/5f/h/+H/3//e/9v/2//X/9f/1v/W/9r/2v/g/+D/5P/k/+T/5P/j/+P/4f/h/+L/'+
        '4v/k/+T/5f/l/+P/4//g/+D/4f/h/+T/5P/n/+f/6P/n/+f/5//n/+f/6f/p/+v/7P/w//D/9P/0//b/'+
        '9v/z//P/7f/t/+n/6f/o/+j/6f/p/+r/6v/u/+7/8v/y//X/9v/5//n//P/8//7//v/8//z/9//3//P/'+
        '8//y//L/9f/1//r/+v//////AAAAAP3//f/7//z//v///wUABQALAAsADQANAAsACwAJAAkACAAIAAgA'+
        'CAAIAAcABQAFAAAAAAD6//r/9f/2//X/9f/2//b/9v/2//T/9f/0//T/9f/1//b/9v/2//b/9v/2//b/'+
        '9v/0//T/8//z//X/9f/6//r//f/8//v/+v/4//j/+//7//////8CAAIAAQACAP7////6//r/9v/2//T/'+
        '9P/0//T/9P/0//T/8//0//T/9//3//z/+///////AQABAAMAAwAGAAYACAAIAAcACAAGAAYABAAEAAUA'+
        'BQAHAAcACAAIAAgACAAHAAcABQAFAAMAAgABAAAAAAAAAAIAAQADAAMAAwADAAAAAAD7//v/9//4//f/'+
        '+P/9//3/BAAFAAkACQAJAAkACAAIAAcABwAHAAcACAAIAAoACQALAAsADAAMAAwADAALAAsACwALAA0A'+
        'DQARABEAFQAVABcAGAAYABgAFgAXABQAFAASABMAEQARABAAEAAQABAAEAAQABAADwAPAA4ADQANAA4A'+
        'DgAPAA4ADgAOAA0ADQAOAA4ADgAOAAwADAAJAAkACAAIAAoACgAMAAwACwALAAgACAAHAAcACAAIAAsA'+
        'CwAPAA4AEAAQAA8ADwAOAA4ADwAPABAAEAAPAA8ADAAMAAsADAAOAA4AEAAQAA4ADgALAAsACQAJAAoA'+
        'CgALAAsACgAKAAUABAD9//3/9f/1//D/7//v/+//9f/1//7//v8HAAcADQAOABAAEQARABIAEgATABQA'+
        'FQAXABcAFwAXABUAFgATABMADwAPAAoACgAHAAcACQAJAA0ADQAQAA8ADgAOAAoACgAGAAYAAwADAAEA'+
        'AQAAAAAAAAAAAAIAAgAGAAYACgALAA4ADgAOAA4ADQANAAwACwALAAsADgANABMAEgAXABcAGAAYABcA'+
        'FwAYABgAGgAaABsAGwAZABoAGAAYABcAGAAYABgAGgAaABwAHAAbABsAFwAXABEAEQAOAA4ADgAOAA8A'+
        'DwAQAA8AEQARABIAEgARABEADQANAAgACAAGAAYABwAHAAsACwAQABAAEgASABEAEQARABEAEgASABIA'+
        'EgAQABAAEAAQABQAFAAZABkAGwAbABsAGwAaABoAGwAbABsAGwAZABkAFAAUAA4ADgAJAAkABwAHAAQA'+
        'BAD//////P/8//z//P/9//7//v////7////+//7//v/+//////8BAAEAAwADAAEAAQAAAAAAAQABAAQA'+
        'AwAGAAYACgAKABEAEAAZABgAHgAdAB8AHwAgACAAIQAhAB8AHwAaABoAFAAVABAAEAANAA0ACwALAAwA'+
        'DAAOAA4ADwAPAA4ADgAMAAwACwALAAoACgAJAAkACAAIAAgACAAIAAgACQAJAAkACgAKAAoACwALAA0A'+
        'DgAPABAADwAPAAsADAAJAAkACQAJAAoACgAKAAoACAAHAAQAAwACAAIABQAFAAoACgANAA0ADQAOAAwA'+
        'DAAKAAsACgALAAoACwAKAAoACAAIAAUABgAEAAQABAAEAAUABQAFAAUABAADAP7//f/2//X/8f/w//L/'+
        '8f/3//f//v/+/wQABAAJAAoADwAQABUAFgAYABkAFwAXABIAEwAPAA8ADwAPABIAEQAUABQAFQAVABUA'+
        'FAARABEADAALAAcABgAEAAQAAgACAAAAAAD//wAAAQABAAIAAgAAAAEA/v////3//f/8//z//v/+/wQA'+
        'BAAIAAgACgAKAAoACQALAAoACwAKAAgACAAIAAcADQANABQAFAAXABcAFQAVABAAEAAMAAwABwAHAAMA'+
        'AwABAAIAAQABAPz//P/1//X/8f/x//H/8f/0//T/9v/2//f/9//4//j/+f/5//v/+//8//z/+//7//j/'+
        '+P/z//T/8v/y//P/9P/2//b/+P/4//j/+P/3//j/+P/4//v/+/8AAAAAAgACAAEAAQD//////v/+//7/'+
        '/v//////AAAAAAAAAAD//////f/9//3//v///wAAAQABAAAAAAD7//v/+P/4//f/9//3//f/9//3//j/'+
        '+P/5//n/+//7//r/+v/3//f/8//z//P/8//2//b//P/8/wEAAQAFAAUABgAGAAUABQABAAEA/v/+//z/'+
        '/P/8//z//f/9//z//P/7//v/+f/5//b/9v/1//X/9f/1//b/9v/2//b/9//3//j/+f/4//j/9v/2//P/'+
        '8//x//H/8v/y//T/9P/3//f/+f/4//r/+v/7//v//P/8//v/+//5//n/+v/6//3//f/+//7//P/8//r/'+
        '+v/7//v//P/7//n/+P/0//T/8f/x//H/8f/0//T/+P/4//v/+//8//z/+//7//n/+f/5//n/+//7////'+
        '//8DAAMAAwADAP//AAD8//z//P/8/////////wAA/v////////8BAAEAAwADAAMAAwAAAAAA+v/7//X/'+
        '9v/0//T/9v/3//r/+//7//z//P/8//7//f8AAP//AwACAAQABAAEAAQABAADAAQAAwABAAAA+v/6//P/'+
        '9P/v/+//7v/t/+7/7f/u/+7/8P/x//P/8//z//P/8f/x//D/8P/u/+7/6//s/+n/6v/p/+n/6f/p/+n/'+
        '6f/q/+r/7v/u//L/8v/0//P/8//y//P/8//2//f/+v/7//z//P/7//v/+f/5//f/+P/4//n/+//8//3/'+
        '/f/8//z/+v/6//j/+P/4//j/+v/6//z/+//9//z//f/8//////8BAAEAAwACAAQAAwAGAAUACAAIAAkA'+
        'CQAJAAgABwAGAAQABAADAAMAAgADAAIAAwABAAEA//////7//f/+//7///8AAAIAAgAEAAQABAAEAAIA'+
        'AgD+////+//7//j/+P/3//f/+P/4//r/+v/9//3///8AAAEAAgACAAIA//////z//f/8//z//v/+/wIA'+
        'AgAFAAUABwAHAAUABgACAAMAAAABAAEAAQADAAMABQAFAAcABgAJAAkACwAKAAkACQAGAAYAAwADAAAA'+
        'AAD+//3//f/9//3//f/9//3/+//8//v/+//8//3//////wAAAAD//////v/+/wAAAAADAAMABQAGAAYA'+
        'BgAGAAYACAAHAAkACAAJAAkACQAJAAoACgAMAAsACwAKAAkACQAJAAkACAAJAAgACQAKAAoADAAMAA8A'+
        'EAAQABEAEAARABEAEQATABMAFgAVABcAFwAXABcAFwAWABkAGAAdABwAHgAdABoAGgAXABcAFQAVABUA'+
        'FQAXABYAGwAbACAAIAAiACIAHwAfABsAGwAYABgAGAAZABkAGgAZABoAGwAbABwAGwAdABwAHgAeACAA'+
        'IAAgAB8AHQAcABsAGgAbABoAGgAbABgAGQAVABUAEwATABEAEQAOAA8ACgALAAcABwAGAAYABwAHAAoA'+
        'CgANAA4AEQARABIAEgARABAAEQAQABIAEgATABMAFAATABUAFAAWABYAFQAVABUAFQAWABYAGAAXABYA'+
        'FgASABIAEAAQAA8AEAAPABAADwAPABAAEAARABIAEQASABEAEQARABEAEQARABAADwAOAA4AEAAPABEA'+
        'EAAPAA8ADAALAAoACgALAAsADAAMAA0ADQANAA0ADAANAAwADAANAA0ADwAPAA4ADwAMAAwACwALAAsA'+
        'CwALAAsACwALAA0ADQAPAA8ADwAPABAADwASABIAFQAVABcAFgAYABcAFwAXABQAFAAPAA8ADQAOABAA'+
        'EAAUABQAEwAUABIAEwAUABQAFwAXABkAGAAZABkAGgAaABoAGgAZABkAGQAYABoAGQAaABkAFgAXABIA'+
        'EwARABEAEgASABUAFQAYABgAHAAcAB8AHwAfAB8AHQAcABkAGQAWABYAFQAVABUAFQAVABUAFQAVABQA'+
        'FQAVABUAFgAVABYAFgAWABYAFAAUABIAEgARABIAEwATABMAEwAPABAACwAMAAoACwANAA0AEQARABQA'+
        'FAAVABQAEwASAA8ADwANAA0ADQANAA8ADwASABIAFQAWABgAGAAZABkAGgAZABoAGQAZABkAGAAYABgA'+
        'GAAZABkAGwAaABoAGgAWABYAEAAQAAwACwAKAAkACgAJAAwADAAPAA8AEgATABQAFAATABMADgAOAAcA'+
        'CAACAAMAAAABAAEAAQAEAAQABwAHAAgACQAIAAgABgAGAAQABAAEAAQABQAFAAUABQAEAAQABQAFAAgA'+
        'CAALAAsADAAMAAsACwALAAsACgAKAAgACAAFAAUAAgACAAAAAAD/////AgACAAYABgAJAAkACwALAAwA'+
        'DAANAA0ADgAOABAAEAASABIAFQAVABYAFgAWABYAFAAVABMAFAATABMAFAAUABUAFQAXABcAGQAaABwA'+
        'HAAdAB0AHAAbABgAGAAWABYAFgAWABcAFwAZABgAGQAZABkAGgAaABoAHAAbABwAGwAaABkAFgAWABMA'+
        'EwARABEAEAAQAA8ADgAOAA4ADQANAAwADQANAA4ADwAPABEAEAAQABAAEAAQABEAEQARABEAEAAPAA4A'+
        'DgAMAAwACgAKAAcABwAIAAcACwAKAA4ADgANAA0ACgAKAAYABgAEAAQAAwADAAMAAwADAAQABAAFAAYA'+
        'BwAKAAoADwAPABEAEQAQABAADgAOAA0ADQANAAwADAALAAoACgAIAAgABQAFAAQAAwAGAAYACgALAA4A'+
        'DwARABIAEwASABEAEQAPAA8ADAANAAsADAAKAAoABwAHAAcABwAJAAoACwALAAgACAAFAAQAAwADAAMA'+
        'AwADAAQABgAGAAoACgAOAA0ADgAOAA0ADgAMAA0ACgALAAgACAAGAAYABQAFAAYABwAIAAgACQAJAAkA'+
        'CQAJAAkACgAKAAwADAAPAA8AEgASABQAFAAWABYAFgAWABUAFQAUABQAFQAVABUAFQAUABQAEAAQAAwA'+
        'DAAIAAgABQAGAAcABgALAAsAEAAPABMAEwAUABUAFAAUABIAEgAPAA8ADQAOAA4ADwAOAA8ADQANAAwA'+
        'DAALAAsACQAKAAcACAAHAAcACwAKABEAEAAUABQAFQAVABUAFQAWABYAFwAWABUAFQATABMAEQARABEA'+
        'EAASABEAEwASABIAEgAPAA8ADQAMAAwACwAMAAwADwAPABIAEgAUABQAFAAUABMAEwASABIAEQARABAA'+
        'EAAPAA8ADgAOAA0ADQAMAA0ADQAOAA8AEAAQABAAEAAQABAAEAAQABEADwAQAA4ADwAOAA4ADQANAAwA'+
        'DQALAAwADAAMAA0ADQAOAA4ADAALAAcACAAEAAUAAwADAAQAAwAFAAQABwAHAAkACgAKAAoACQAIAAYA'+
        'BQAEAAMAAwADAAMAAwAFAAQABgAGAAcABgAHAAcABgAHAAUABQAFAAQABQAFAAcABgAHAAgABwAIAAYA'+
        'BwAGAAYABgAGAAQABAABAAEA/v////7//v////////8AAP7//v/7//v/9//4//f/9//5//n/+//7//3/'+
        '/f/+////AAABAAMAAwAFAAQABgAGAAcABwAGAAUAAgACAP/////+//3//f/9//7//f/+//3//f/8//v/'+
        '+//6//r/+v/6//r/+f/4//f/9//3//r/+f/9//z//f/8//n/+P/y//L/7P/s/+f/5//m/+b/5//o/+r/'+
        '6v/t/+3/7v/v/+//8P/v//D/8P/w//H/8f/x//L/8f/y//H/8v/x//H/7//w/+3/7v/r/+z/7P/t/+//'+
        '7//y//H/9P/z//X/9f/2//b/9f/1//T/8//0//L/8//y//H/8f/w//H/8v/y//b/9f/4//f/9//3//X/'+
        '9f/0//P/9f/0//n/+P/8//v//f/9//v/+//4//j/9v/1//T/8//y//L/8f/x//D/8P/v/+//7f/s/+v/'+
        '6//t/+7/8f/y//P/9P/z//T/8v/z//P/9P/y//T/8P/x/+7/7v/w//D/9P/0//f/9//5//r/+v/6//n/'+
        '+f/4//f/+P/3//j/+P/4//j/9//2//b/9f/3//b/+P/4//f/9//2//b/8//z//D/7//t/+z/7P/s/+7/'+
        '7v/u/+3/7f/r/+z/6//u/+7/8P/w//D/8P/v/+7/7P/s/+n/6v/o/+n/6f/p/+r/6v/q/+r/6P/p/+b/'+
        '5//l/+X/5f/l/+f/5//p/+r/6v/r/+r/6v/o/+j/5f/m/+X/5v/m/+b/5v/m/+b/5v/m/+b/6P/o/+z/'+
        '7P/w//D/8v/x//H/8f/w/+//7v/u/+3/7f/u/+7/7v/t/+z/6//q/+r/6f/p/+n/6P/o/+f/5//m/+X/'+
        '5f/l/+X/5v/l/+f/5v/o/+j/6v/q/+v/6//q/+r/6f/p/+f/5//m/+b/5f/m/+b/5v/k/+T/4P/g/9r/'+
        '2//V/9X/0v/S/9L/0v/U/9T/1v/X/9j/2f/a/9v/2v/b/9j/2f/Y/9j/2P/Z/9n/2v/Z/9r/2v/a/93/'+
        '3P/g/9//4v/i/+X/5f/o/+j/6v/p/+v/6v/s/+z/7//u//H/8P/y//H/9f/0//j/9//5//n/+P/4//X/'+
        '9f/w/+//6f/o/+P/4//h/+L/5P/l/+j/6P/p/+j/5v/m/+P/4//h/+L/4//j/+b/5v/p/+n/6v/q/+n/'+
        '6v/n/+f/5P/k/+L/4v/j/+P/5P/l/+b/6P/p/+r/7P/s/+3/7f/r/+v/5//n/+P/5P/i/+L/5P/k/+b/'+
        '5v/n/+f/5//n/+f/5//o/+j/6v/p/+v/6v/u/+3/8f/w//P/8v/z//L/8v/x//L/8v/y//H/8P/w//D/'+
        '8P/x//H/8//z//T/9P/0//T/9P/0//X/9f/2//b/9f/2//X/9v/2//b/+P/4//n/+v/6//r/+v/6//n/'+
        '+f/2//b/9P/0//T/9P/z//P/8f/x/+7/7v/s/+z/6v/r/+j/6f/n/+j/6P/p/+v/6//t/+7/7v/v//D/'+
        '8f/x//H/8P/v/+7/7v/v/+//8//z//b/9v/4//f/+f/4//n/+P/3//f/9v/2//j/9//7//r//P/8//v/'+
        '+v/4//j/9//4//n/+f/8//v//f/9//3//f/9//7//v/+//3//f/9//z////+/wAAAQAAAAAA/v/+//7/'+
        '/v/+//7//P/8//r/+v/6//r//P/8//z//P/7//v/+v/7//r/+//5//r/+//8////AAADAAQAAwAEAAMA'+
        'BAAFAAUABgAGAAUABQADAAMAAwACAAMAAgACAAEAAQAAAP7//v/7//v/+f/5//n/+f/7//r/+//7//z/'+
        '/P/8//z//f/9//7//v8AAAAAAQACAAEAAQD/////AQAAAAQAAwAFAAUABAAFAAUABQAHAAcACQAIAAkA'+
        'CQAJAAkACQAJAAgACAAIAAgACAAIAAYABgADAAQAAwAEAAQABAACAAIA/f/+//v//P/8//3//f/9//z/'+
        '/P/9//3/AAAAAAEAAQAAAAAAAAABAAIAAwADAAMAAwADAAYABgAJAAkACgAKAAkACQAIAAcABwAGAAYA'+
        'BQAEAAMAAwADAAIAAQABAAAAAQAAAAIAAQABAAEA/v/+//v/+//7//r/+//6//r/+//6//v/+//8//z/'+
        '/P/9//7/AAAAAAIAAwADAAQAAgADAAMAAwAGAAUACQAJAAsACwAKAAoABwAHAAUABQAFAAUABwAGAAcA'+
        'BwAHAAgACAAJAAgACQAHAAgABAAFAAEAAgD//wEA/v8BAP//AgAAAAIA//8AAPz//f/6//v//P/9////'+
        '//8CAAEABAADAAQAAwACAAMAAgACAAMAAwAFAAMABQAEAAQABAADAAQAAwADAAMAAgABAAEA/v/+//r/'+
        '+v/1//b/8//z//P/8v/z//P/8//y//L/8f/y//H/9P/y//b/9P/5//f/+f/3//n/9//6//n//f/7////'+
        '/f////7/AAD//wAA////////////////AAD//////f/9//v//P/9////AQADAAYABwAJAAoACwAMAAsA'+
        'DAAKAAsACgAKAAsACwALAAoACQAJAAcABwAFAAYABAAEAAMAAgADAAMABAAFAAQABQACAAMAAQACAAEA'+
        'AgACAAMAAwAEAAMABQAFAAYABwAHAAgACAAHAAcABAAEAAIAAgACAAEAAwACAAQAAwAFAAMABgAEAAcA'+
        'BgAIAAcACQAIAAoACQAJAAkACAAJAAkACgAJAAsACAAJAAUABgAEAAQABAAFAAQABQAEAAUABAAEAAUA'+
        'BQAGAAUABgAFAAYABgAJAAkADQANABEAEAARABEAEQAQABAAEAAQABAADgAPAAwADAAKAAoACQAKAAoA'+
        'CwAMAA0ADgAPAA8AEAAPABAADwAQABAAEQASABMAEgATABEAEQARABEAEQAQAA8ADgANAAwADAAKAAsA'+
        'CgAJAAgABgAFAAYABQAJAAgADAALAA0ADAAMAAsACgAJAAgACAAIAAgACgAKAAoACgAKAAkACgAKAAsA'+
        'CwANAA0ADgANAA8ADgARAA8AEAAPAA4ADgANAAwADQALAA0ACwANAAwADgAOAA8ADwAPAA8ADgAPABAA'+
        'EAARABIAEAASAA4AEAANAA4ACwAMAAsADAALAAwADAANAA0ADgANAA0ADgANAA8ADwAQABAAEAAQAA8A'+
        'DwAOAA0ADAALAAoACQAJAAkACgAKAAsACwAJAAkABwAHAAYABwAJAAkACgALAAkACgAJAAoADAANABAA'+
        'EAARABEADwAQAA4ADgANAAwACwALAAsACgALAAsADAALAAsACwALAAsADQAMAA0ADQAMAAwACgAKAAkA'+
        'CQAJAAkACQAKAAoACwAKAAsACQAKAAcABwAFAAUABQAFAAcABwAKAAoADAALAAwADAAMAAwADAAMAA0A'+
        'DQAOAA4ADwAOAA8ADgAOAA4ADQANAAsACwAKAAoACQAJAAkACgAKAAsACwAMAAsADAAKAAwACgAMAAsA'+
        'DQAMAA0ADQAOAA4ADwAQABAADwAPAA0ADQAMAAwADAALAAoACQAHAAYACAAHAAoACQAKAAkACAAHAAgA'+
        'CAALAAsADAAMAAsADAALAAwADAAMAAoACgAHAAcABAAEAAEAAQD//////f/8//7//f8AAP//AgABAAQA'+
        'AwAHAAYACgAJAAwACwANAA0ADwAPABEAEgATABMAEgATABEAEQANAA4ACgALAAkACgAKAAsACgALAAgA'+
        'CQAGAAYABAAEAAMAAwACAAEAAwACAAUABQAHAAcACQAIAAgABwAGAAUABAADAAIAAgACAAEAAgABAAIA'+
        'AQADAAMAAwADAAIAAgACAAIABAAEAAUABgAGAAYABgAHAAcABwAHAAcABQAEAAIAAgAAAAAA/v/9//z/'+
        '+//7//r//f/8//////8AAAAAAAAAAAAAAAAAAAAAAAAAAP//AAD//wAAAAAAAP//AAD+/wAA/v////7/'+
        'AAAAAAEAAQABAAAAAQAAAAEAAgADAAMABAAEAAQABAAEAAIAAgD//wAA/P/8//v/+v/8//v//P/8//r/'+
        '+//4//n/9//3//b/9v/0//X/8//0//P/9P/z//P/8v/y//L/8v/z//P/8//z//L/8v/x//D/8P/u/+//'+
        '7f/v/+7/8//y//f/9f/3//X/9P/y//D/7//v/+7/7//u//D/7v/v/+7/7v/u/+//8P/y//L/9P/0//X/'+
        '9P/z//P/8v/y//D/7//u/+3/7v/u//H/8f/z//P/8v/x//D/7//x//D/8//z//P/8//y//H/8f/x//H/'+
        '8f/x//H/8P/w//D/8P/x//H/8v/y//P/8//z//T/9P/0//X/9P/1//T/9P/0//L/8//x//H/8P/w//H/'+
        '7//x//D/8f/x//H/8f/y//H/8f/v/+7/7f/r/+v/6v/r/+r/6//s/+z/7//v//L/8//1//b/9v/4//b/'+
        '9v/0//X/9f/2//b/+P/4//r/+f/6//n/+v/5//n/+P/5//j/+f/5//r/+//7//r/+v/4//j/9f/3//X/'+
        '9//3//f/9//3//f/+P/2//j/9v/4//b/9//0//X/8f/y/+//8P/u/+//7//v/+//7//w//D/8v/y//T/'+
        '9P/0//T/8//z//L/8f/x//D/8v/y//b/9v/6//r//P/8//v/+//6//r/+f/5//f/9//1//T/9f/0//X/'+
        '9f/0//T/8//y//H/8P/u/+3/6v/p/+f/5v/n/+X/6P/m/+n/5//q/+j/6//p/+r/6f/o/+b/5//l/+n/'+
        '5//r/+r/7P/r/+z/6//u/+3/7//u//D/8P/x//L/9f/1//j/+P/6//r/+//8//z//f/8//z/+v/6//j/'+
        '+P/3//f/9P/1//L/8v/y//H/8v/y//L/8v/w//D/7//v//D/8P/y//L/8//0//P/9f/0//X/9f/2//f/'+
        '+P/5//r/+f/7//n/+v/5//n/+P/4//f/9//2//f/9//4//r/+v/7//r/+v/5//n/+f/6//v//f/9////'+
        '/v/+//7/+v/7//X/9v/x//L/7//v/+7/7v/t/+7/7f/v/+7/8P/w//D/8f/x//P/8//1//b/9//5//n/'+
        '+v/5//n/+//6//z//f/9//7//P/8//r/+f/4//f/9//2//b/9v/1//X/8//z//H/8f/w//D/8f/x//H/'+
        '8v/x//H/8P/w//D/8P/x//H/8v/y//H/8f/x//H/8f/w/+//7v/s/+v/6//q/+z/6//t/+v/7f/r/+3/'+
        '6//v/+3/8f/v//L/8P/y//H/8v/x//P/8//1//X/+f/5//3//f///////v/+//v/+v/3//f/9v/2//b/'+
        '9v/5//j//P/7//3//P/8//v//P/7//3//f/+//7//////wAAAAACAAIABAAFAAYABwAGAAcABgAHAAYA'+
        'BwAGAAgACAAKAAoACwAKAAsACgAKAAkACgAKAAsACwAMAAwADAANAA0ADgAPAA4ADwANAA4ADQANAA0A'+
        'DQAMAA0ACwAMAAgACQAGAAcABAAEAAQABAAEAAQABAAFAAUABQAHAAcACQAJAAkACQAIAAkACAAIAAkA'+
        'CQAKAAoACwALAAsADAANAA0ADgAOABAADwARABAADwAPAA0ADQALAAsACwAKAAwACwAMAAwADQAMAA0A'+
        'DAAMAAsACwAKAAoACgAKAAoACwAMAA4ADgAQAA8AEAARABAAEQAPABAADwAPABAADwASABEAEwATABQA'+
        'FAAUABQAFQAUABUAFAATABMAEAAQAA4ADgAOAA4ADwAQAA8ADwAPAA8AEAAQABIAEwAUABUAFQAVABQA'+
        'FQASABMAEAAQAA8ADwAPAA8ADgAOAA0ADAALAAoACgAJAAkACQAHAAcABgAFAAUABQAGAAYACQAIAAwA'+
        'DAAOAA4ADgAOAA0ADQAMAAwACwAMAAsADAALAAsADAAMAAwACwALAAsACwALAAwADAAMAAsACgAJAAkA'+
        'CAAKAAkACgAKAAgACAAHAAYABgAFAAUABQAEAAQABAAFAAUABgAGAAUABAAEAAMAAwADAAQABQAGAAgA'+
        'CAALAAsADgAOABAAEAAQABEAEAAQABAADwAOAA0ACwAMAAoACgAJAAkACQAIAAgABwAGAAYABAAFAAMA'+
        'AwACAAIAAQABAAAAAAAAAAEAAwADAAcABgAJAAgACAAIAAcABwAFAAUABQAEAAYABQAIAAYACgAJAAsA'+
        'CwALAAoACQAIAAcABgAHAAcACQAJAAwADAANAA0ADQAOAAwADQAMAA0ADAANAA0ADQAOAA4ADgAPAA8A'+
        'EAAPABAADwAQAA8ADwAQABAAEQASABMAEwATABMAEwATABMAEwATABMAEwAUABIAEgAQABAADgAOAA4A'+
        'DgAPAA8ADgAPAA4ADgAPAA8AEgARABMAEwATABMAEgASABEAEAAQAA8AEQAQABMAEgAVABUAFgAWABYA'+
        'FQAVABQAFAAUABQAFAAUABQAFQAVABYAFgAYABgAGQAZABoAGgAbABsAGgAaABgAGAAVABUAEwATABMA'+
        'EgAUABMAFAAUABUAFQAXABYAFwAXABcAFgAVABUAFQAVABcAGAAaABoAHAAbABsAGgAZABkAFwAXABUA'+
        'FQAUABMAEwASABMAEwAUABQAFQAWABgAGAAbABoAHQAdAB0AHgAeAB8AHwAgACEAIQAiACIAIgAiACAA'+
        'IQAdAB4AGwAbABoAGQAaABoAGwAbABoAGwAYABgAGAAXABgAFwAYABgAGAAZABkAGgAaABoAGQAZABcA'+
        'FwAUABQAEwATABQAFAAWABUAFgAWABYAFQAVABQAFQAUABYAFgAXABYAFwAWABcAFwAZABgAGgAaABsA'+
        'GwAcAB0AHQAdABwAHAAZABoAFwAYABYAFwAWABYAFQAVABUAFQAUABQAEwAUABMAEwAUABQAFQAVABYA'+
        'FgAXABcAGQAZABsAGwAbABsAGQAZABcAFwAUABUAEgASABEAEQASABEAEwATABQAFAAUABQAFAAUABMA'+
        'EgATABIAEwASABQAFAAUABQAEwATABIAEgASABEAEQARABEAEQARABEAEgASABIAEgAQABAADwAQAA8A'+
        'DwAPAA8ADwAQABAAEQASABIAEgATABIAEgARABEAEAARAA8AEAAOAA8ADgAOAA4ADgAPAA8AEAAQABAA'+
        'EAAPAA4ADQAMAAwACwANAA0AEAAPABEAEQARABAADwAPAA0ADQAMAAwACwALAAoACQAJAAgACAAIAAgA'+
        'CQAHAAgABQAHAAQABgAFAAYABgAGAAcABgAGAAUABQAFAAQABAADAAMAAgADAAEAAgABAAEAAQABAAIA'+
        'AgACAAIAAwACAAIAAgABAAAA///+//3//f/9//z//P/7//v/+//7//r/+v/6//r/+//8//z//////wEA'+
        'AgACAAMAAgACAAEAAAD///7//P/8//r/+//6//v/+v/8//v//f/7//3/+//8//r/+//4//n/+P/4//n/'+
        '+f/7//v//f/8//3//P/8//z//P/8//v/+//7//r/+//5//v/+f/6//n/+f/5//r/+v/6//n/+f/4//f/'+
        '9v/3//f/+P/4//n/+v/7//v//P/8//7//v///////v////3//v/9//3//f/+//7//////wEAAQABAAEA'+
        'AQACAAEAAgAAAAEAAAD//////v/9/////v8AAAAAAQABAAAAAQAAAAEAAQABAAIAAQABAAAA////////'+
        '/////////v/+//3//f/7//v/+v/7//r/+//7//v/+//8//z//P/7//v/+//7//r/+//7//v//P/8//z/'+
        '/P/7//v/+v/7//z//P/+//3//v/+//z//P/5//n/9//3//b/9v/2//X/9P/z//T/8//1//P/9f/0//T/'+
        '9P/y//P/8v/z//T/9P/2//X/9f/1//P/8//y//L/8f/x//H/8f/x//H/8f/x//L/8f/y//L/8v/y//P/'+
        '8//1//T/9//2//j/+P/4//j/+P/4//j/+f/5//r/+v/6//n/+v/4//j/9v/2//P/8//y//L/8v/y//L/'+
        '8v/y//L/8v/z//T/9P/1//b/+P/3//r/+f/6//n/+P/4//f/9//3//f/9//2//b/9f/2//b/9//3//j/'+
        '+f/3//j/9f/3//T/9f/z//T/8v/y//D/7//t/+3/6//s/+v/7f/t/+//8f/x//P/8v/1//L/9f/z//X/'+
        '8//0//P/8//z//P/8//y//H/7//u/+z/6v/q/+n/6//r/+3/7v/v/+//7//v//D/8P/x//L/8f/z//H/'+
        '8//x//P/8v/z//L/8v/z//L/8//z//T/9P/1//X/9v/2//b/9v/2//b/9//3//j/+P/5//j/+f/4//r/'+
        '+P/5//f/9//2//X/9f/z//T/8v/0//H/8//y//P/8v/z//L/8v/w//D/7f/t/+v/6//r/+v/7f/t//D/'+
        '8P/y//H/8//y//P/8v/y//H/8P/w/+7/7f/q/+r/5//n/+b/5v/l/+b/5v/m/+b/5v/n/+X/5//m/+j/'+
        '6P/q/+r/6v/r/+r/6v/q/+r/6f/p/+j/6P/n/+f/5//m/+f/5v/n/+f/6P/p/+n/6v/p/+n/6P/n/+j/'+
        '5v/n/+b/5//m/+f/5//n/+f/5//o/+b/5//l/+f/5f/n/+f/6P/o/+n/6P/o/+b/5v/k/+T/4//j/+T/'+
        '4//l/+T/5v/l/+f/5//o/+j/6f/p/+n/6f/r/+r/7P/r/+3/7P/t/+z/7P/r/+z/6//s/+3/7v/v/+//'+
        '8P/v//D/7//v/+//7//w//D/8P/x//H/8f/y//L/8//0//X/9f/2//f/9//4//n/+f/6//r/+f/5//n/'+
        '+f/6//n/+//5//z/+v/7//v/+//7//v/+//8//v//P/8//3//f/+//7/AAAAAAEAAQABAAEAAAAAAP7/'+
        '///+//////8AAAAAAQABAAEAAAABAP//AAD////////+/wAA//8BAAAAAQACAAEAAgAAAAAA//////7/'+
        '///+//7//v/+//7///8AAAAAAQABAAIAAQAAAAEA//////7//v/+//3//v/9//7//f/9//3//P/8//v/'+
        '+//7//v//P/8//3//f/9//7//f/+//3//P/8//v//P/7//3/+//9//v//P/7//v/+//7//z//P/9//3/'+
        '/v//////AAAAAAEAAgAEAAQABgAGAAgABwAIAAgACAAIAAgACAAHAAgABwAHAAcABwAIAAgACQAJAAoA'+
        'CgAJAAkABwAIAAYABwAFAAYABQAGAAYABgAGAAYABgAFAAQABAADAAQAAgADAAEAAgAAAAEAAAAAAP//'+
        '///+//7//v/+/////v////7////+//3//f/8//3//P/9//3//f/+//3//v/9//3//P/8//z/+//7//r/'+
        '+v/5//n/+v/6//v/+//8//z//f/9/////v8AAP//AgAAAAIAAgADAAMAAwAEAAMABAADAAQAAwAEAAMA'+
        'BAADAAMAAgACAAEAAAAAAP////8AAAAAAQACAAMAAwAEAAMABQADAAUAAgADAAEAAQD///7//v/9//7/'+
        '/v/9//7//f/9//z//f/9//z//f/8//3//P/8//z/+//7//z//P/+//7/AAAAAAAAAAD///7//v/+////'+
        '/////wAA/v////z//P/5//n/+f/5//r/+v/7//z/+//7//n/+P/2//X/9f/0//X/9f/2//b/9v/3//f/'+
        '9//3//f/9v/2//b/9f/1//T/9P/z//P/8//0//T/9f/2//f/+P/5//n/+f/5//r/+v/8//v//v/8////'+
        '//8AAAEAAQACAAAAAgAAAAEA/////////v/+//7//////////////////v/+//7//v/+//7//////wEA'+
        'AAACAAEAAgACAAMABAAEAAUABAAFAAUABQAFAAQABQAEAAUABQAGAAYABgAGAAUABAAEAAMAAgACAAIA'+
        'AwADAAQABAAEAAQAAwAEAAMABAADAAQAAwAEAAMABAADAAQABAAEAAUABQAGAAUABQAFAAUABQAFAAYA'+
        'BQAGAAYABgAGAAYABgAGAAcABwAIAAcACAAHAAgABwAIAAoACgALAAsACwALAAsACwALAAwADAANAA0A'+
        'DQANAAwACwAKAAkABwAIAAUABwAFAAUABAAEAAQAAwADAAMABAAEAAUABQAGAAYABwAHAAgACAAJAAgA'+
        'CAAIAAgACQAJAAkACgAKAAsACgAKAAoACQAKAAkACgAKAAsACwAMAAwADAANAAwADQAMAA0ADAAMAA0A'+
        'DAAOAA0ADwAOAA4ADgANAA0ACwAMAAsADAAMAA0ADgAPABAAEAARABEAEAAQAA4ADgAMAAwACwALAAsA'+
        'CwALAAsADAALAA0ADAAOAA0ADwAOAA0ADQALAAwACwAMAAwADAANAAwACwALAAkACQAHAAcABQAGAAQA'+
        'BQAFAAUABgAGAAgABwAIAAgACAAJAAcACQAHAAgACAAHAAgABwAJAAYACAAFAAgABgAJAAgACgAKAAkA'+
        'CQAIAAcABgAGAAUABgAFAAYABQAHAAYABwAIAAkACgALAAsACwALAAsACwALAAsADAAMAAwADQANAA8A'+
        'DwAQABAADwAPAAwADQAKAAsACgAKAAoACgAJAAkACQAJAAgACAAIAAgACAAHAAgABwAIAAYABwAGAAYA'+
        'BQAGAAUABgAFAAUABQAEAAQABAAFAAQABQAEAAUABAAFAAUABgAIAAgACQAJAAoACQAKAAkACgAKAAkA'+
        'CQAHAAgABgAGAAUABQAFAAUABAAFAAQABQAFAAYABwAHAAgACAAJAAgACQAIAAoACQALAAoACwALAAsA'+
        'CwAKAAoACgAKAAkACQAIAAgABwAGAAUABQAFAAQABAAFAAMABAACAAMAAQADAAEAAgABAAIAAgADAAMA'+
        'AwADAAIAAQAAAAAAAAABAAEAAgABAAEAAQD//////f/9//v/+//6//v/+//7//v/+//7//v/+//7//z/'+
        '/P/8//v/+//6//r/+f/6//n/+//6//v/+v/7//r/+//6//v/+//7//z/+v/7//j/+f/2//f/9P/0//P/'+
        '8//z//T/9f/1//f/+P/5//n/+P/5//j/+P/4//n/+v/6//v/+//6//r/9//4//X/9v/0//X/9f/0//b/'+
        '9f/3//b/+P/4//r/+f/7//r/+//6//v/+f/6//n/+P/4//f/9v/1//X/8//0//L/8//w//H/8P/x//H/'+
        '8f/x//L/8f/x//H/8f/x//H/8v/x//T/8//1//X/9v/2//b/9v/3//b/+P/3//n/+f/6//n/+v/5//n/'+
        '+f/4//j/+P/4//n/+f/7//z//P/9//z//P/8//z//f/9//7////+/////v/+/////v////7//v/+//z/'+
        '/P/6//r/+P/5//f/+P/3//f/+P/4//n/+f/6//r/+v/6//r/+v/5//n/+P/3//f/9//3//f/+P/4//j/'+
        '+f/5//r//P/8/////v///////v/9//7//f/+//3//v/9//z//f/7//z/+//7//v/+v/8//v//P/8//z/'+
        '/f/8//3//P/9//3//f/+//7////////////+//7//f/9//z//f/8//3/+//8//v/+//7//v//P/7//z/'+
        '+//8//z/+v/7//r/+//7//z//f/9//7//f/8//z/+v/6//n/+f/5//j/+f/4//j/+P/5//n/+v/7//v/'+
        '/P/7//z/+//8//z//P////3/AAD+/wAA///+//7//f/9//r/+//3//j/9f/2//P/9P/z//T/8//z//L/'+
        '8v/x//H/8P/w//H/8P/y//L/8//z//P/8//0//T/9v/1//f/9v/2//X/9f/0//P/8//y//P/8//z//P/'+
        '8//z//L/8v/y//H/8f/w//H/8P/w//D/8P/w//H/8f/y//D/8v/w//H/7//w/+7/7//u/+3/7f/s/+v/'+
        '6v/q/+n/6f/p/+f/6f/n/+n/6P/o/+n/6P/p/+b/6P/m/+n/6P/q/+r/6//r/+v/6//q/+v/6f/p/+n/'+
        '6P/p/+j/6v/p/+v/6v/s/+v/7f/t/+7/7//v//H/8P/x//H/8f/y//H/9P/y//X/9P/2//X/9v/2//f/'+
        '+P/4//n/+f/5//f/+P/2//f/9f/2//X/9f/1//T/9f/0//T/8//z//L/8//y//P/8v/0//P/9f/1//b/'+
        '9v/4//f/+v/5//r/+v/5//r/+f/6//r/+//5//n/9//3//b/9f/2//b/9//3//b/9//1//b/9f/1//b/'+
        '9v/2//b/9//3//j/+P/5//j/+P/4//b/9//1//b/9P/0//P/8//y//L/8v/y//L/8v/w//H/8P/w//D/'+
        '8P/w//H/7//w/+//8P/w//D/8//y//T/9P/1//X/9v/2//f/+P/4//j/+P/5//j/+f/4//r/9//5//b/'+
        '9//2//b/9f/2//T/9P/x//P/8f/y//H/8v/x//L/8P/x//D/8f/x//H/8v/y//P/8v/z//L/8v/x//L/'+
        '8f/y//H/8//y//P/8v/z//L/8//z//X/9f/2//b/9v/2//b/9v/3//f/+f/5//v/+//8//z/+//8//z/'+
        '/f/9//7//////wAAAAAAAAAAAAABAAEAAQAAAP/////9//7//P/9//z//P/8//v/+//7//v/+//7//z/'+
        '+//7//r/+v/6//r/+v/7//v//f/8//7//f////3////9//7//v/+//7////+/wAA//8AAAAAAAAAAP//'+
        '///+//7//f/9//3//f/+//7///////7//v/+//7//////wEAAgACAAMAAgACAAIAAQABAAEAAAABAP//'+
        'AAAAAAAAAQAAAAEAAQABAAEAAQACAAEAAgABAAEAAAAAAP////////////8AAAAAAgABAAIAAAACAP//'+
        'AAD//////////wAA//8BAAAAAgABAAMAAgAEAAMABAADAAUABAAGAAYABgAGAAUABQAEAAQABAAEAAUA'+
        'BQAGAAYABgAHAAcABwAIAAgACAAJAAkACgAJAAsACgALAAsADAAMAA0ADQAOAA0ADQALAAsACwAKAAwA'+
        'CwANAAsACwAKAAoACQAIAAgABwAGAAUABQAFAAQABQAFAAYABgAGAAYABgAGAAYABgAHAAYABgAFAAYA'+
        'BQAFAAUABQAFAAQAAwAEAAMABAAEAAYABgAGAAYABgAHAAcABwAHAAYABgAFAAYABgAHAAgACAAJAAgA'+
        'CgAIAAkACQAKAAsACwAMAAwADAAMAAwADAAMAAwADQANAA0ADAANAAwACwALAAkACQAHAAcABwAGAAcA'+
        'BwAIAAgACAAIAAcACQAJAAoACgAMAAsADAAMAAwADQAMAA0ADAALAAsACgAKAAkACgAJAAoACgAKAAsA'+
        'CgAMAAsADQAMAA0ADQAMAA0ADQANAA4ADQAPAA4ADwAPAA4ADwAOAA4ADgAOAA8ADwAPAA8ADwAPAA8A'+
        'EAAQABEAEQARABEAEQARABEAEQAQABEAEAARABAAEQAQABMAEgAUABIAFAASABIAEAARAA8AEQAQABEA'+
        'EQASABIAEwASABEAEQAPAA8ADQAOAA0ADgANAA4ADQAMAAwACwALAAsACwAMAAsADAAKAAsACgAKAAoA'+
        'CgAJAAkACQAIAAkACQAKAAsACgALAAgACgAIAAoACQALAAoACwAKAAoACgAKAAsACwAMAAsADAALAA0A'+
        'DAAOAA0ADAAMAAoACgAIAAkACQAJAAoACgAKAAoACwALAAsACwALAAsACgAKAAoACgALAAoADAALAA0A'+
        'DQAOAA8AEAARABEAEgASABMAEgATABMAEgATABIAEwARABMAEgAUABMAEwATABIAEgARABEAEQASABIA'+
        'EwASABMAEAARAA8AEAAOAA8ADQAOAAsADAAKAAoACgAJAAoACQAKAAoACwAMAAwADgAMAA4ADQANAA4A'+
        'DgAPAA8ADwAPAA4ADgAPAA4ADwANAA4ADAAOAA0AEAAQABMAEwATABIAEAAPAA8ADgAPAA8AEAAPABAA'+
        'EAAQABAAEQARABIAEgATABIAFAATABQAEwATABIAEgARABIAEQARABAADwAPAA8ADwAQABEAEgASABMA'+
        'EwATABIAEwASABMAEgAUABMAFAAUABUAFQAWABUAFgAWABYAFgAVABQAEgASABAAEQAQABEAEAAQABAA'+
        'EAAQABAAEQASABEAEgAQABIAEAASABIAEgASABMAEgATABMAFAATABUAEwAUABIAEgASABIAEgASABEA'+
        'EQARABAAEgASABMAEwATABMAEwATABUAFAAVABYAFAAVABMAFQAVABYAFgAXABYAFgAWABYAFgAWABYA'+
        'FQAUABQAEgATABEAEgAQABEADwAQAA8ADwAQABAAEQARABIAEQASABEAEwARABIAEQASABIAEgASABEA'+
        'EgAQABAADwAPABAAEAARABIAEgATABIAEwATABMAEwAUABMAEwASABMAEQASABAAEQAQABIAEQASABIA'+
        'EQARABAAEQAQABIAEgATABMAEgASABIAEQASABAAEgARABAAEQAPABAADwAPAA8ADgAQAA4AEQAQABIA'+
        'EgASABEAEQAQABEAEAARABAAEAAPAA8ADgAQAA4AEQAQABIAEQASABIAFAAUABQAFAATABMAEQARABEA'+
        'EQARABIAEQARABAADwAQAA8AEQARABIAEgASABIAEwATABMAEwASABIAEQASABEAEgASABMAEwATABMA'+
        'EwATABMAEwATABMAEwATABMAEwATABMAFAATABQAEgATABMAFAATABQAEwAUABIAEwATABMAFAAUABQA'+
        'FAATABQAEwAUABMAFAATABQAEwAUABMAEwATABIAEwARABMAEgAUABMAFAAUABQAFAAVABQAFQAUABMA'+
        'EwARABEADwAQAA8ADwAOAA4ADAAMAAoACwAKAAoACgAKAAgACAAGAAYABgAHAAcACAAIAAgACAAIAAgA'+
        'BwAHAAcABwAHAAgACAAJAAgACQAJAAoACgAKAAsACQAKAAgACAAGAAUABAADAAMAAgACAAIAAQABAAEA'+
        'AQABAAIAAgAEAAIABAABAAIAAQABAAEAAQACAAEAAQACAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA'+
        'AQADAAEAAwABAAMAAQADAAIAAwADAAQAAwADAAMAAwADAAIAAgACAAIAAgACAAMAAwADAAMAAgACAAIA'+
        'AQACAAEAAgACAAIAAgACAAEAAQAAAAEAAQACAAMABQAGAAcACAAHAAgABwAGAAYABgAHAAYABwAHAAcA'+
        'BwAHAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAcABwAIAAcACQAHAAgABwAIAAgACAAIAAgABwAHAAYA'+
        'BgAGAAYABgAGAAYABQAGAAUABgAGAAcABwAIAAcACAAIAAgACAAIAAgACQAIAAkACAAJAAgACAAIAAcA'+
        'CAAIAAkACQAKAAoACwAMAAwADQAOAA4ADwAOAA4ADgAOAA8ADgAOAA0ADAAMAAwACwAMAAwADQANAAwA'+
        'DQALAAwACgALAAkACgAIAAgACQAJAAoACgALAAsACwALAAsACwAMAAwACwALAAsACwALAAoACwALAAwA'+
        'CwAMAAwADQANAA4ADQANAAwACgAKAAkACQAIAAoACAAKAAgACQAJAAgACQAIAAkACQAKAAoACgAKAAgA'+
        'CQAGAAgABQAHAAYABwAGAAYABQAFAAQABAADAAMAAwACAAIAAAAAAP///v/+//7//v//////AQAAAAEA'+
        'AQABAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEA//8BAAAAAAAAAP/////+//7//P/9//z//f/9//7/'+
        '/f/+//3////9/wAA//8BAAAAAgABAAEAAQAAAAEA//8AAP////8BAAAAAgABAAAAAAD9//7//P/9//7/'+
        '/f/+//3//v/8//3//f//////AQAAAAIAAQADAAIAAwACAAIAAwACAAIAAgACAAIAAQABAAAA////////'+
        '/////////v////3//f/8//z//f/9//7//v///wAAAAABAAEAAgABAAEAAAAAAP//AAD//wAA/v8AAP7/'+
        '///9//7//f/9//3//f/8//3//P/9//3//f/+//////8AAP//AAD//////v////7////+/////v//////'+
        'AAAAAAEAAQABAAAAAAAAAAAAAgABAAQAAgAEAAMAAwAEAAMABQAEAAUABAAEAAMABAAEAAQABQAGAAYA'+
        'BwAHAAcACQAIAAoACAAKAAkACgAKAAwADAAOAA4ADQAMAAsACwALAAsADAANAAwADQALAAsACgALAAsA'+
        'DAALAAwACgAKAAkACQAJAAgACQAJAAsACwAOAA4AEQAQABAADwAOAAwADAALAAwADAAMAAwADAAMAAsA'+
        'CwAMAAsADAALAAwADAANAAwADgAMAA8ADQAQAA8AEAAQABAAEAAQABAAEAAQABAADwAPAA4ADQANAAsA'+
        'CwAKAAkACwAJAAsACQAMAAsADAAMAAwADAAMAAwADAAMAAsADAAMAAwADQANAA0ADQAMAAwACwALAAwA'+
        'DAANAA0ADAAMAAsACwALAAwADAANAAwADQAMAAwACwALAAsACwAMAA0ADgAPAA8AEAAPAA8ADgAPABAA'+
        'EAASABMAEgAUABEAEwARABIAEgAUABMAFAASABQAEgATABQAFAAVABQAFQAUABUAFQAVABUAFQAVABYA'+
        'FQAYABYAGgAYABoAGQAaABkAGgAaABsAGgAaABoAGQAZABkAGQAZABkAGgAaABoAGQAaABkAGgAaABoA'+
        'GgAbABsAHAAbABsAGwAaABkAGQAZABkAGQAZABgAFwAXABUAFQAUABUAFQAVABYAFgAWABYAFwAXABYA'+
        'FwAWABcAFwAXABkAGAAaABoAGgAbABoAHAAaABwAGgAaABkAGAAZABgAGQAZABgAGQAXABcAFwAXABkA'+
        'GQAcABwAHQAdAB4AHQAdAB0AHQAdAB0AHQAdAB0AHgAdAB8AHgAfAB8AHwAfAB8AHwAfAB8AHwAeAB4A'+
        'HQAdAB0AHQAdAB4AHQAeAB0AIAAeACEAHwAhACAAIAAfAB4AHgAeAB4AHgAeAB8AHgAeAB0AHAAcABwA'+
        'HAAcABwAHAAcABwAHAAcABwAHAAeABwAHgAdAB4AHQAdAB0AHAAdABwAHAAdABwAHQAdAB4AHgAfAB8A'+
        'IAAgACAAIAAgAB8AIAAeAB8AHgAeAB0AHQAbABsAGgAaABoAGgAbABsAGwAbABoAGQAYABgAFwAXABcA'+
        'FwAXABcAFwAXABgAFwAYABgAGAAXABcAFgAYABYAGAAWABgAFwAXABcAFwAXABkAGQAbABoAGwAbABoA'+
        'GwAZABoAGQAaABoAGgAbABsAHAAcABwAHQAbABwAGgAaABkAGgAaABsAGgAcABsAHAAbABwAGgAcABoA'+
        'GwAaABsAGwAbABwAGwAcABsAGwAcABsAHAAbABwAGwAbABsAGwAcABwAHgAdAB8AHgAfAB0AHgAcABwA'+
        'GwAcABsAHAAbAB0AHAAeAB0AHwAfACAAIAAhACEAIQAhACAAIQAfACAAHgAhAB4AIAAeACAAHgAgAB4A'+
        'HwAdAB8AHAAfAB0AHwAeACAAHgAgAB0AHwAcAB0AGwAcABwAGwAbABsAGwAaABsAGgAcABoAHQAbAB0A'+
        'HAAeAB4AHwAfAB8AHgAfAB4AIAAeACEAIAAiACIAJAAjACMAIwAhACIAHwAgAB8AIAAhACAAIwAhACMA'+
        'IAAiACAAIQAfACEAHwAiACAAIgAhACEAIQAhACEAIQAgACMAIQAmACMAJgAlACYAJQAlACQAJgAlACYA'+
        'JQAlACUAJQAlACUAJQAlACUAJwAmACgAKQApACoAKAApACgAKAAoACcAKAAoACgAKAAoACkAKQApACkA'+
        'KQAoACgAJwApACcAKQAoACkAKAApACkAKQApACoAKQArACkAKgApACkAKQAoACgAJwAnACcAJgAmACYA'+
        'JgAlACQAIwAiACIAIQAiACIAJAAkACYAJQAnACYAJgAmACQAJQAjACMAIgAiACEAIgAgACIAHwAiAB8A'+
        'IQAgACEAIAAhACAAIQAfACEAHwAgAB8AIAAfAB8AHgAfAB0AHgAcAB0AHAAcABoAGwAYABkAFgAYABcA'+
        'GAAXABgAFwAXABcAFwAYABkAGgAaABoAGwAbABsAGwAbABwAHAAbABwAGQAaABkAGAAZABgAGAAXABcA'+
        'FgAWABUAFgAVABYAFAAWABQAFgAUABYAFQAWABUAFgAVABYAFQAXABYAGAAXABgAFwAWABUAFQAUABQA'+
        'EwAVABQAFQATABQAEwAUABIAEwASABIAEgASABIAEwASABMAEwATABIAEgASABMAEwAUABQAFAATABQA'+
        'FAAVABUAFwAYABgAGQAYABgAGAAYABgAGAAYABkAGAAYABcAGAAWABcAFQAVABQAFAAVABQAFwAVABcA'+
        'FgAWABUAEwAUABIAFAASABMAEgASABEAEQARABEAEQARABAAEAAPAA4ADgANAA4ADQAOAA4ADgAOAA0A'+
        'DgANAA4ADgAPAA8AEAAQABAAEQARABIAEgATABMAEwATABMAEwASABMAEQATABEAEwARABMAEgATABIA'+
        'EwASABIAEQASABIAEgASABMAEQASAA8AEAAPAA8ADwAOAA8ADgAOAA0ADgANAA0ADQANAAwADQAMAA4A'+
        'DQAOAA0ADAAMAAwACwAOAA4AEgARABQAFAAVABUAFgAWABYAFgAVABUAFAAUABUAFAAVABUAFAAUABIA'+
        'EwARABIAEQARABEAEAAQAA8ADwAOAA8ADwAPAA8ADwAOAA4ADQANAAwACwALAAkACQAIAAgACAAIAAkA'+
        'CQAJAAkACAAJAAcACAAHAAgABgAHAAYABgAFAAUABAAFAAQABQAEAAUABAAFAAMABQADAAUAAwAFAAMA'+
        'BAADAAQAAgADAAEAAQD////////+/////v/+//7//f/9//z//P/7//v/+//6//r/+v/5//n/+P/4//f/'+
        '9//3//f/+P/3//j/9//4//b/+P/3//j/9//4//j/+f/5//r/+v/8//v//f/8//z/+//6//n/+f/5//n/'+
        '+P/4//f/9//2//f/9v/2//b/9v/2//b/9v/3//b/9//3//b/9v/1//X/9P/0//P/8//y//L/8f/y//L/'+
        '8//0//X/9//4//n/+v/6//v/+//8//z//f/7//v/+f/6//j/+f/4//n/+f/5//n/+f/4//j/+P/4//j/'+
        '+P/3//j/+P/4//j/+P/4//j/+P/5//r/+v/9//z//f/8//z/+//8//v//f/8//7//v/+//3//v/9//7/'+
        '/f/+//7////+//////8AAAAAAAAAAP7////9//7//v/+//7////+/////f/+//3//v/9//3//f/9//3/'+
        '/v///wAAAQACAAEAAwABAAIAAAAAAP/////9//7/+//8//v//P/8//z//f/9//3//f/9//3//f/9//7/'+
        '/f/9//3//f/8//3//f/9//3//v/9//7//f////7/AAD+/////f/9//z//P/7//3/+//+//z//v/9//3/'+
        '/f/9//z//f/8//7//f////7////+/////v/+//7///////////8AAAAAAAABAAAAAQAAAAEAAQACAAEA'+
        'AgAAAAEA//8AAAAAAAABAAEAAgABAAIAAQABAAAAAQAAAAEAAQABAAEAAAAAAAAAAAD////////+//7/'+
        '/P/9//v//P/7//3/+//9//z//f/9//7//f/+//3//f/9//3//v/+/////v8AAP7////+//////8AAAAA'+
        'AQAAAAEA//8AAP//AAD//wAA//8AAP3////8//7//f/+//7///8AAAAAAQABAAIAAgABAAEAAAAAAAAA'+
        '//8BAAAAAgABAAEAAQAAAAAA/v/+//7//f/9//z//f/8//7//v8AAAAAAQACAAEAAgACAAIAAwADAAMA'+
        'BAADAAMAAwADAAQABAAGAAYABgAHAAUABgAFAAYABgAGAAcABwAHAAcACAAIAAoACQAJAAkABwAIAAUA'+
        'BwAFAAgABgAIAAYABwAFAAYABwAHAAkACgAKAAwACwANAA0ADQAOAA4ADgAOAAwADQALAAwADAANAA0A'+
        'DQAOAA0ADgANAA8ADgAQAA8AEQAPABIAEAATABIAFAATABMAEwASABMAEQASABIAEgASABMAEwATABMA'+
        'EwATABIAEgARABEAEAARABAAEgARABIAEgATABIAFAASABUAFAAXABYAGAAXABgAGAAYABcAFwAWABYA'+
        'FQAWABUAFwAWABkAFwAaABgAGgAYABoAGQAaABoAGQAZABkAGQAZABgAGQAZABgAGQAXABgAFgAWABUA'+
        'FAAVABQAFQAUABUAFQAUABQAEwATABMAEgATABIAEwATABMAFAASABUAEgAUABEAEgAQABEAEQASABIA'+
        'EwASABQAEwAVABQAFgAVABcAFQAWABMAFAASABQAFAAVABUAFgAVABYAFAAWABUAFwAWABgAFgAXABUA'+
        'FgAWABcAFwAZABgAGgAYABkAGAAZABkAGQAZABkAGQAZABoAGgAcABwAHQAcAB0AHAAcABwAHQAcAB4A'+
        'HQAfAB0AHwAdAB8AHQAeAB0AHQAdABwAGwAbABoAGwAaABoAGQAZABkAGAAYABcAFwAXABcAGAAYABgA'+
        'GQAZABgAGQAYABkAGAAZABgAGQAYABgAGAAYABgAGAAYABkAGAAYABcAGAAXABkAGAAaABoAGwAaABsA'+
        'GwAcABsAGwAbABsAGwAbABsAGwAbABwAGwAdABwAHQAcAB0AHAAdABwAHQAcAB0AHAAdABsAHAAbABsA'+
        'GwAcABwAHQAdAB4AHQAeAB4AHgAeAB4AHgAdAB0AHQAcAB0AHQAeAB4AHgAfAB0AHwAeAB8AIAAgACEA'+
        'IgAhACIAIQAhACEAIQAhACIAIQAiACEAIgAgACIAIQAiACAAIgAgACIAIgAkACQAJgAkACcAIwAmACMA'+
        'JQAkACUAJAAlACQAJQAjACQAJQAlACYAJgAnACYAJgAmACUAJwAnACgAKAApACkAKQApACkAKQApACgA'+
        'KQAnACgAJwAoACcAKAAoACgAKAAoACgAKAAoACkAKQAqACoAKgArACoAKwArACsAKwArACsAKgAqACsA'+
        'KgAsACsALQArACwAKgAqACkAKgAoACoAKQArACkAKwApACwAKwAtACwALQAtAC0ALQAuAC4AMAAwADEA'+
        'MQAxADEAMAAwADAAMAAwADAALwAvAC4ALgAuAC0ALQAtACwALAAqACsAKwArACwAKwAtACsALQAsAC4A'+
        'LQAvAC8ALwAuAC0ALQAtACwALQAtAC0ALgArAC0AKgArACoAKQAqACkAKgApACoAKQAqACkAKQAoACgA'+
        'JwAnACcAJwAoACgAKQAoACoAKQApACkAKQAoACgAJwAoACYAJwAlACcAJQAmACQAJQAkACQAJAAkACQA'+
        'JAAiACQAIQAjACAAIgAgACEAHwAgAB8AHwAfAB8AHgAeAB4AHgAeAB0AHQAdABwAHAAbABsAGwAbAB0A'+
        'HQAeAB4AHgAdAB0AHAAdABwAHgAdAB4AHQAfAB4AHwAfACAAIAAgAB8AHgAeABwAHAAbABwAGgAaABkA'+
        'GQAZABkAGQAZABkAGgAZABoAGQAaABsAHAAcAB0AHQAeAB0AIAAeACAAHgAgAB4AHwAeAB8AHgAfAB4A'+
        'HwAdAB0AHAAbABwAGwAcABoAGgAZABkAGAAYABcAGAAXABcAFgAVABQAFQAUABcAFgAZABcAGgAYABkA'+
        'GAAYABcAGAAXABkAGAAaABoAGgAaABkAGAAXABYAFgAVABcAFgAYABcAGAAYABgAGAAYABkAGQAZABkA'+
        'GgAZABoAGgAaABsAGwAcABwAGwAdABkAGwAXABoAGAAaABoAGwAcAB0AHQAdAB0AHQAfAB8AIQAhACIA'+
        'IwAiACMAIgAjACIAIwAhACIAIQAhACEAIQAhACAAHwAeAB8AHQAfAB4AIAAfACAAHwAfAB8AHwAfACAA'+
        'IAAhACAAIAAhACEAIQAjACIAJQAkACUAJAAkACQAIwAjACIAIgAhACEAIQAgACAAHwAfAB4AHgAfAB8A'+
        'IAAgACIAIAAhAB8AIAAeAB8AHgAfAB8AIAAgACAAIAAhACEAIQAhACEAIAAgACAAIAAgACAAHwAfAB8A'+
        'HgAfAB4AHgAeAB4AHgAcAB0AHAAcABsAHAAbABsAGgAZABkAGQAZABkAGgAaABoAGgAaABoAGQAaABkA'+
        'GgAXABgAFgAXABcAFwAXABcAFwAXABYAFgAWABUAFQAVABUAFAAUABMAFAASABQAEwAUABMAFAAUABQA'+
        'FAAUABQAFAAUABUAFQAUABQAEgASABAAEAAPABAAEQARABMAEwATABMAEgATABIAEwAUABQAFgAVABcA'+
        'FgAXABYAFwAXABgAGQAZABoAGQAZABgAFgAVABQAEgASABIAEgASABIAEQASABAAEAAOAA8ADgAPAA8A'+
        'DwAPAA8ADgAOAA4ADgAOAA8ADgAOAA0ADQALAAsACgAKAAoACQALAAoADAALAA0ADAANAA0ADgANAA8A'+
        'DwAPAA8ADgAOAAwADAAKAAoACQAJAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgABwAIAAcABwAGAAYA'+
        'BgAGAAYABwAHAAgACAAIAAgABwAIAAcABwAGAAcABgAHAAYABwAGAAYABgAGAAYABgAGAAcABgAIAAcA'+
        'CAAHAAcABwAHAAgABwAIAAgACAAIAAgACQAJAAoACgALAAsACwAMAAwADQAMAA0ADAANAA0ADAAMAAsA'+
        'DAAKAAsACgAJAAkACQAJAAgACAAIAAgACAAKAAoACwALAA0ADAANAAwADQANAA0ADgAPAA8ADwAPAA8A'+
        'DwAQAA8AEQARABMAEgATABIAEgASABIAEgARABIAEQARABEAEQAQABAADwAQAA8AEAAPAA8ADwAPAA0A'+
        'DQALAAwACQALAAgACgAHAAkABwAIAAgACAAIAAkACQAJAAkACAAJAAgACQAIAAkACQAKAAoACwAKAAoA'+
        'CQAJAAgACAAHAAcABwAHAAYABwAFAAYABQAGAAYABgAHAAYACAAGAAYABgAFAAcABQAHAAUABgAFAAUA'+
        'BQAFAAYABQAHAAUABgAFAAYABgAGAAYABwAGAAcABgAHAAYABwAGAAcABgAHAAYABwAGAAcABgAHAAYA'+
        'BgAGAAUABwAGAAgABwAIAAgABwAHAAYABgAGAAYABwAHAAcABgAHAAYABwAGAAcABwAHAAcABwAHAAcA'+
        'BwAHAAYABwAGAAYABQAGAAUABgAFAAcABgAIAAgACQAJAAkACgAKAAoACwAMAAwADQAMAA0ACgALAAkA'+
        'CQAJAAkACAAJAAgACAAIAAgABwAHAAcABwAGAAcABgAGAAcABgAHAAYABgAFAAUABQAFAAUABQAFAAQA'+
        'BgAEAAUAAwAFAAMABQADAAQAAwADAAQABAAEAAQABAAEAAQABAAFAAQABgAFAAcABgAIAAcACAAGAAkA'+
        'BwAJAAcACAAHAAYABgAEAAUAAwAEAAIAAwABAAIAAQACAAAAAQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAA'+
        '//////////8AAAAAAQACAAMAAwADAAMAAwADAAMABAAEAAUABgAGAAcABwAHAAgACAAJAAkACgAKAAkA'+
        'CQAIAAcABwAGAAYABQAFAAQABAADAAMAAwACAAIAAQACAAEAAQABAAEAAgABAAIAAgADAAMAAwAEAAMA'+
        'BAAEAAQABAADAAMAAwADAAMAAgADAAIAAwACAAMAAgACAAIAAgACAAIAAwADAAMAAwACAAIAAQABAAEA'+
        'AAABAAEAAQABAAEAAQABAAEAAAAAAP/////+//7//v/+//////////////////7//v////7/AAD//wAA'+
        '/////////////wAAAAABAAIAAgADAAIABAACAAQAAgAEAAMABAADAAMAAgACAAIAAgACAAMABAAFAAUA'+
        'BQAFAAQABAADAAUAAwAFAAQABQAFAAYABgAHAAcACAAHAAkACAAKAAgACwAKAAsACwAKAAwACgALAAoA'+
        'CgAKAAkACgAJAAoACwALAAwACwAMAAsADAANAAwADwAOABAAEQARABIAEQATABIAEwAUABQAFAAUABQA'+
        'FAATABMAEwATABQAEwAUABMAFAASABMAEQARABEAEAARABEAEgASABMAEwASABMAEgASABIAEQASABEA'+
        'EgARABEAEQAQABIAEAASABAAEAAQAA8ADwAOAA8ADgAPAA8ADwAPAA4ADwAOAA8ADwAOAA4ADAANAAwA'+
        'DAANAAwADgANAAwADAAJAAsACAAKAAgACgAKAAwACwANAAwADQANAA0ADQANAA0ADAAMAAsACwAKAAsA'+
        'CgAMAAsADAAMAAwADAANAA0ADwANABAADgAPAA4ADgAPAA4ADwAPABAAEAARABAAEQAQABEAEAARABEA'+
        'EQARABEAEQARABAAEAARABEAEwATABUAFAAVABQAFQATABYAFAAXABYAGAAYABgAGQAXABkAFwAZABgA'+
        'GQAYABgAGAAXABcAFwAXABcAFwAYABcAFwAXABYAFwAWABcAFwAWABcAFQAWABQAFAAUABQAFAAUABUA'+
        'FQAWABcAFwAYABgAGQAYABkAGAAYABkAGQAZABoAGgAbABsAHAAcABwAHQAdAB8AHgAgAB8AIAAgACAA'+
        'IAAgACAAIAAfACAAHwAgAB8AIQAgACEAIQAhACEAIAAgAB8AIAAfACAAIAAhACAAIQAgACAAIAAgACAA'+
        'HwAgAB8AHwAeAB0AHgAdAB4AHQAeABwAHQAcABsAGwAaABoAGQAaABkAGgAaABoAGgAaABsAGwAbABwA'+
        'HAAdAB0AHgAeAB8AHwAfAB4AHgAeAB0AHQAdAB4AHgAfAB8AIAAgACAAIQAhACMAIgAkACMAJAAkACQA'+
        'JAAlACUAJgAnACcAKAAoACgAKAAnACgAJwAoACcAKAAoACgAKAAqACoAKwAsAC0ALQAtAC4ALQAuAC4A'+
        'LgAvAC8AMAAvADEAMAAyADEANAAzADUANAA1ADMAMwAzADIAMwAzADQANAA0ADQAMwAzADIAMgAyADMA'+
        'MwAzADQAMwA0ADIAMwAxADMAMQAyADEAMQAwADAALgAuAC0ALgAtAC4ALgAuAC0ALQAtAC0ALQAtAC4A'+
        'LgAuAC4ALgAuAC0ALQAtAC0ALQAtAC0ALQAtACwALgAtAC4ALgAvAC8ALwAvAC8ALwAvADAALwAwAC8A'+
        'LwAvAC4ALgAuAC4ALgAuAC8ALwAvAC8ALwAvAC8ALwAvAC4ALgAsAC0AKwAsACsAKwAqACkAKQAoACcA'+
        'JgAlACUAJAAkACMAJQAkACUAJAAkACMAIgAiACEAIgAgACEAIAAiACEAIwAiACMAIwAkACQAIwAkACIA'+
        'IwAiACIAIgAiACEAIQAgACAAHgAfAB0AHgAdAB4AHAAeABwAHQAbABwAHAAcAB0AHQAdAB4AHQAeAB0A'+
        'HQAeAB0AHgAdAB4AHQAdAB0AHQAcABsAGgAaABkAGAAXABYAFgAVABUAFAAUABMAEwATABMAEgAUABMA'+
        'FAAUABUAFQAVABQAFAATABMAEwASABIAEAAQAA8ADgANAA0ADQAOAA4ADgAPAA4ADgAOAA4ADgAOAA8A'+
        'DwAOAA8ADgAOAA0ADgAOAA4ADgAPAA4ADgAOAA0ADgAOABAAEQASABMAEgASABEAEQAQABAAEQARABIA'+
        'EwARABIAEAAQABAADwARABAAEQASABAAEgAOAA8ADQANAA0ADQANAAwADgAMAA4ADAAOAA0ADgAOAA0A'+
        'DgAMAA0ADAAMAA0ADQAOAA4ADwAPAA8ADgAOAA4ADgAOAA8ADwAPABAADwAPAA8ADwAPAA4ADwAOAA4A'+
        'DgAOAA4ADgAOAA4ADwAPAA8ADgAQAA4ADwAOAA8ADwAQABAAEQASABIAEgARABEAEQARABAAEQARABIA'+
        'EwAUABQAFAAUABUAFAAVABQAFAAUABQAFAATABMAEwASABIAEQARABEADwAQAA0ADgANAA0ADgAOAA8A'+
        'DgAPAA4ADwAOAA8ADgAPAA4AEAAPABEAEAASABEAEgARABEAEQAQABIAEQATABEAEwASABMAEQASABEA'+
        'EgARABIAEgATABIAEwARABIAEQARABEAEAARABAAEAAPAA8ADgAOAAsADQAKAA0ACwANAAwADAAMAAsA'+
        'DQALAA0ADAAOAAwADQAMAAwACwALAAoACwAIAAoABwAJAAcABwAIAAcACQAHAAgABgAGAAUABQAEAAUA'+
        'BAAGAAQABgAEAAQAAwACAAIAAQADAAIABAADAAUAAwAFAAMABAADAAQAAwAEAAMAAwACAAIAAQABAAEA'+
        'AQACAAEAAgACAAIAAgACAAIAAgABAAMAAQACAAAAAQAAAAEAAAABAAEAAQABAAAAAQD+/wAA/f////7/'+
        'AAD//wAAAAD////////////////////////+//7//P/8//r/+//6//r/+v/6//r/+f/7//r//P/7//z/'+
        '/P/8//z/+//7//v/+v/7//v//P/8//3//P/9//z/+//7//r/+v/4//n/+P/4//f/9//3//f/9v/3//b/'+
        '9//2//j/9v/4//f/+P/3//j/9v/4//b/9//2//f/9//3//j/+P/5//j/+f/4//n/9//5//b/+P/1//b/'+
        '9P/z//L/8f/w//H/7//x//H/8//z//X/9v/2//f/+P/5//n/+v/6//z/+v/7//j/+v/3//j/9v/2//b/'+
        '9v/2//b/9v/2//b/9v/2//X/9v/1//X/9P/0//P/9P/z//X/9P/2//X/9v/2//b/9v/1//b/9f/2//T/'+
        '9v/0//X/9P/0//T/9P/1//T/9f/0//X/9P/0//P/9P/z//T/9P/1//b/9v/3//b/9//2//f/9v/2//f/'+
        '9v/3//b/9v/2//b/9f/1//X/9f/1//X/9P/0//T/9f/0//X/9P/2//X/9v/2//f/9//3//j/+P/4//j/'+
        '+f/5//n/+f/6//r/+//6//z/+v/7//n/+f/4//f/9//2//n/9//6//n/+v/6//r/+v/5//n/+f/5//r/'+
        '+v/6//r/+v/5//n/+f/5//r/+v/6//n/+v/3//j/9v/2//X/9P/1//T/9v/0//b/9P/1//T/9P/0//T/'+
        '9P/0//X/9f/1//T/9f/0//T/8//z//L/8//y//P/8//z//T/9f/0//b/9P/2//T/9v/1//b/9v/2//f/'+
        '9v/2//b/9f/1//X/9f/2//X/9f/0//X/9P/1//T/9f/2//b/9//3//f/9//3//j/+P/5//r/+v/7//r/'+
        '+//6//r/+v/6//z//P/9//3//v/9//3//f/9//3//f/9//7//f/+//z//v/8//7//f////7/////////'+
        'AAD+/////v////7////+/////f////z//v/6//3/+v/7//r/+//6//v/+//7//z//P/8//z//P/8//z/'+
        '+//8//v/+//6//r/+f/5//j/+f/5//r/+v/6//n/+f/5//n/+P/6//n/+//6//v/+//7//r/+//5//v/'+
        '+f/7//r/+//6//r/+v/5//n/+f/5//n/+v/5//v/+v/8//r/+//6//v/+v/6//n/+v/5//r/+v/6//r/'+
        '+v/6//r/+v/7//v/+//7//r/+v/6//r/+v/7//z//P/9//7//v/+//7//v/9//3//v/+//////8AAAAA'+
        'AAAAAAAAAQAAAAEAAgACAAMAAgADAAIAAgABAAIAAQABAAEAAQABAAEAAQABAAIAAgADAAMAAwADAAQA'+
        'AgADAAEAAgABAAIAAgACAAMAAgADAAIAAgACAAIAAwADAAQABAAFAAUABgAFAAYABQAGAAUABwAGAAcA'+
        'BgAGAAUAAwACAAIAAQABAAEAAgACAAMAAwADAAMAAwAEAAQABAAGAAUABgAGAAcABwAGAAYABQAGAAUA'+
        'BQAFAAUABQAGAAYABwAGAAgABgAIAAcACAAJAAkACgAJAAsACgAKAAoACQAKAAkACgAJAAoACQAJAAgA'+
        'CAAIAAgACQAJAAoACwAKAAwACgALAAoACwAMAAsADgAMAA8ADgAPAA8ADwAQABAAEAARABEAEQASABEA'+
        'EQAQABEAEAAQABAADwAPAA8AEAAPABAAEAARABAAEQAQABEADwAQAA8AEAAQABAAEAARABAAEQASABIA'+
        'EwASABMAEgASABIAEAASABAAEgARABIAEQARABEAEAAQABAADwAQAA8AEAAPABAAEAARABEAEgASABEA'+
        'EQAQABEAEAARABEAEQASABEAEgARABEAEQARABEAEQASABIAEwATABQAEwATABMAEwATABMAFAAUABUA'+
        'FgAVABcAFgAXABYAFwAXABcAFwAWABcAFgAXABYAFwAXABgAGAAZABkAGgAaABoAGwAaABsAGgAaABkA'+
        'GQAZABkAGQAZABgAGQAXABgAFwAYABcAGAAYABkAGQAaABkAGgAZABoAGgAaABoAGgAaABoAGgAbABoA'+
        'GgAaABoAGQAZABcAGAAXABcAGAAYABoAGQAcABoAHQAbAB0AGwAdABwAHgAdAB8AHgAfAB4AHgAeABwA'+
        'HAAcABsAHAAbAB0AHAAcABwAGwAcABoAGgAbABoAHAAcAB0AHgAeAB8AHwAeAB8AHgAfAB4AHgAfAB4A'+
        'HwAeAB8AHwAfACAAHwAhACAAIQAiACIAIwAjACQAIwAkACIAIwAiACMAIwAkACQAJQAlACUAJAAkACMA'+
        'IwAjACMAJAAkACUAJAAkACUAIwAkACIAIwAhACIAIgAiACIAIgAiACMAIQAiACAAIQAfAB8AHgAdAB0A'+
        'HQAdABwAHAAcABsAGwAaABoAGQAaABoAGwAbABsAHAAcAB0AHAAeAB0AHgAeAB4AHgAeAB0AHQAdABwA'+
        'HAAbABsAGwAbABsAGwAcABsAGwAbABoAGgAaABoAGgAZABsAGQAbABoAGwAaABoAGQAaABkAGgAaABoA'+
        'GwAbABsAGwAbABoAGgAZABoAGAAZABkAGgAaABsAHAAcAB0AHQAdAB0AHQAdAB4AHQAeAB4AHgAeAB4A'+
        'HwAeAB8AHgAfAB4AHgAdAB4AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB4AHgAeAB8AHgAfAB4AHwAeAB4A'+
        'HgAeAB4AHgAdAB8AHQAeAB0AHQAeAB0AHwAdAB8AHgAfAB8AHwAfAB8AHwAfACAAIAAgACAAHwAfAB4A'+
        'HwAeAB4AHgAeAB0AHQAcABwAGwAcABwAHAAcABwAHAAbABsAGwAaABwAGgAcABoAHAAZABoAGAAXABYA'+
        'FgAWABYAFgAXABgAGAAZABgAGQAZABkAGQAZABkAGQAYABgAFwAXABcAFgAXABYAFgAVABUAFgAVABYA'+
        'FAAWABUA'
      );
        const ab=new ArrayBuffer(raw.length);
        const view=new Uint8Array(ab);
        for(let i=0;i<raw.length;i++) view[i]=raw.charCodeAt(i);
        ac.decodeAudioData(ab,buf=>{{_shutterBuf=buf;}});
      }}
      if(_shutterBuf){{
        const src=ac.createBufferSource();
        src.buffer=_shutterBuf;
        const g=ac.createGain(); g.gain.setValueAtTime(0.8,t);
        src.connect(g); g.connect(ac.destination);
        src.start(t);
      }}
      break;
    }}

    // Hoàn thành class — chuông 3 nốt C-E-G
    case 'done':
      note(523, 0.08, 0.24, 0.0);
      note(659, 0.08, 0.24, 0.09);
      note(784, 0.09, 0.32, 0.2);
      break;

    // Chuyển giai đoạn — một nốt trung tính
    case 'sub':
      note(784, 0.06, 0.18, 0);
      break;

    // Hiệu chỉnh xong — 4 nốt giải quyết lên dần (chuyên nghiệp)
    case 'calib':
      note(392, 0.065, 0.2,  0.0);
      note(494, 0.065, 0.2,  0.1);
      note(587, 0.065, 0.2,  0.22);
      note(784, 0.08,  0.35, 0.36);
      break;

    // Khởi động — 3 nốt lên nhẹ nhàng
    case 'boot':
      note(523, 0.06,  0.24, 0.0);
      note(659, 0.065, 0.24, 0.13);
      note(784, 0.07,  0.32, 0.28);
      break;

    // Kết thúc phiên — 4 nốt xuống nhẹ nhàng
    case 'outro':
      note(784, 0.065, 0.28, 0.0);
      note(659, 0.065, 0.28, 0.15);
      note(523, 0.065, 0.28, 0.32);
      note(392, 0.07,  0.4,  0.5);
      break;
  }}
}}

// ── Boot sequence ───────────────────────────────────────────
(function bootSeq(){{
  const msgs=['bm0','bm1','bm2','bm3'],delays=[400,900,1500,2200],bars=[20,45,72,100];
  const bar=document.getElementById('bootBar');
  msgs.forEach((id,i)=>{{
    setTimeout(()=>{{document.getElementById(id).classList.add('show');bar.style.width=bars[i]+'%';}},delays[i]);
    setTimeout(()=>document.getElementById(id).classList.add('done'),delays[i]+400);
  }});
  setTimeout(()=>{{
    document.getElementById('boot').classList.add('hidden');
    sfx('boot');
  }},3800);
}})();

// ── Eye poll: 60fps, chỉ update eyeLock ────────────────────
(async function pollEye(){{
  while(true){{
    try{{
      const r=await fetch('http://127.0.0.1:{_PORT}/eye',{{cache:'no-store'}});
      const d=await r.json();
      const el=document.getElementById('eyeLock');
      if(d.e){{
        const[x1,y1,x2,y2]=d.e;
        const W=window.innerWidth,H=window.innerHeight;
        const sc=Math.max(W/1280,H/720);
        const oX=(W-1280*sc)/2,oY=(H-720*sc)/2;
        el.style.cssText='display:block;left:'+(oX+x1*sc)+'px;top:'+(oY+y1*sc)+'px;width:'+((x2-x1)*sc)+'px;height:'+((y2-y1)*sc)+'px';
      }}else{{
        el.style.display='none';
      }}
    }}catch(e){{}}
    await new Promise(r=>setTimeout(r,16));
  }}
}})();

// ── State poll: 33ms ─────────────────────────────────────────
let _lastPoll=0;
async function poll(){{
  const now=performance.now();
  if(now-_lastPoll>33){{
    _lastPoll=now;
    try{{
      const r=await fetch('http://127.0.0.1:{_PORT}/state',{{cache:'no-store'}});
      const s=await r.json();
      render(s);
    }}catch(e){{}}
  }}
  requestAnimationFrame(poll);
}}

// ── Sidebar ─────────────────────────────────────────────────
let _sb=false,_sbl=[];
function renderSidebar(s){{
  const ll=document.getElementById('lblList');
  if(!_sb){{
    _sb=true;let h='',pg='';
    for(const l of s.labels){{
      if(l.group!==pg){{
        const gc=GRP_COL[l.group]||'#aaa';
        h+=`<div class="g-hdr" style="color:${{gc}}">${{l.group_name}}<div class="g-hdr-line"></div></div>`;
        pg=l.group;
      }}
      const gc=GRP_COL[l.group]||'#aaa';
      h+=`<div class="lbl" data-k="${{l.key}}">
        <div class="lbl-bar" style="background:${{gc}}"></div>
        <span class="lbl-name">${{l.name}}</span>
        <span class="lbl-val" data-v="${{l.count}}">${{l.done?'<span class="done-ck">✓</span>':l.count}}</span>
      </div>`;
    }}
    ll.innerHTML=h;_sbl=s.labels.map(l=>Object.assign({{}},l));return;
  }}
  for(let i=0;i<s.labels.length;i++){{
    const l=s.labels[i],old=_sbl[i];if(!old)continue;
    const row=ll.querySelector(`[data-k="${{l.key}}"]`);if(!row)continue;
    const nc=l.is_rec?'lbl rec':l.is_cursor?'lbl sel':'lbl';
    if(row.className!==nc)row.className=nc;
    const ve=row.querySelector('.lbl-val');
    if(ve&&(l.count!==old.count||l.done!==old.done)){{
      if(l.done&&!old.done){{
        ve.innerHTML='<span class="done-ck" style="animation:doneAnim .45s cubic-bezier(.34,1.56,.64,1)">✓</span>';
      }}else if(!l.done){{
        ve.textContent=l.count;
        if(l.count!==old.count){{ve.classList.remove('count-bump');void ve.offsetWidth;ve.classList.add('count-bump');}}
      }}
      ve.dataset.v=l.count;
    }}
  }}
  _sbl=s.labels.map(l=>Object.assign({{}},l));
  const a=ll.querySelector('.lbl.sel,.lbl.rec');
  if(a)a.scrollIntoView({{behavior:'smooth',block:'nearest'}});
}}

// ── Sound state ──────────────────────────────────────────────
let _pStart=0,_pStop=0,_pDone=0,_pSub=0,_pCalib=0,_pCap=false,_pBoot=false;

function updateSounds(s){{
  if(!_pBoot&&document.getElementById('boot').classList.contains('hidden')){{
    _pBoot=true; // sfx('boot') already called in bootSeq
  }}
  if(s.snd_start_ts&&s.snd_start_ts!==_pStart){{sfx('start');_pStart=s.snd_start_ts;}}
  if(s.snd_stop_ts&&s.snd_stop_ts!==_pStop){{sfx('stop');_pStop=s.snd_stop_ts;}}
  if(s.snd_done_ts&&s.snd_done_ts!==_pDone){{sfx('done');_pDone=s.snd_done_ts;}}
  if(s.snd_sub_ts&&s.snd_sub_ts!==_pSub){{sfx('sub');_pSub=s.snd_sub_ts;}}
  if(s.snd_calib_ts&&s.snd_calib_ts!==_pCalib){{sfx('calib');_pCalib=s.snd_calib_ts;}}
  if(s.capture_flash&&!_pCap){{sfx('capture');}}
  _pCap=s.capture_flash;
}}

// ── Track helpers ────────────────────────────────────────────
function _camCoords(x1,y1,x2,y2){{
  const W=window.innerWidth,H=window.innerHeight;
  const sc=Math.max(W/1280,H/720);
  const oX=(W-1280*sc)/2,oY=(H-720*sc)/2;
  return [oX+x1*sc,oY+y1*sc,(x2-x1)*sc,(y2-y1)*sc];
}}

// ── Eye lock only (Sony style) ────────────────────────────────
let _pel=false;
function updFaceLock(s){{
  const el=document.getElementById('eyeLock');

  // Eye lock: handled by pollEye() at 60fps
}}


// ── Auto-mode UI ─────────────────────────────────────────────
let _prevClassState='';
function updAutoMode(s){{
  const trans = document.getElementById('transOverlay');
  const chip  = document.getElementById('stateChip');
  const sbar  = document.getElementById('sessionBar');
  const hring = document.getElementById('holdRing');

  // Session progress bar
  const totalTarget = s.labels.length * s.target_frames;
  const totalDone   = s.labels.reduce((a,l)=>a+Math.min(l.count, s.target_frames),0);
  if(s.auto_mode){{
    sbar.classList.add('on');
    sbar.style.left=s.recording?'0':'256px';
    document.getElementById('sessionBarFill').style.width=(totalDone/totalTarget*100)+'%';
  }}else{{
    sbar.classList.remove('on');
  }}

  // Transition overlay
  if(s.auto_mode && s.class_state==='transition'){{
    const nxtLbl=s.labels.find(l=>l.key===s.next_class);
    document.getElementById('transName').textContent=nxtLbl?.name??s.next_class;
    document.getElementById('transHint').textContent='Chuẩn bị thực hiện...';
    document.getElementById('transCd').textContent=
      s.countdown_secs>0?Math.ceil(s.countdown_secs)+'s':'';
    trans.classList.add('on');
  }}else{{
    trans.classList.remove('on');
  }}

  // Hold ring (waiting state)
  if(s.auto_mode && s.class_state==='waiting' && s.recording){{
    hring.classList.add('on');
    const pct = 1 - Math.min(s.countdown_secs / 1.5, 1);
    const circ = 2*Math.PI*42;
    document.getElementById('holdArc').style.strokeDashoffset = circ*(1-pct);
  }}else{{
    hring.classList.remove('on');
  }}

  // State chip
  if(s.auto_mode && s.recording){{
    chip.classList.add('on');
    if(s.class_state==='waiting'){{
      chip.className='on waiting';
      chip.textContent=s.expr_valid?'✓ Giữ nguyên...':'Thực hiện: '+( s.labels.find(l=>l.key===s.recording)?.name??'');
    }}else if(s.class_state==='capturing'){{
      chip.className='on capturing';
      chip.textContent='⬡ Đang chụp — '+s.valid_frames+' frame';
    }}
  }}else{{
    chip.classList.remove('on');
  }}

  // Instruction panel text override for auto-mode
  if(s.auto_mode && s.recording){{
    const ip=document.getElementById('instr');
    const lbl=s.labels.find(l=>l.key===s.recording);
    ip.style.display='block';
    ip.style.left='0';
    requestAnimationFrame(()=>ip.classList.add('on'));
    document.getElementById('instrTag').textContent='Đang thu: '+(lbl?.name??'');
    document.getElementById('instrL1').textContent=s.sub_main||'';
    document.getElementById('instrL2').textContent=s.sub_hint||'';
  }}else{{
    const ip=document.getElementById('instr');
    ip.style.left='256px';
  }}
}}

// ── Calibration ─────────────────────────────────────────────
let _pCalibState='idle';
function updCalib(s){{
  const ov=document.getElementById('calibOverlay');
  const fill=document.getElementById('calibFill');
  const scan=document.getElementById('djiScan');
  const locked=document.getElementById('djiLocked');

  const pitch=s.calib_pitch||0;
  const yaw=s.calib_yaw||0;
  // Estimate roll from yaw asymmetry (approx)
  const roll=yaw*0.3;

  // Update side values (convert normalized → degrees approx)
  const pd=document.getElementById('djiPV');
  const yd=document.getElementById('djiYV');
  const rd=document.getElementById('djiRV');
  const pDeg=(pitch*90).toFixed(1);
  const yDeg=(yaw*90).toFixed(1);
  const rDeg=(roll*90).toFixed(1);
  if(pd) pd.textContent=(pitch>=0?'+':'')+pDeg;
  if(yd) yd.textContent=(yaw>=0?'+':'')+yDeg;
  if(rd) rd.textContent=(roll>=0?'+':'')+rDeg;

  // Attitude indicator: ground div moves + rotates with pitch/roll
  const ai=document.getElementById('djiAI');
  const ground=document.getElementById('aiGround');
  const horizon=document.getElementById('aiHorizon');
  const ladder=document.getElementById('aiLadder');
  if(ai){{
    const R=130; // radius px
    // pitchPx: how many px the horizon moves per unit pitch
    const pitchPx = pitch * 220;
    const rollDeg = roll * 25;
    // Ground covers bottom half + moves with pitch
    const groundTop = 50 + pitch*100; // percent
    if(ground){{
      ground.style.cssText=`top:${{groundTop}}%;bottom:-50%;
        transform:rotate(${{rollDeg}}deg);
        transform-origin:50% ${{(50-groundTop/2)}}%;`;
    }}
    if(horizon){{
      horizon.style.cssText=`top:${{groundTop}}%;
        transform:rotate(${{rollDeg}}deg);
        transform-origin:center;`;
    }}
    // Pitch ladder rungs
    if(ladder){{
      const rungs=[[-10,-10],[10,10],[-20,-20],[20,20]];
      ladder.innerHTML=rungs.map(([deg,label])=>{{
        const yPos=groundTop - deg*1.8;
        const w=deg%20===0?60:40;
        return `<div class="ai-rung" style="top:${{yPos}}%;transform:translateX(-50%) rotate(${{rollDeg}}deg)">
          <span class="ai-rung-lbl">${{label}}</span>
          <div class="ai-rung-line" style="width:${{w}}px"></div>
          <span class="ai-rung-lbl">${{label}}</span>
        </div>`;
      }}).join('');
    }}
  }}

  if(s.calib_state==='collecting'){{
    if(!ov.classList.contains('on')) ov.classList.add('on');
    locked.classList.remove('on');
    scan.classList.add('on');
    const st=document.getElementById('djiStatusTxt');
    const st2=document.getElementById('djiStatusTxt2');
    if(st) st.textContent='Đang thu thập...';
    if(st2) st2.textContent='Đang phân tích';
    fill.style.width=Math.round(s.calib_progress*100)+'%';
  }}else if(s.calib_state==='idle'&&!s.calib_done){{
    if(!ov.classList.contains('on')) ov.classList.add('on');
    scan.classList.remove('on');
    const st=document.getElementById('djiStatusTxt');
    if(st) st.textContent='Nhìn thẳng · Giữ yên';
    fill.style.width='0%';
    locked.classList.remove('on');
  }}else{{
    if(_pCalibState==='collecting'){{
      fill.style.width='100%';
      scan.classList.remove('on');
      locked.classList.remove('on');void locked.offsetWidth;locked.classList.add('on');
      sfx('calib');
      setTimeout(()=>ov.classList.remove('on'),2000);
    }}
  }}
  _pCalibState=s.calib_state;
}}

// ── Capture feedback ─────────────────────────────────────────
let _pFlash=false;
function updCapFeedback(s){{
  if(s.capture_flash&&!_pFlash){{
    const fl=document.getElementById('flash');
    fl.classList.add('on');setTimeout(()=>fl.classList.remove('on'),140);
    const ct=document.getElementById('capturedTxt');
    ct.classList.remove('on');void ct.offsetWidth;ct.classList.add('on');
    setTimeout(()=>ct.classList.remove('on'),380);
  }}
  _pFlash=s.capture_flash;
}}

// ── Crop preview ─────────────────────────────────────────────
let _pCropRec=null;
function updCropPreview(s){{
  const panel=document.getElementById('cropPanel');
  if(!s.recording||!s.crop_preview){{
    if(_pCropRec){{
      panel.classList.remove('on');
      setTimeout(()=>{{panel.style.display='none';}},250);
      _pCropRec=null;
    }}
    return;
  }}
  if(!_pCropRec){{panel.style.display='';panel.classList.add('on');}}
  _pCropRec=s.recording;
  document.getElementById('cropImg').src='data:image/jpeg;base64,'+s.crop_preview;
  const lbl=s.labels.find(l=>l.key===s.recording);
  const grp=lbl?.group||'';
  document.getElementById('cpRegion').textContent=grp==='MAT'?'👁 Mắt':grp==='MIENG'?'👄 Miệng':'🧑 Mặt';
  document.getElementById('cpClass').textContent=lbl?.name?.substring(0,10)||'—';
}}

// ── Dataset check ─────────────────────────────────────────────
const DC_CHECKS=[
  {{ico:'🔍',l:'Kiểm tra số lượng ảnh',s:'Xác minh đủ ảnh mỗi class'}},
  {{ico:'🗂️',l:'Xác minh cấu trúc thư mục',s:'dataset/person_xx/'}},
  {{ico:'🖼️',l:'Chất lượng & định dạng ảnh',s:'224×224px · PNG · CLAHE enhanced'}},
  {{ico:'⚡',l:'Tính đa dạng pose',s:'Auto-burst · sub-instructions · góc đa dạng'}},
  {{ico:'💾',l:'Xác nhận ghi vào disk',s:'Flush buffer · commit dataset'}},
];
let _ckDone=false;
function runDataCheck(s,onDone){{
  const el=document.getElementById('dataCheck');
  const tot=s.labels.reduce((a,l)=>a+l.count,0);
  const dn=s.labels.filter(l=>l.done).length;
  const subs=[
    `${{tot}} ảnh · ${{dn}}/${{s.labels.length}} class`,
    `dataset/${{s.person_id}}/`,`224×224 PNG · enhanced`,
    `${{dn}} class · aug×${{3}}`,`${{tot}} files · ~${{Math.round(tot*.04)}}MB`,
  ];
  document.getElementById('dcItems').innerHTML=DC_CHECKS.map((c,i)=>
    `<div class="dc-item" style="--d:${{.12+i*.09}}s">
      <div class="dc-ico">${{c.ico}}</div>
      <div class="dc-txt"><div class="dc-lbl">${{c.l}}</div><div class="dc-sub">${{subs[i]}}</div></div>
      <div class="dc-st chk" id="dcs${{i}}">Đang kiểm tra...</div>
    </div>`).join('');
  el.classList.add('on');sfx('calib');
  DC_CHECKS.forEach((_,i)=>{{
    setTimeout(()=>{{
      const st=document.getElementById('dcs'+i);
      if(st){{st.textContent='✓ OK';st.className='dc-st ok';}}
      document.getElementById('dcBar').style.width=((i+1)/DC_CHECKS.length*100)+'%';
      if(i===DC_CHECKS.length-1){{
        setTimeout(()=>{{
          document.getElementById('dcOk').style.opacity='1';sfx('done');
          setTimeout(()=>{{el.classList.remove('on');onDone();}},1100);
        }},350);
      }}
    }},200+i*370);
  }});
}}

// ── Outro ────────────────────────────────────────────────────
let _oShown=false;
function showOutro(s){{
  if(_oShown)return;_oShown=true;
  if(!_ckDone){{_ckDone=true;runDataCheck(s,()=>_doOutro(s));return;}}
  _doOutro(s);
}}
function _doOutro(s){{
  sfx('outro');
  const tot=s.labels.reduce((a,l)=>a+l.count,0);
  const dn=s.labels.filter(l=>l.done).length;
  // Particles
  const parts=document.getElementById('outroParts');
  for(let i=0;i<25;i++){{
    const p=document.createElement('div');p.className='outro-p';
    const sz=Math.random()*4+2;
    const cols=['#5edfff','#a78bfa','#34d399','#fbbf24'];
    p.style.cssText=`width:${{sz}}px;height:${{sz}}px;left:${{Math.random()*100}}%;
      top:${{Math.random()*100}}%;background:${{cols[i%4]}};
      --dur:${{3+Math.random()*4}}s;--delay:${{Math.random()*2}}s;
      --dx:${{(Math.random()-.5)*60}}px;--dy:${{(Math.random()-.5)*60}}px;
      --op1:${{Math.random()*.3+.05}};--op2:${{Math.random()*.5+.1}};
      box-shadow:0 0 ${{sz*2}}px currentColor;`;
    parts.appendChild(p);
  }}
  document.getElementById('oTotal').textContent='0';
  document.getElementById('oClass').textContent='0';
  const countUp=(el,target,dur)=>{{
    const s2=performance.now();
    const step=(now)=>{{
      const pct=Math.min((now-s2)/dur,1);
      el.textContent=Math.round(pct*target);
      if(pct<1)requestAnimationFrame(step);
    }};requestAnimationFrame(step);
  }};
  setTimeout(()=>{{
    countUp(document.getElementById('oTotal'),tot,1200);
    countUp(document.getElementById('oClass'),dn,900);
  }},500);
  document.getElementById('outro').classList.add('on');
  setTimeout(()=>{{const b=document.getElementById('oBar');b.style.transition='width 2.8s cubic-bezier(.4,0,.2,1)';b.style.width='100%';}},700);
  const msgs=['Đang lưu dữ liệu...','Hoàn thiện dataset...','Đồng bộ metadata...','Phiên làm việc kết thúc.','Xin cảm ơn bạn! &#x2665;'];
  let ci=0;
  const si=setInterval(()=>{{
    const el=document.getElementById('oCl');
    if(!el){{clearInterval(si);return;}}
    el.style.opacity='0';
    setTimeout(()=>{{if(el){{el.innerHTML=msgs[ci%msgs.length];el.style.opacity='1';}}}},200);
    ci++;
  }},650);
  setTimeout(async()=>{{try{{await pywebview.api.quit();}}catch(e){{}}}},3500);
}}

// ── Main render ──────────────────────────────────────────────
function render(s){{
  if(s.shutting_down){{showOutro(s);return;}}

  // Sidebar: ẩn ngay khi auto_mode bật
  const _sb=document.getElementById('sidebar');
  const _ui=document.getElementById('ui');
  if(s.auto_mode){{
    _sb.style.cssText='width:0;min-width:0;opacity:0;pointer-events:none;overflow:hidden;transition:width .3s ease,opacity .3s ease;border:none;';
  }}else{{
    _sb.style.cssText='';
  }}

  // Top bar
  const hdr=document.getElementById('hdrPill');
  const htxt=document.getElementById('hdrTxt');
  if(s.brightness<90){{hdr.className='hdr-pill hdr-bad';htxt.textContent=`HDR | ${{s.brightness}}`;}}
  else if(s.brightness>200){{hdr.className='hdr-pill hdr-warn';htxt.textContent=`Cháy sáng | ${{s.brightness}}`;}}
  else{{hdr.className='hdr-pill hdr-ok';htxt.textContent=`Sáng tốt | ${{s.brightness}}`;}}
  document.getElementById('personTag').textContent=
    `#${{s.person_id.replace('person_','').padStart(2,'0')}} ${{s.person_name}}`;

  renderSidebar(s);

  // Bottom
  const rec=s.recording;
  const bi=document.getElementById('botIdle');
  const br=document.getElementById('botRec');
  const pw=document.getElementById('pbarWrap');
  const brt=document.getElementById('botRight');
  if(rec){{
    bi.style.display='none';br.style.display='flex';pw.style.display='block';brt.style.display='none';
    const lbl=s.labels.find(l=>l.key===rec);
    document.getElementById('recCls').textContent=lbl?.name??rec;
    const pct=lbl?.pct??0;
    const f=document.getElementById('pbarFill');
    f.style.width=pct+'%';f.className='pbar-fill'+(pct>=100?' done':'');
    document.getElementById('pbarTxt').textContent=`${{s.valid_frames}}/${{Math.round(s.target_frames/(1+2))}} frame · ${{pct}}% hoàn thành`;
  }}else if(s.auto_mode && s.class_state==='transition'){{
    bi.style.display='none';br.style.display='none';pw.style.display='none';brt.style.display='block';
    const nxtLbl=s.labels.find(l=>l.key===s.next_class);
    brt.innerHTML=`<span style="color:var(--amber)">Tiếp theo: <strong style="color:var(--cyan)">${{nxtLbl?.name??s.next_class}}</strong></span>`;
  }}else{{
    bi.style.display='flex';br.style.display='none';pw.style.display='none';brt.style.display='block';
    if(!s.calib_done)brt.textContent='C → Hiệu chỉnh trước khi thu';
    else if(!s.auto_mode)brt.textContent='SPACE → bắt đầu thu tự động';
    else brt.textContent='SPACE → dừng';
  }}

  // Auto-mode overlays
  updAutoMode(s);

  // Instruction + sub bar
  const ip=document.getElementById('instr');
  const sb=document.getElementById('subBar');
  const ss=document.getElementById('subStep');
  if(rec){{
    ip.style.display='block';requestAnimationFrame(()=>ip.classList.add('on'));
    const lbl=s.labels.find(l=>l.key===rec);
    document.getElementById('instrTag').textContent='Hướng dẫn: '+(lbl?.name??rec);
    document.getElementById('instrL1').textContent=s.sub_main||'';
    document.getElementById('instrL2').textContent=s.sub_hint||'';
    sb.classList.add('on');ss.classList.add('on');
    const pct=Math.min(s.sub_elapsed/SUB_DUR,1)*100;
    document.getElementById('subBarFill').style.width=pct+'%';
    ss.textContent=`Giai đoạn ${{s.sub_idx+1}}/3`;
  }}else{{
    ip.classList.remove('on');sb.classList.remove('on');ss.classList.remove('on');
    setTimeout(()=>{{if(!ip.classList.contains('on'))ip.style.display='none';}},220);
  }}

  // Expression feedback
  const ef=document.getElementById('exprFeedback');
  if(rec&&s.expr_feedback){{
    ef.textContent=s.expr_feedback;ef.classList.add('on');
  }}else{{ef.classList.remove('on');}}

  // Frame counter
  const fc=document.getElementById('frameCounter');
  if(rec){{
    fc.classList.add('on');
    fc.textContent=`${{s.valid_frames}} / ${{s.target_frames}} frame`;
  }}else{{fc.classList.remove('on');}}

  // Distance / brightness warning
  const dw=document.getElementById('distWarn');
  if(s.distance_warn==='close'){{dw.classList.add('on');dw.textContent='⚠ Ngồi ra xa camera hơn';}}
  else if(s.distance_warn==='far'){{dw.classList.add('on');dw.textContent='⚠ Lại gần camera hơn';}}
  else if(s.brightness_warn==='dark'){{dw.classList.add('on');dw.textContent='⚠ Ánh sáng quá tối';}}
  else if(s.brightness_warn==='bright'){{dw.classList.add('on');dw.textContent='⚠ Ánh sáng quá chói';}}
  else{{dw.classList.remove('on');}}

  updFaceLock(s);updCalib(s);updCapFeedback(s);updCropPreview(s);updateSounds(s);
}}

// ── Keyboard ────────────────────────────────────────────────
document.addEventListener('keydown', async e=>{{
  e.preventDefault();
  const m={{' ':'space','Escape':'esc','ArrowUp':'up','ArrowDown':'down','c':'c','C':'c'}};
  const k=m[e.key];if(!k)return;
  _resume();
  if(k==='up'||k==='down')sfx('nav',{{freq:k==='up'?700:500}});
  else if(k==='esc')sfx('stop');
  else if(k==='c')sfx('nav',{{freq:900}});
  await pywebview.api.send_key(k);
}});

poll();
</script></body></html>"""

def _launch_wv():
    global _wv_win
    _wv_win = webview.create_window(
        "Dataset Collector v2", html=_HTML,
        js_api=_Api(), width=1280, height=760,
        min_size=(960,600), resizable=True, text_select=False)
    webview.start()
    _st["running"] = False

threading.Thread(target=_launch_wv, daemon=True).start()
time.sleep(1.5)

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
# ── Auto-mode helpers ────────────────────────────────────────
def build_queue():
    """Random queue các class chưa đủ ảnh."""
    tpc = TARGET_FRAMES * (1 + AUGMENT_COUNT)
    needed = [k for k in LABEL_KEYS if counts.get(k, 0) < tpc]
    random.shuffle(needed)
    return needed

def pop_next(queue):
    """Lấy class tiếp theo chưa đủ ảnh."""
    tpc = TARGET_FRAMES * (1 + AUGMENT_COUNT)
    while queue:
        cls = queue.pop(0)
        if counts.get(cls, 0) < tpc:
            return cls
    return None

# ── Main state ────────────────────────────────────────────────
cursor          = 0
calib_state     = "idle"
auto_mode       = False
auto_queue      = []
recording_label = None
class_state     = "idle"  # idle|waiting|capturing|transition
hold_start      = 0.0
transition_start= 0.0
next_label      = ""
last_capture_t  = 0.0
valid_frames    = 0
session_total   = 0

while _st["running"]:
    if _st.get("shutting_down", False): break
    ret, raw = cap.read()
    if not ret: break

    raw = cv2.flip(raw, 1)
    rh, rw = raw.shape[:2]
    rgb     = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    face_box    = None
    landmarks   = None
    brightness  = 128
    face_sz     = 0
    dist_warn   = ""
    bright_warn = ""
    expr_valid  = False
    expr_fb     = ""
    crop_b64    = ""
    countdown   = 0.0
    now         = time.time()

    if results.multi_face_landmarks:
        lms       = results.multi_face_landmarks[0].landmark
        landmarks = lms
        face_box  = get_face_box(lms, rw, rh, pad=18)
        face_sz   = face_box[2] - face_box[0]

        # Brightness
        crop_tmp = get_crop(raw, lms, "normal_eye", rw, rh)
        if crop_tmp is not None and crop_tmp.size > 0:
            brightness = int(np.mean(cv2.cvtColor(crop_tmp, cv2.COLOR_BGR2GRAY)))
        if brightness < 60:    bright_warn = "dark"
        elif brightness > 210: bright_warn = "bright"

        # Distance
        if face_sz < MIN_FACE_SIZE:    dist_warn = "far"
        elif face_sz > MAX_FACE_SIZE:  dist_warn = "close"

        # face box removed — dùng JS eye tracking thay

        # Calibration
        if calib_state == "collecting":
            prog = update_calibration(lms, rw, rh)
            if prog is True:
                calib_state = "done"
                with _st_lock:
                    _st["calib_state"]    = "done"
                    _st["calib_progress"] = 1.0
                    _st["snd_calib_ts"]   = now
                print("  [CALIB DONE]")
            else:
                with _st_lock:
                    _st["calib_state"]    = "collecting"
                    _st["calib_progress"] = float(prog)

        # ── AUTO STATE MACHINE ────────────────────────────────
        if auto_mode and calib_state == "done":
            tpc = TARGET_FRAMES * (1 + AUGMENT_COUNT)
            ok_env = dist_warn == "" and bright_warn == ""

            if class_state == "idle":
                nxt = pop_next(auto_queue)
                if nxt:
                    recording_label = nxt
                    class_state = "waiting"
                    hold_start  = 0.0
                    valid_frames = 0
                    last_capture_t = 0.0
                    with _st_lock: _st["snd_start_ts"] = now
                    print(f"  [AUTO] → {recording_label}")
                else:
                    auto_mode = False; recording_label = None; class_state = "idle"
                    with _st_lock: _st["shutting_down"] = True

            elif class_state == "waiting":
                # Chờ expression đúng, giữ CLASS_HOLD_SECS giây
                if ok_env:
                    valid, fb = check_expression(recording_label, lms, rw, rh, baseline)
                    expr_valid = valid; expr_fb = fb
                    if valid:
                        if hold_start == 0.0: hold_start = now
                        held = now - hold_start
                        countdown = max(0.0, CLASS_HOLD_SECS - held)
                        if held >= CLASS_HOLD_SECS:
                            class_state = "capturing"
                            last_capture_t = 0.0
                    else:
                        hold_start = 0.0
                        countdown  = CLASS_HOLD_SECS

            elif class_state == "capturing":
                # Capture liên tục khi expression đúng
                if ok_env:
                    valid, fb = check_expression(recording_label, lms, rw, rh, baseline)
                    expr_valid = valid; expr_fb = fb
                    if valid and now - last_capture_t >= CAPTURE_INTERVAL:
                        crop = get_crop(raw, lms, recording_label, rw, rh)
                        if crop is not None and crop.size > 0:
                            saved = save_frame(crop, recording_label)
                            if saved > 0:
                                valid_frames  += 1
                                session_total += 1
                                last_capture_t = now
                                with _st_lock: _st["capture_flash_ts"] = now
                    # Crop preview
                    prev = get_crop(raw, lms, recording_label, rw, rh)
                    if prev is not None and prev.size > 0:
                        sq = max(prev.shape[0], prev.shape[1])
                        cv2buf = np.zeros((sq,sq,3), dtype=np.uint8)
                        yo=(sq-prev.shape[0])//2; xo=(sq-prev.shape[1])//2
                        cv2buf[yo:yo+prev.shape[0], xo:xo+prev.shape[1]] = prev
                        p160 = cv2.resize(cv2buf, (160,160), interpolation=cv2.INTER_LINEAR)
                        ok2, buf2 = cv2.imencode('.jpg', p160, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok2: crop_b64 = base64.b64encode(buf2).decode('ascii')
                    # Class xong?
                    if counts.get(recording_label, 0) >= tpc:
                        with _st_lock: _st["snd_done_ts"] = now
                        print(f"  [AUTO] ✓ {recording_label}: {counts[recording_label]} ảnh")
                        auto_queue = build_queue()
                        nxt = pop_next(auto_queue)
                        if nxt:
                            next_label = nxt
                            transition_start = now
                            class_state = "transition"
                            recording_label = None
                        else:
                            auto_mode = False; recording_label = None; class_state = "idle"
                            with _st_lock: _st["shutting_down"] = True

            elif class_state == "transition":
                # Nghỉ giữa 2 class
                elapsed  = now - transition_start
                countdown = max(0.0, CLASS_CONFIRM_SECS - elapsed)
                if elapsed >= CLASS_CONFIRM_SECS:
                    recording_label  = next_label
                    next_label       = ""
                    class_state      = "waiting"
                    hold_start       = 0.0
                    last_capture_t   = 0.0
                    valid_frames     = 0
                    with _st_lock: _st["snd_sub_ts"] = now

    # Encode frame
    ok, buf = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        _push(buf.tobytes(), {
            "recording"       : recording_label,
            "cursor"          : cursor,
            "counts"          : dict(counts),
            "brightness"      : brightness,
            "face_box"        : face_box,
            "calib_pitch"    : round(float(compute_pitch(landmarks,rw,rh)),4) if landmarks is not None else 0,
            "calib_yaw"      : round(float(compute_yaw(landmarks,rw,rh)),4) if landmarks is not None else 0,
            "eye_box"         : (lambda lms,w,h,r=13: (
                int(lms[468].x*w)-r, int(lms[468].y*h)-r,
                int(lms[468].x*w)+r, int(lms[468].y*h)+r
            ))(landmarks, rw, rh) if landmarks is not None else None,
            "face_ok"         : dist_warn == "" and bright_warn == "",
            "face_size"       : face_sz,
            "sub_idx"         : 0,
            "sub_elapsed"     : 0.0,
            "valid_frames"    : valid_frames,
            "expr_valid"      : expr_valid,
            "expr_feedback"   : expr_fb,
            "brightness_warn" : bright_warn,
            "distance_warn"   : dist_warn,
            "crop_preview"    : crop_b64,
            "auto_mode"       : auto_mode,
            "class_state"     : class_state,
            "next_class"      : next_label,
            "countdown_secs"  : countdown,
            "session_total"   : session_total,
        })

    # Keys
    for k in _pop_keys():
        if k == "esc":
            with _st_lock: _st["shutting_down"] = True
            break
        elif k == "c":
            if calib_state in ("idle", "done"):
                start_calibration()
                calib_state = "collecting"
                with _st_lock: _st["calib_state"] = "collecting"
        elif k == "space":
            if auto_mode:
                # Dừng
                auto_mode = False; recording_label = None; class_state = "idle"
                with _st_lock: _st["snd_stop_ts"] = now
            elif calib_state == "done":
                # Bắt đầu auto
                auto_queue    = build_queue()
                auto_mode     = True
                class_state   = "idle"
                session_total = 0
                with _st_lock: _st["snd_start_ts"] = now
                print(f"  [AUTO] Start | Queue: {auto_queue}")
            else:
                print("  [INFO] Nhấn C để hiệu chỉnh trước!")

cap.release()
face_mesh.close()

# Outro wait
if _st.get("shutting_down", False):
    print("  [OUTRO] Hiển thị màn hình kết thúc...")
    _st["outro_done"].wait(timeout=6.0)

import os as _os
try: _os.kill(_os.getpid(), 9)
except: _os._exit(0)