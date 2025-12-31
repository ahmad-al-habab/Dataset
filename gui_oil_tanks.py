# gui_oil_tanks.py
# GUI بالـ Tkinter لاختيار صورة وتشغيل موديل YOLO (GPU فقط) مع واجهة محسّنة

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# تحميل المودل مرة واحدة (على كرت الشاشة GPU:0)
# عدّل المسار حسب مكان best.pt عندك
# --------------------------------------------------
MODEL_PATH = "runs/oil_tanks_seg_pro/weights/best.pt"
model = YOLO(MODEL_PATH)

# --------------------------------------------------
# متغيّرات عالمية للـ GUI
# --------------------------------------------------
img_path = None            # مسار الصورة الأصلية
input_img_tk = None        # الصورة في يسار الواجهة
output_img_tk = None       # الصورة الناتجة في يمين الواجهة


def load_image():
    """فتح صورة من الجهاز وعرضها في اللوحة اليسار."""
    global img_path, input_img_tk, output_img_tk

    filename = filedialog.askopenfilename(
        title="اختر صورة",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
    )
    if not filename:
        return

    img_path = filename

    # نقرأ الصورة ونحوّلها لـ PIL لعرضها في Tkinter
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("❌ خطأ في قراءة الصورة")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # نغيّر الحجم فقط للعرض (لا يؤثر على الكشف)
    img_pil = img_pil.resize((512, 512))
    input_img_tk = ImageTk.PhotoImage(img_pil)

    panel_input_img.config(image=input_img_tk)
    panel_input_img.image = input_img_tk

    # نفضي لوحة النتيجة
    panel_output_img.config(image=None)
    panel_output_img.image = None
    panel_output_text.config(text="النتيجة بعد الكشف")


def detect_image():
    """تشغيل المودل على الصورة الحالية وعرض النتيجة مع البوكسات فقط (أزرق + نص أحمر)."""
    global img_path, output_img_tk

    if not img_path:
        print("⚠ يرجى اختيار صورة أولاً")
        return

    # قراءة الصورة الأصلية (BGR)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("❌ خطأ في قراءة الصورة")
        return

    # نأخذ قيمة الـ confidence من السلايدر
    conf_th = conf_scale.get()

    # تشغيل المودل (Segmentation) لكن سنستخدم البوكسات فقط للعرض
    results = model.predict(
        img_bgr,
        task="segment",
        imgsz=512,       # حجم الإدخال للمودل (أصغر = استهلاك أقل للذاكرة)
        conf=conf_th,    # من السلايدر
        device=0,        # دائماً GPU
        verbose=False
    )

    r = results[0]

    # نحول الصورة إلى RGB (للعرض)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # لو فيه بوكسات، ارسمها
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            # مستطيل أزرق واضح (RGB = أزرق)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # نكتب (tank + confidence) فوق البوكس باللون الأحمر
            if scores is not None:
                conf_val = float(scores[i])
                label = f"tank {conf_val:.2f}"

                # ممكن نرسم خلفية صغيرة سوداء للنص عشان يكون أوضح
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                y_text = max(y1 - 10, 0)
                cv2.rectangle(
                    img_rgb,
                    (x1, y_text - th - 4),
                    (x1 + tw + 4, y_text + 2),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    img_rgb, label, (x1 + 2, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA
                )

    # للصورة الناتجة: نغير الحجم فقط للعرض
    result_pil = Image.fromarray(img_rgb)
    result_pil = result_pil.resize((512, 512))
    output_img_tk = ImageTk.PhotoImage(result_pil)

    panel_output_img.config(image=output_img_tk)
    panel_output_img.image = output_img_tk
    panel_output_text.config(text="النتيجة بعد الكشف")

    print("✅ Detection done.")


# --------------------------------------------------
# إنشاء واجهة Tkinter
# --------------------------------------------------
root = tk.Tk()
root.title("Project 1 Oil Tanks")
root.configure(bg="#0f172a")  # خلفية داكنة

# حجم مبدئي للنافذة
root.geometry("1200x700")

# ----------------- الإطار العلوي (الصور) -----------------
top_frame = tk.Frame(root, bg="#0f172a")
top_frame.pack(padx=15, pady=15, fill="both", expand=True)

# إطار للصورة الأصلية
left_frame = tk.LabelFrame(
    top_frame,
    text="الصورة الأصلية",
    bg="#0f172a",
    fg="white",
    font=("Segoe UI", 11, "bold"),
    bd=2,
    labelanchor="n"
)
left_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)

panel_input_img = tk.Label(left_frame, bg="#1f2937")
panel_input_img.pack(padx=10, pady=10, fill="both", expand=True)

# إطار للصورة الناتجة
right_frame = tk.LabelFrame(
    top_frame,
    text="النتيجة",
    bg="#0f172a",
    fg="white",
    font=("Segoe UI", 11, "bold"),
    bd=2,
    labelanchor="n"
)
right_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)

panel_output_img = tk.Label(right_frame, bg="#1f2937")
panel_output_img.pack(padx=10, pady=(10, 0), fill="both", expand=True)

panel_output_text = tk.Label(
    right_frame,
    text="النتيجة بعد الكشف",
    bg="#0f172a",
    fg="#9ca3af",
    font=("Segoe UI", 10)
)
panel_output_text.pack(pady=8)

# ----------------- إطار الكنترولات (الأزرار + السلايدر) -----------------
control_frame = tk.Frame(root, bg="#0b1120")
control_frame.pack(fill="x", padx=15, pady=(0, 15))

# زر اختيار صورة
btn_load = tk.Button(
    control_frame,
    text="📂 اختر صورة",
    command=load_image,
    width=18,
    bg="#1e293b",
    fg="white",
    activebackground="#334155",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    relief="ridge",
    bd=2
)
btn_load.grid(row=0, column=0, padx=8, pady=8, sticky="w")

# زر الكشف
btn_detect = tk.Button(
    control_frame,
    text="🚀 ابدأ الكشف",
    command=detect_image,
    width=18,
    bg="#0ea5e9",
    fg="white",
    activebackground="#0284c7",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    relief="ridge",
    bd=2
)
btn_detect.grid(row=0, column=1, padx=8, pady=8, sticky="w")

# سطر السلايدر
lbl_conf = tk.Label(
    control_frame,
    text="Confidence Threshold (عتبة الثقة):",
    bg="#0b1120",
    fg="#e5e7eb",
    font=("Segoe UI", 10)
)
lbl_conf.grid(row=1, column=0, padx=8, pady=5, sticky="w")

conf_scale = tk.Scale(
    control_frame,
    from_=0.1,
    to=0.9,
    orient=tk.HORIZONTAL,
    resolution=0.01,
    length=350,
    bg="#0b1120",
    fg="white",
    troughcolor="#1f2937",
    highlightthickness=0
)
conf_scale.set(0.4)  # قيمة افتراضية
conf_scale.grid(row=1, column=1, padx=8, pady=5, sticky="w")

# ----------------- رسالة توضيحية تحت -----------------
info_label = tk.Label(
    root,
    text="With GPU",
    bg="#0f172a",
    fg="#9ca3af",
    font=("Segoe UI", 9)
)
info_label.pack(padx=15, pady=(0, 10))

# تشغيل الواجهة
root.mainloop()
