from ultralytics import YOLO

def main():
    # 🟢 نستخدم موديل أكبر شوية من s: نستخدم m
    model = YOLO("yolov8m-seg.pt")   # لو مش محمّل رح ينزل أوتوماتيك

    model.train(
        data="data.yaml",        # نفس الملف اللي سكربت prepare_dataset.py عمله
        task="segment",          # إحنا شغالين Segmentation
        epochs=200,              # عدد epochs أكبر للتدريب الاحترافي
        imgsz=768,               # صورة أكبر شوي لتحسين دقة الخزانات
        batch=4,                 # batch مناسب لكرت RTX 3050
        name="oil_tanks_seg_pro",# اسم تجربة التدريب الجديدة
        project="runs",          # المجلد اللي يحفظ فيه النتايج
        device=0,                # 🟢 استخدم GPU (RTX 3050)
        patience=40,             # EarlyStopping لو ما تحسّن 40 epoch يوقف
        workers=2,               # عدد الــ workers للـ dataloader
        save=True,
        pretrained=True,         # نبدأ من وزن جاهز
        verbose=True
    )

if __name__ == "__main__":
    main()
