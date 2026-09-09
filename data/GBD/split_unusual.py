import os
import csv

# ====== 1) Update to your dataset root directory ======
DATASET_ROOT = r"path/to/Gastrovision"

# ====== 2) 类别名 -> label(0~26) 的映射顺序（按你截图那27个文件夹名）======
CLASS_NAMES = [
    "Accessory tools",
    "Angiectasia",
    "Barrett's esophagus",
    "Blood in lumen",
    "Cecum",
    "Colon diverticula",
    "Colon polyps",
    "Colorectal cancer",
    "Duodenal bulb",
    "Dyed-lifted-polyps",
    "Dyed-resection-margins",
    "Erythema",
    "Esophageal varices",
    "Esophagitis",
    "Gastric polyps",
    "Gastroesophageal_junction_normal z-line",
    "Ileocecal valve",
    "Mucosal inflammation large bowel",
    "Normal esophagus",
    "Normal mucosa and vascular pattern in the large bowel",
    "Normal stomach",
    "Pylorus",
    "Resected polyps",
    "Resection margins",
    "Retroflex rectum",
    "Small bowel_terminal ileum",
    "Ulcer",
]

# 输出文件
OUT_LIST_CSV = os.path.join(DATASET_ROOT, "gastrovision_label.csv")
OUT_COUNT_CSV = os.path.join(DATASET_ROOT, "gastrovision_class_counts.csv")

# 允许的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def main():
    class_to_label = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    rows = []
    counts = {name: 0 for name in CLASS_NAMES}
    missing_dirs = []

    for class_name, label in class_to_label.items():
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_dir):
            missing_dirs.append(class_dir)
            continue

        for root, _, files in os.walk(class_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    # 你要的格式：类别名 + 图片名（如果有子目录，也一并保留）
                    rel_under_class = os.path.relpath(os.path.join(root, fn), start=class_dir)
                    rel_path = os.path.join(class_name, rel_under_class).replace("\\", "/")

                    rows.append([rel_path, label])
                    counts[class_name] += 1

    # 1) 保存 path(类别名/图片名) + label 的CSV
    with open(OUT_LIST_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    # 2) 保存每类图片数量统计CSV
    with open(OUT_COUNT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "label", "count"])
        for class_name, label in class_to_label.items():
            writer.writerow([class_name, label, counts.get(class_name, 0)])

    print(f"[OK] 共收集到 {len(rows)} 张图片")
    print(f"[OK] 列表CSV: {OUT_LIST_CSV}")
    print(f"[OK] 统计CSV: {OUT_COUNT_CSV}")

    if missing_dirs:
        print("\n[WARN] 以下类别文件夹未找到（请检查名字/空格/大小写是否一致）：")
        for d in missing_dirs:
            print("  -", d)

if __name__ == "__main__":
    main()
