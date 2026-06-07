import os
import shutil
from pathlib import Path
from tqdm import tqdm

RAW_DATASET_DIR = Path("trash_project/dataset/raw/garbage_classification")
FINAL_DATASET_DIR = Path("trash_project/dataset/final")

FINAL_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

CLASS_MAPPING = {
    "cardboard": "cardboard",
    "paper": "paper",
    "metal": "metal",
    "plastic": "plastic",

    "green-glass": "glass",
    "brown-glass": "glass",
    "white-glass": "glass",
    "glass": "glass",

    "biological": "trash",
    "trash": "trash",
    "clothes": "trash",
    "shoes": "trash",
    "batteries": "trash",
    "battery": "trash"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def reset_final_folder():
    if FINAL_DATASET_DIR.exists():
        shutil.rmtree(FINAL_DATASET_DIR)

    for class_name in FINAL_CLASSES:
        class_dir = FINAL_DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)


def find_class_folders(raw_dir):
    found = {}

    for root, dirs, files in os.walk(raw_dir):
        root_path = Path(root)
        folder_name = root_path.name.lower().strip()

        if folder_name in CLASS_MAPPING:
            image_files = [
                file for file in root_path.iterdir()
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if len(image_files) > 0:
                found[folder_name] = root_path

    return found


def copy_images_to_final(class_folders):
    counts = {class_name: 0 for class_name in FINAL_CLASSES}

    for source_class, source_folder in class_folders.items():
        target_class = CLASS_MAPPING[source_class]
        target_folder = FINAL_DATASET_DIR / target_class

        image_files = [
            file for file in source_folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\nMapping {source_class} → {target_class}")
        print(f"Images found: {len(image_files)}")

        for image_file in tqdm(image_files):
            counts[target_class] += 1

            new_filename = f"{target_class}_{source_class}_{counts[target_class]:06d}{image_file.suffix.lower()}"
            target_path = target_folder / new_filename

            shutil.copy2(image_file, target_path)

    return counts


def print_final_counts():
    print("\nFinal dataset count:")

    total = 0

    for class_name in FINAL_CLASSES:
        class_dir = FINAL_DATASET_DIR / class_name

        files = [
            file for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"{class_name}: {len(files)}")
        total += len(files)

    print("Total:", total)


def main():
    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset mentah tidak ditemukan: {RAW_DATASET_DIR}\n"
            "Pastikan dataset Kaggle sudah diextract ke folder tersebut."
        )

    print("Searching class folders...")
    class_folders = find_class_folders(RAW_DATASET_DIR)

    print("\nDetected class folders:")
    for class_name, path in class_folders.items():
        print(f"{class_name}: {path}")

    missing_classes = []

    for source_class in CLASS_MAPPING.keys():
        if source_class not in class_folders:
            missing_classes.append(source_class)

    important_classes = [
        "cardboard",
        "paper",
        "metal",
        "plastic",
        "green-glass",
        "brown-glass",
        "white-glass",
        "biological",
        "trash"
    ]

    important_missing = [
        class_name for class_name in important_classes
        if class_name not in class_folders
    ]

    if len(important_missing) > 0:
        print("\nWarning: beberapa folder penting tidak ditemukan:")
        for class_name in important_missing:
            print("-", class_name)

        print("\nCek lagi struktur folder hasil extract Kaggle kamu.")

    reset_final_folder()

    copy_images_to_final(class_folders)

    print_final_counts()

    print("\nDataset final berhasil dibuat di:")
    print(FINAL_DATASET_DIR)


if __name__ == "__main__":
    main()