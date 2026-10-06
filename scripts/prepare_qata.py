"""
Prepare QaTa-COV19-v2 dataset for C2Seg.

Source structure (on server):
  <qata_root>/
    Train Set/
      Images/          covid_*.png
      Ground-truths/   mask_covid_*.png
    Test Set/
      Images/
      Ground-truths/
  LViT/
    Train_Val_text_for_Covid19.xlsx   (Image + Description)
    Train_ID_for_Covid19.xlsx         (Image — train split IDs)
    Val_ID_for_Covid19.xlsx           (Image — val split IDs)
    Test_text_for_Covid19.xlsx        (Image + Description)

Output structure:
  <out_root>/
    train/  images/  masks/  texts.json
    val/    images/  masks/  texts.json
    test/   images/  masks/  texts.json

Usage:
  python scripts/prepare_qata.py \
    --qata_root /path/to/QaTa-COV19-v2 \
    --out_root  data/c2seg
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def load_text_xlsx(xlsx_path: Path) -> dict:
    """Returns {image_filename: description}, key = covid_X.png (no 'mask_' prefix)."""
    df = pd.read_excel(xlsx_path)
    mapping = {}
    for _, row in df.iterrows():
        fname = str(row["Image"]).strip()
        text  = str(row["Description"]).strip()
        text  = text.replace(" EOF XXX", "").strip()
        img_fname = fname.replace("mask_", "") if fname.startswith("mask_") else fname
        mapping[img_fname] = text
    return mapping


def load_id_xlsx(xlsx_path: Path) -> set:
    """Returns set of image filenames (covid_X.png) from an ID file."""
    df = pd.read_excel(xlsx_path)
    ids = set()
    for fname in df["Image"]:
        fname = str(fname).strip()
        img_fname = fname.replace("mask_", "") if fname.startswith("mask_") else fname
        ids.add(img_fname)
    return ids


def copy_split(image_files, src_img_dir, src_mask_dir, out_dir, text_map):
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    texts = {}
    skipped = 0
    missing_text = 0

    for img_name in sorted(image_files):
        src_img  = src_img_dir  / img_name
        src_mask = src_mask_dir / f"mask_{img_name}"

        if not src_img.exists() or not src_mask.exists():
            skipped += 1
            continue

        shutil.copy2(src_img,  out_dir / "images" / img_name)
        shutil.copy2(src_mask, out_dir / "masks"  / img_name)

        text = text_map.get(img_name, "")
        if not text:
            missing_text += 1
        texts[img_name] = text

    with open(out_dir / "texts.json", "w") as f:
        json.dump(texts, f, indent=2)

    print(f"  {len(texts)} samples | skipped={skipped} | missing_text={missing_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qata_root", required=True,
                        help="Path to QaTa-COV19-v2 folder (contains 'Train Set', 'Test Set', 'LViT')")
    parser.add_argument("--out_root", default="data/c2seg")
    args = parser.parse_args()

    qata = Path(args.qata_root)
    out  = Path(args.out_root)
    lvit = qata / "LViT"

    train_img_dir  = qata / "Train Set" / "Images"
    train_mask_dir = qata / "Train Set" / "Ground-truths"
    test_img_dir   = qata / "Test Set"  / "Images"
    test_mask_dir  = qata / "Test Set"  / "Ground-truths"

    # load text & split IDs
    print("Loading annotations...")
    trainval_text = load_text_xlsx(lvit / "Train_Val_text_for_Covid19.xlsx")
    test_text     = load_text_xlsx(lvit / "Test_text_for_Covid19.xlsx")
    train_ids     = load_id_xlsx(lvit / "Train_ID_for_Covid19.xlsx")
    val_ids       = load_id_xlsx(lvit / "Val_ID_for_Covid19.xlsx")

    print(f"  train_ids={len(train_ids)} | val_ids={len(val_ids)} | test_text={len(test_text)}")

    # all images in Train Set
    all_train_imgs = {
        f.name for f in train_img_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    }
    # any image not in train_ids or val_ids goes to train (safety fallback)
    assigned = train_ids | val_ids
    leftover = all_train_imgs - assigned
    if leftover:
        print(f"  {len(leftover)} images not in any ID file → assigned to train")
    final_train = (train_ids & all_train_imgs) | leftover
    final_val   = val_ids & all_train_imgs

    test_imgs = {
        f.name for f in test_img_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    }

    print(f"Split → train={len(final_train)} | val={len(final_val)} | test={len(test_imgs)}")

    print("Preparing train...")
    copy_split(final_train, train_img_dir, train_mask_dir, out / "train", trainval_text)

    print("Preparing val...")
    copy_split(final_val, train_img_dir, train_mask_dir, out / "val", trainval_text)

    print("Preparing test...")
    copy_split(test_imgs, test_img_dir, test_mask_dir, out / "test", test_text)

    print(f"\nDone. Dataset at: {out.resolve()}")


if __name__ == "__main__":
    main()
