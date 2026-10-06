#!/usr/bin/env python3
"""
Check C2Seg data integrity.

Usage:
  python scripts/check_data.py --data_root /path/to/c2seg
  python scripts/check_data.py --data_root /path/to/c2seg --visualize --n_samples 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def check_folder_structure(root: Path) -> dict:
    """Check if folder structure is correct."""
    print("=" * 60)
    print("1. FOLDER STRUCTURE CHECK")
    print("=" * 60)
    print(f"Data root: {root.resolve()}")
    print(f"Exists: {root.exists()}")
    print()

    stats = {}
    all_ok = True

    for split in ['train', 'val', 'test']:
        split_dir = root / split
        print(f"--- {split.upper()} ---")
        stats[split] = {'images': 0, 'masks': 0, 'texts': 0, 'ok': True}

        if not split_dir.exists():
            print(f"  ❌ Folder không tồn tại: {split_dir}")
            stats[split]['ok'] = False
            all_ok = False
            continue

        # Check images
        img_dir = split_dir / 'images'
        if img_dir.exists():
            imgs = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
            stats[split]['images'] = len(imgs)
            print(f"  ✅ images/: {len(imgs)} files")
        else:
            print(f"  ❌ images/ không tồn tại")
            stats[split]['ok'] = False
            all_ok = False
            imgs = []

        # Check masks
        mask_dir = split_dir / 'masks'
        if mask_dir.exists():
            masks = list(mask_dir.glob('*.png'))
            stats[split]['masks'] = len(masks)
            print(f"  ✅ masks/: {len(masks)} files")
        else:
            print(f"  ❌ masks/ không tồn tại")
            stats[split]['ok'] = False
            all_ok = False
            masks = []

        # Check texts.json
        text_file = split_dir / 'texts.json'
        if text_file.exists():
            with open(text_file) as f:
                texts = json.load(f)
            stats[split]['texts'] = len(texts)
            print(f"  ✅ texts.json: {len(texts)} entries")

            # Check missing/empty texts
            img_names = {p.name for p in imgs}
            missing_text = img_names - set(texts.keys())
            empty_text = [k for k, v in texts.items() if not v.strip()]
            if missing_text:
                print(f"  ⚠️  {len(missing_text)} images không có text")
            if empty_text:
                print(f"  ⚠️  {len(empty_text)} entries có text rỗng")
        else:
            print(f"  ❌ texts.json không tồn tại")
            stats[split]['ok'] = False
            all_ok = False

        # Check image-mask pairing
        if imgs and masks:
            img_stems = {p.stem for p in imgs}
            mask_stems = {p.stem for p in masks}
            missing_masks = img_stems - mask_stems
            if missing_masks:
                print(f"  ⚠️  {len(missing_masks)} images thiếu mask")
                all_ok = False
            else:
                print(f"  ✅ Tất cả images đều có mask tương ứng")
        print()

    return stats, all_ok


def check_mask_values(root: Path, n_check: int = 100) -> bool:
    """Check if masks are binary (0, 255)."""
    print("=" * 60)
    print("2. MASK VALUES CHECK")
    print("=" * 60)

    all_ok = True
    for split in ['train', 'val', 'test']:
        mask_dir = root / split / 'masks'
        if not mask_dir.exists():
            continue

        masks = list(mask_dir.glob('*.png'))[:n_check]
        print(f"Checking {len(masks)} masks in {split}...")

        all_values = set()
        issues = []

        for mask_path in masks:
            mask = np.array(Image.open(mask_path).convert('L'))
            unique_vals = np.unique(mask)
            all_values.update(unique_vals)

            if not set(unique_vals).issubset({0, 255}):
                issues.append((mask_path.name, unique_vals))

        print(f"  Unique values: {sorted(all_values)}")

        if not issues:
            print(f"  ✅ Tất cả masks đều binary (0, 255)")
        else:
            print(f"  ⚠️ {len(issues)} masks có values khác:")
            for name, vals in issues[:3]:
                print(f"     {name}: {list(vals)}")
            all_ok = False
        print()

    return all_ok


def check_image_sizes(root: Path, n_check: int = 200) -> None:
    """Check image sizes distribution."""
    print("=" * 60)
    print("3. IMAGE SIZES")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        img_dir = root / split / 'images'
        if not img_dir.exists():
            continue

        imgs = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        imgs = imgs[:n_check]

        sizes = {}
        for img_path in imgs:
            img = Image.open(img_path)
            size = img.size  # (W, H)
            sizes[size] = sizes.get(size, 0) + 1

        print(f"{split}:")
        for size, count in sorted(sizes.items(), key=lambda x: -x[1])[:5]:
            print(f"  {size[0]}x{size[1]}: {count} images")
    print()


def print_statistics(stats: dict) -> None:
    """Print data statistics and compare with paper."""
    print("=" * 60)
    print("4. DATA STATISTICS")
    print("=" * 60)

    total = 0
    print(f"{'Split':<10} {'Images':>8} {'Masks':>8} {'Texts':>8}")
    print("-" * 40)
    for split in ['train', 'val', 'test']:
        if split in stats:
            s = stats[split]
            print(f"{split:<10} {s['images']:>8} {s['masks']:>8} {s['texts']:>8}")
            total += s['images']
    print("-" * 40)
    print(f"{'Total':<10} {total:>8}")
    print()
    print("Paper QaTa-COV19 expected:")
    print("  Train: ~7,145")
    print("  Val:   ~1,016")
    print("  Test:  ~2,113")
    print("  Total: ~10,274")
    print()


def show_sample_texts(root: Path, n: int = 5) -> None:
    """Show sample text descriptions."""
    print("=" * 60)
    print("5. SAMPLE TEXT DESCRIPTIONS")
    print("=" * 60)

    for split in ['train']:
        text_file = root / split / 'texts.json'
        if not text_file.exists():
            continue

        with open(text_file) as f:
            texts = json.load(f)

        print(f"Sample from {split}:")
        for i, (fname, text) in enumerate(list(texts.items())[:n]):
            print(f"\n[{fname}]")
            print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")
    print()


def visualize_samples(root: Path, split: str = 'train', n_samples: int = 4) -> None:
    """Visualize samples with matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed. Skipping visualization.")
        return

    print("=" * 60)
    print("6. VISUALIZATION")
    print("=" * 60)

    split_dir = root / split
    img_dir = split_dir / 'images'
    mask_dir = split_dir / 'masks'

    with open(split_dir / 'texts.json') as f:
        texts = json.load(f)

    imgs = sorted(img_dir.glob('*.png'))[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, img_path in enumerate(imgs):
        img = Image.open(img_path).convert('RGB')
        mask_path = mask_dir / (img_path.stem + '.png')
        mask = Image.open(mask_path).convert('L')
        text = texts.get(img_path.name, "[No text]")

        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Image: {img_path.name}")
        axes[i][0].axis('off')

        axes[i][1].imshow(mask, cmap='gray')
        axes[i][1].set_title(f"Mask")
        axes[i][1].axis('off')

        axes[i][2].imshow(img)
        axes[i][2].imshow(np.array(mask), alpha=0.5, cmap='Reds')
        title = f"Text: {text[:60]}..." if len(text) > 60 else f"Text: {text}"
        axes[i][2].set_title(title)
        axes[i][2].axis('off')

    plt.tight_layout()
    save_path = root / 'data_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.close()


def test_dataset_class(root: Path) -> bool:
    """Test loading data with C2SegDataset class."""
    print("=" * 60)
    print("7. DATASET CLASS TEST")
    print("=" * 60)

    try:
        # Add parent to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from datasets.c2seg_dataset import C2SegDataset

        train_ds = C2SegDataset(
            root_dir=str(root / 'train'),
            image_size=224,
            augment=False
        )

        print(f"Dataset loaded: {len(train_ds)} samples")

        # Test get item
        image, mask, text = train_ds[0]
        print(f"Sample 0:")
        print(f"  Image shape: {image.shape}")  # [3, 224, 224]
        print(f"  Mask shape:  {mask.shape}")   # [1, 224, 224]
        print(f"  Mask range:  [{mask.min():.2f}, {mask.max():.2f}]")
        print(f"  Text:        {text[:80]}...")
        print()

        # Validate shapes
        assert image.shape == (3, 224, 224), f"Bad image shape: {image.shape}"
        assert mask.shape == (1, 224, 224), f"Bad mask shape: {mask.shape}"
        assert 0 <= mask.min() <= mask.max() <= 1, f"Bad mask range"

        print("✅ Dataset class works correctly!")
        return True

    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check C2Seg data integrity")
    parser.add_argument('--data_root', required=True, help='Path to c2seg data folder')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples to visualize')
    parser.add_argument('--skip_dataset_test', action='store_true', help='Skip dataset class test')
    args = parser.parse_args()

    root = Path(args.data_root)

    if not root.exists():
        print(f"❌ Data root không tồn tại: {root}")
        sys.exit(1)

    print()
    print("🔍 C2SEG DATA CHECK")
    print("=" * 60)
    print()

    # Run checks
    stats, structure_ok = check_folder_structure(root)
    mask_ok = check_mask_values(root)
    check_image_sizes(root)
    print_statistics(stats)
    show_sample_texts(root)

    if args.visualize:
        visualize_samples(root, n_samples=args.n_samples)

    if not args.skip_dataset_test:
        dataset_ok = test_dataset_class(root)
    else:
        dataset_ok = True

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = structure_ok and mask_ok and dataset_ok
    if all_ok:
        print("✅ Data check PASSED! Ready for training.")
    else:
        print("⚠️ Data check có issues, xem chi tiết ở trên.")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
