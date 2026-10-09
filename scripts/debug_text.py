#!/usr/bin/env python3
"""Check text diversity in dataset."""

import json
from pathlib import Path
from collections import Counter
import sys

def analyze_texts(data_root: str):
    print("=" * 60)
    print("TEXT DIVERSITY ANALYSIS")
    print("=" * 60)
    
    for split in ['train', 'val']:
        text_file = Path(data_root) / split / 'texts.json'
        if not text_file.exists():
            continue
            
        with open(text_file) as f:
            texts = json.load(f)
        
        print(f"\n--- {split.upper()} ---")
        print(f"Total samples: {len(texts)}")
        
        # Count unique texts
        text_list = list(texts.values())
        unique_texts = set(text_list)
        print(f"Unique texts: {len(unique_texts)}")
        print(f"Diversity ratio: {len(unique_texts)/len(texts)*100:.1f}%")
        
        # Most common texts
        counter = Counter(text_list)
        print(f"\nTop 10 most common texts:")
        for text, count in counter.most_common(10):
            print(f"  [{count:4d}x] {text[:70]}...")
        
        # Check similarity by prefix
        prefixes = Counter([t[:50] for t in text_list])
        print(f"\nUnique 50-char prefixes: {len(prefixes)}")
        
        # Simulate batch similarity
        print(f"\n--- Simulating batch of 256 ---")
        import random
        random.seed(42)
        batch_texts = random.sample(text_list, min(256, len(text_list)))
        batch_unique = len(set(batch_texts))
        print(f"Unique texts in batch: {batch_unique}/256")
        print(f"Batch diversity: {batch_unique/256*100:.1f}%")


if __name__ == '__main__':
    data_root = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg"
    analyze_texts(data_root)
