#!/bin/bash
# Ablation study script for C2Seg
# Tests different combinations of BCAM and K-Gate

echo "=============================================="
echo "C2Seg Ablation Study"
echo "=============================================="

# Configuration
CONFIG="configs/c2seg.yaml"
EPOCHS=20  # Short runs for ablation

# Create results directory
mkdir -p logs/ablation

echo ""
echo "[2/4] Testing: BCAM only (no K-Gate)"
echo "----------------------------------------------"
python train_c2seg.py --stage train --config $CONFIG --pretrained none --no-kgate 2>&1 | tee logs/ablation/bcam_only.log

echo ""
echo "[3/4] Testing: K-Gate only (no BCAM) - will fallback to visual"
echo "----------------------------------------------"
python train_c2seg.py --stage train --config $CONFIG --pretrained none --no-bcam 2>&1 | tee logs/ablation/kgate_only.log

echo ""
echo "[4/4] Testing: Full model (BCAM + K-Gate)"
echo "----------------------------------------------"
python train_c2seg.py --stage train --config $CONFIG --pretrained none 2>&1 | tee logs/ablation/full_model.log

echo ""
echo "=============================================="
echo "Ablation Study Complete!"
echo "=============================================="
echo "Results saved to logs/ablation/"
echo ""
echo "Summary:"
grep -h "Epoch \[${EPOCHS}/" logs/ablation/*.log | while read line; do
    echo "$line"
done
