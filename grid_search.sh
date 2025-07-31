#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

gamma_values=(1.3 1.4 1.5)
delta_values=(0.05 0.1)

BASE_CMD="python -m robustness.main --dataset cifar --data ./data/cifar --adv-train 1 --arch spectral_resnet18 --out-dir ../data/logs/checkpoints/ --epochs 2 --weight-decay 1e-2 --step-lr 2 --workers 10 --constraint random_smooth --eps 0.5 --attack-lr 1.5 --loss-type margin_barrier --batch-size 64"

echo "Starting grid search..."

for gamma in "${gamma_values[@]}"; do
   for delta in "${delta_values[@]}"; do
      echo "----------------------------------------------------"
      echo "Running with gamma=${gamma} and delta=${delta}"
      echo "----------------------------------------------------"

      FULL_CMD="${BASE_CMD} --gamma ${gamma} --delta ${delta}"
      eval "${FULL_CMD}"

      echo "\n\n"
   done
done