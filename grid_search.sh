#!/bin/bash

# Set a fallback for PyTorch on MPS devices if needed
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- Configuration ---
# Define the parameter values for the grid search
gamma_values=(1.3 1.4 1.5)
delta_values=(0.05 0.1)

# Base command for the training script
BASE_CMD="uv run -m robustness.main --dataset cifar --data ./data/cifar --adv-train 1 --arch spectral_resnet18 --out-dir ../data/logs/checkpoints/ --epochs 10 --weight-decay 1e-2 --step-lr 2 --workers 2 --constraint random_smooth --eps 0.5 --attack-lr 1.5 --loss-type margin_barrier --batch-size 64"

# --- Initialization ---
# File to log results of all runs
LOG_FILE="grid_search_log.csv"
# File to save the best results
RESULTS_FILE="best_results.txt"

# Initialize variables to track the best metrics
# Set lowest_loss to a very high number to ensure the first result is always lower
lowest_loss=99999999.0
highest_precision=0.0

# Variables to store the parameters for the best results
best_loss_params=""
best_prec_params=""

# Create the log file and write the header
echo "gamma,delta,final_loss,final_precision" > "${LOG_FILE}"

echo "Starting grid search..."
echo "Results will be saved to ${LOG_FILE} and ${RESULTS_FILE}"

# --- Grid Search Loop ---
for gamma in "${gamma_values[@]}"; do
  for delta in "${delta_values[@]}"; do
    echo "===================================================="
    echo "Running with gamma=${gamma} and delta=${delta}"
    echo "===================================================="

    # Construct the full command for the current run
    FULL_CMD="${BASE_CMD} --gamma ${gamma} --delta ${delta}"

    # Execute the command. The output is shown in real-time (with 'tee')
    # and also captured in the 'output' variable for parsing later.
    output=$(eval "${FULL_CMD}" 2>&1 | tee /dev/tty)

    # From the captured output, find the last line containing "Loss" and "AdvPrec1"
    last_line=$(echo "${output}" | grep "Loss" | tail -n 1)

    # Initialize current metrics to default values
    current_loss="N/A"
    current_precision="N/A"

    # Check if a valid log line was found
    if [[ -n "${last_line}" ]]; then
      # Parse the last line to extract Loss and AdvPrec1
      current_loss=$(echo "${last_line}" | awk '{print $7}')
      current_precision=$(echo "${last_line}" | awk '{print $10}')

      echo "" # Add a newline for cleaner summary output
      echo "---"
      echo "Run Summary (gamma=${gamma}, delta=${delta}):"
      echo "Final Loss: ${current_loss}, Final AdvPrec1: ${current_precision}"
      echo "---"

      # --- Update Best Loss ---
      # Use 'bc' for floating-point comparison
      if (( $(echo "${current_loss} < ${lowest_loss}" | bc -l) )); then
        lowest_loss=${current_loss}
        best_loss_params="gamma=${gamma}, delta=${delta}"
        echo ">>> New lowest loss found: ${lowest_loss}"
      fi

      # --- Update Best Precision ---
      if (( $(echo "${current_precision} > ${highest_precision}" | bc -l) )); then
        highest_precision=${current_precision}
        best_prec_params="gamma=${gamma}, delta=${delta}"
        echo ">>> New highest precision found: ${highest_precision}"
      fi
    else
      echo "Could not parse training output for this run."
    fi

    # Log the final results for the current run to the CSV file
    echo "${gamma},${delta},${current_loss},${current_precision}" >> "${LOG_FILE}"

    echo -e "\n\n"
  done
done

# --- Final Report ---
# Save the best results to the results file
{
  echo "Grid Search Summary"
  echo "================================="
  echo ""
  echo "Lowest Loss Achieved:"
  echo "  - Loss: ${lowest_loss}"
  echo "  - Parameters: ${best_loss_params}"
  echo ""
  echo "Highest Precision Achieved:"
  echo "  - Precision (AdvPrec1): ${highest_precision}"
  echo "  - Parameters: ${best_prec_params}"
  echo ""
  echo "================================="
  echo "A full log of all runs is available in ${LOG_FILE}"
} > "${RESULTS_FILE}"

echo "Grid search finished."
echo "Best results have been saved to ${RESULTS_FILE}"```