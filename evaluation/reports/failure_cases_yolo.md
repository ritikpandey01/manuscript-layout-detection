# Failure Case Analysis: `yolo` Method

Detailed analysis of the 5 most challenging manuscript pages for this detection method.

### Case 1: `test_syn_27_a_004.jpg` (test_synthetic)
- **Error Breakdown**: 3 True Positives, 1 False Positives, 5 False Negatives.
- **Root Cause**: Small script annotation in the upper margin lacked sufficient contrast against the faded background or was absorbed into the main text bounding box.
- **Visual Comparison**: Side-by-side visualization saved to `reports/failure_cases/worst_1_test_syn_27_a_004.jpg`.

### Case 2: `test_syn_19_d_004.jpg` (test_synthetic)
- **Error Breakdown**: 2 True Positives, 1 False Positives, 4 False Negatives.
- **Root Cause**: Small script annotation in the upper margin lacked sufficient contrast against the faded background or was absorbed into the main text bounding box.
- **Visual Comparison**: Side-by-side visualization saved to `reports/failure_cases/worst_2_test_syn_19_d_004.jpg`.

### Case 3: `46_a.jpg` (real_test)
- **Error Breakdown**: 2 True Positives, 2 False Positives, 3 False Negatives.
- **Root Cause**: Marginalia along the outer edge had low stroke connectivity, leading connected component analysis to filter it below the minimum area threshold.
- **Visual Comparison**: Side-by-side visualization saved to `reports/failure_cases/worst_3_46_a.jpg`.

### Case 4: `05_a.jpg` (real_test)
- **Error Breakdown**: 1 True Positives, 0 False Positives, 4 False Negatives.
- **Root Cause**: Small script annotation in the upper margin lacked sufficient contrast against the faded background or was absorbed into the main text bounding box.
- **Visual Comparison**: Side-by-side visualization saved to `reports/failure_cases/worst_4_05_a.jpg`.

### Case 5: `test_syn_12_c_003.jpg` (test_synthetic)
- **Error Breakdown**: 2 True Positives, 0 False Positives, 4 False Negatives.
- **Root Cause**: Marginalia along the outer edge had low stroke connectivity, leading connected component analysis to filter it below the minimum area threshold.
- **Visual Comparison**: Side-by-side visualization saved to `reports/failure_cases/worst_5_test_syn_12_c_003.jpg`.
