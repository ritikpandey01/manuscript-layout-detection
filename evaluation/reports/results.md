# Manuscript Layout Region Detection — Evaluation Results

Evaluation across dataset splits using **AP@0.5** and **macro mAP@0.5**.

## Method: `rules`

| Split | macro mAP@0.5 | header AP (N) | footer AP (N) | main_text AP (N) | side_text AP (N) | filler AP (N) |
|---|---|---|---|---|---|---|
| **val** | **0.253** | 0.091 (18) | 0.071 (11) | 1.000 (50) | 0.000 (20) | 0.103 (123) |
| **test_synthetic** | **0.270** | 0.223 (17) | 0.161 (19) | 0.889 (56) | 0.000 (21) | 0.076 (150) |
| **real_test** | **0.304** | 0.000 (6) | 0.000 (0) | 1.000 (9) | 0.000 (13) | 0.218 (4) |

## Method: `yolo`

| Split | macro mAP@0.5 | header AP (N) | footer AP (N) | main_text AP (N) | side_text AP (N) | filler AP (N) |
|---|---|---|---|---|---|---|
| **val** | **0.574** | 0.629 (18) | 0.273 (11) | 1.000 (50) | 0.545 (20) | 0.423 (123) |
| **test_synthetic** | **0.741** | 0.818 (17) | 0.636 (19) | 1.000 (56) | 0.636 (21) | 0.613 (150) |
| **real_test** | **0.402** | 0.182 (6) | 0.000 (0) | 1.000 (9) | 0.091 (13) | 0.333 (4) |

## Method: `hybrid`

| Split | macro mAP@0.5 | header AP (N) | footer AP (N) | main_text AP (N) | side_text AP (N) | filler AP (N) |
|---|---|---|---|---|---|---|
| **val** | **0.992** | 0.989 (18) | 0.995 (11) | 1.000 (50) | 0.995 (20) | 0.984 (123) |
| **test_synthetic** | **0.968** | 0.986 (17) | 0.937 (19) | 0.995 (56) | 0.995 (21) | 0.925 (150) |
| **real_test** | **0.435** | 0.222 (6) | 0.000 (0) | 1.000 (9) | 0.278 (13) | 0.245 (4) |

*Note: `real_test` results are indicative with low support (9 pages total, draft boxes).*