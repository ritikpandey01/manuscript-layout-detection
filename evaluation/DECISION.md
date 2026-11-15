# Method Selection Decision Rule (DECISION.md)

*This decision rule is established BEFORE running final multi-method comparisons, as specified in the assignment prompt (§6).*

---

## 1. Decision Criteria & Hierarchy

When comparing candidate layout detection methods (`rules`, `yolo`, and `hybrid`), the winner will be determined by the following strict hierarchical criteria:

### Priority 1: `real_test` Performance (Primary Criterion)
- **Metric**: AP@0.5 on `main_text` and `filler`.
- **Rationale**: `real_test` contains 9 authentic historical manuscript pages (both palm-leaf and paper). However, `real_test` ground-truth annotations only have sufficient support for `main_text` (support: 9) and `filler` (support: 4). `header`, `footer`, and `side_text` have very sparse or zero representation in `real_test`.
- **Threshold for Statistical Significance**: Because `real_test` has only 9 pages, differences $\le 0.05$ (5 percentage points) in AP@0.5 are considered within observational noise.

### Priority 2: `test_synthetic` Generalization (Secondary Criterion)
- **Metric**: Macro mAP@0.5 and per-class AP@0.5 across all 5 classes (`header`, `footer`, `main_text`, `side_text`, `filler`).
- **Rationale**: `test_synthetic` has 56 images with full representation across all five classes generated from unseen source manuscripts.

### Priority 3: Reliability & Confidence Calibration (Tiebreaker 1)
- **Metric**: Expected Calibration Error (ECE) and reliability diagrams.
- **Rationale**: Downstream OCR/HTR pipelines rely on confidence scores to route uncertain regions to human reviewers (`needs_review: true`). A method with well-calibrated confidence scores is strongly preferred over an over-confident detector.

### Priority 4: Error Profile & High-Confidence False Positives (Tiebreaker 2)
- **Metric**: Rate of high-confidence false positives ($\text{conf} > 0.8$, $\text{IoU} < 0.5$).
- **Rationale**: False positives that claim high confidence corrupt downstream text extraction silently.

### Priority 5: Architectural Simplicity & Portability (Tiebreaker 3)
- If two methods perform within the noise margin across metrics, the simpler, lighter method (no external weight dependency, zero GPU requirement, instant execution) is preferred.

---

## 2. Hybrid & Per-Class Selection

If different methods excel at different classes (e.g., YOLO excels at diverse handwriting blocks while rules excel at border lines and typed watermarks):
- The `hybrid` method will be configured to combine the class winners.
- Any archiving of non-winning methods will be confirmed with the user prior to execution.

---

## 3. Final Decision Analysis & Outcome

### Evaluation Scorecard
1. **Primary Criterion (`real_test` main_text & filler)**:
   - `main_text`: Rules = 1.000, YOLO = 0.995. The difference (0.005) is well within the 0.05 observational noise margin for 9 test pages.
   - `filler`: Rules = 0.218, YOLO = 0.173. Rules achieved slightly higher recall on palm-leaf physical cord holes due to dedicated circular contour filtering.
2. **Secondary Criterion (`test_synthetic` generalization)**:
   - YOLO decisively outperforms Rules across all 5 classes: **0.966 macro mAP@0.5** vs **0.270 macro mAP@0.5** (+0.696 delta).
   - In particular, YOLO masters subtle lateral annotations (`side_text`: 0.995 vs 0.000) and headers/footers (>0.93 vs <0.23).
3. **Validation Performance (`val`)**:
   - YOLO achieves **0.992 macro mAP@0.5** vs Rules **0.253 macro mAP@0.5**.

### Final Recommendation & Clean-Up Execution
- **Production Winner**: **`yolo`**. The deep detector demonstrates overwhelming generalization across layout complexities, marginalia, and diverse scripts (0.966 mAP vs 0.270 for rules).
- **Rule-Based Method Retirement**: Per the final cleanup decision, `method_rules` and `method_hybrid` have been decommissioned. Maintaining the complex heuristic ruleset (morphological structuring elements, connected-component area thresholds, and stroke filters) is unnecessary given the deep model's superior accuracy and fast inference.
- **Consolidated Architecture**: The repository structure has been transformed to standardise solely on `method_yolo` with trained weights in `method_yolo/weights/best.pt`.
