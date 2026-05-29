# Faithfulness Evaluation: Our Approach vs XAI_tools_auto

## Overview

| | XAI_tools_auto | Our approach |
|---|---|---|
| **Purpose** | Evaluate faithfulness of one XAI method globally across a dataset | Evaluate faithfulness of cluster-specific saliency patterns for a subgroup |

---

## Detailed Differences

### 1. Subjects used for saliency ranking
- **XAI_tools_auto**: all test subjects (both classes, correct and incorrect predictions)
- **Ours**: correctly predicted positive subjects within each cluster only

### 2. Saliency representation for ranking
- **XAI_tools_auto**: raw gradient values
- **Ours**: PSD (power spectral density) of gradient maps — frequency-transformed before ranking

### 3. Feature space
- **XAI_tools_auto**: three separate experiments:
  - Spatial — channels (n_channels features)
  - Temporal — time windows (n_time_bins features)
  - Frequency — frequency bins (20 log-spaced bins)
- **Ours**: one joint experiment — (channel × frequency band) pairs = n_channels × n_bands = 85 features

### 4. EEG that gets masked
- **XAI_tools_auto**: full test set (all subjects)
- **Ours**: cluster subjects only

### 5. Baseline
- **XAI_tools_auto**: trial-level accuracy on the full test set → varies per fold (e.g. 0.73)
- **Ours**: subject-level majority vote accuracy on cluster subjects → always 1.0 by construction (subjects are pre-selected as correctly predicted)

### 6. Accuracy metric
- **XAI_tools_auto**: trial-level accuracy (correct trials / total trials)
- **Ours**: subject-level majority vote (correct subjects / total subjects)

### 7. Masking replacement method
- **XAI_tools_auto**: two variants:
  - **AR** — PGD adversarial examples (model-targeted perturbation)
  - **ROAD** — noisy imputer using interpolation from neighbours (model-agnostic, more natural replacement)
- **Ours**: PGD adversarial only (no ROAD variant)

### 8. Masking steps
- **XAI_tools_auto**: k = 1 to k = max_features, masking one feature at a time
- **Ours**: 5% to 100% of total features in 5% increments (20 fixed steps)

### 9. MoRF/LeRF curve aggregation (per-fold handling)
- **XAI_tools_auto**: computes one MoRF and one LeRF curve per fold, then averages curves across all 5 folds
- **Ours**: computes curves per fold (only for cluster subjects held out in that fold), then takes a weighted average across folds proportional to number of subjects per fold

### 10. Metrics (AOC / AUC / ABC)
Both use the same formulas — clip curves to [chance, baseline], then:
- AOC = mean(1 − clipped MoRF)
- AUC = mean(clipped LeRF)
- ABC = mean(max(0, LeRF − MoRF))

Key difference: XAI_tools_auto baseline varies (real model accuracy), ours is always 1.0.

### 11. Spearman consistency
- **XAI_tools_auto**: ranks 11 methods (6 gradient methods × signed/abs variants + random baseline)
- **Ours**: ranks 6 gradient methods only

---

## Why baseline is always 1.0 in our approach

Subjects are included in clustering only if the model correctly predicted them
(`pred_label == target_class and true_label == target_class`). When the unmasked
baseline is then computed on those same subjects with the same model, it is
guaranteed to be 1.0. This is not a bug — it is a consequence of the cluster
design. The faithfulness metrics (AOC/AUC/ABC) remain meaningful: they measure
how much accuracy degrades toward chance when cluster-specific features are masked.

---

## Why temporal masking is not applicable in our approach

The clustering is done entirely in PSD space — PSD averages power over time,
discarding all temporal information. The cluster centroid has shape
`(n_channels, n_bands)` with no time axis. Without temporal resolution in the
centroid, there is no basis for ranking time windows, so temporal masking is
not applicable. Temporal masking would require clustering on raw gradients
instead of PSD.
