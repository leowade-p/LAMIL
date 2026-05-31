# SC-CBBCT Dataset

## Overview

The SC-CBBCT dataset is a cone-beam breast CT (CBBCT) dataset collected for breast lesion analysis. The dataset contains CBBCT scans, benign/malignant diagnostic labels, and tumor segmentation masks.

This dataset was retrospectively collected from CBBCT scan records of 213 patients at the First Affiliated Hospital of Chongqing Medical University between October 2019 and May 2021. It includes 108 benign cases and 105 malignant cases.

The dataset is intended to support research on breast lesion classification, tumor segmentation, and computer-aided diagnosis using CBBCT imaging.

## Ethics Statement

This retrospective study was approved by the Institutional Review Board of the First Affiliated Hospital of Chongqing Medical University.

- IRB approval number: 2022-K313
- Study type: Retrospective study
- Informed consent: Waived by the ethics committee
- Data status: All data have been de-identified before release

> Note: If the final ethics approval number for data sharing is different from the study approval number, please replace the number above with the final institutional approval number.

## Data

### Images

The original CBBCT images are stored in NIfTI format. The voxel spacing of the original images is:

```text
0.273 × 0.273 × 0.273 mm³
```

The preprocessing procedure includes:

1. Cropping non-tissue background regions.
2. Zero-padding images to a uniform size of 400 × 400 pixels.
3. Converting single-channel CT slices into three-channel pseudo-color PNG images using a multi-window strategy.

The window center/width combinations are:

```text
[28, 277]
[109, 527]
[-86, 598]
```

### Labels

Each case is labeled as benign or malignant.

The benign/malignant labels are based on pathological results from surgical specimens.

### Tumor Masks

Tumor masks were manually annotated by two radiologists:

- Radiologist 1: 5 years of breast MRI diagnostic experience
- Radiologist 2: 15 years of breast MRI diagnostic experience

Disagreements during annotation were resolved through discussion with a senior radiologist:

- Radiologist 3: 20 years of breast MRI diagnostic experience

The final tumor masks were determined after consensus review.

## Dataset Split

The dataset was split at the patient level for 5-fold cross-validation.

For each fold, the approximate ratio of training, validation, and testing subsets is:

```text
3 : 1 : 1
```

Patient-level splitting was used to avoid data leakage.

## Access

The SC-CBBCT dataset is available upon reasonable request, subject to institutional approval and completion of a Data Use Agreement.

To request access, please contact:

```text
Contact person: Yineng Zheng
Email: yinengzheng@cqmu.edu.cn
Institution: The First Affiliated Hospital of Chongqing Medical University
```

Applicants should provide:

1. Research purpose
2. Institutional affiliation
3. Intended use of the dataset
4. Agreement not to attempt patient re-identification
5. Agreement not to redistribute the dataset without permission

## Licensing

The SC-CBBCT dataset is released under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

See `license.txt` for the full license text.

## Citation

If you use this dataset, please cite the corresponding paper:

```bibtex
@article{SCCBBCT,
  title   = {SC-CBBCT Dataset},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]}
}
```

## Disclaimer

The dataset is provided for research purposes only. It is not intended for clinical diagnosis, treatment planning, or commercial medical use.
