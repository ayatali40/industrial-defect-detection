# MVTec AD dataset (manual download required)

This project does **not** ship or download the MVTec Anomaly Detection dataset.
Obtain it directly from the official source:

<https://www.mvtec.com/research-teaching/datasets/mvtec-ad>

## License

MVTec AD is (c) 2019 MVTec Software GmbH and is released under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)** license. It is intended for non-commercial use only. See
the `readme.txt` and `license.txt` shipped with the dataset.

If you use the dataset in scientific work, cite:

> Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
> "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
> IEEE Conference on Computer Vision and Pattern Recognition, 2019.

## Expected layout

```
data/raw/mvtec_ad/
├── bottle/
├── cable/
├── ...
└── transistor/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── <defect_type>/
    ├── ground_truth/
    │   └── <defect_type>/
    │       └── <image_stem>_mask.png
    ├── readme.txt
    └── license.txt
```

## How to configure

`configs/config.yaml` contains:

```yaml
paths:
  data_root: transistor
data:
  category: transistor
  supported_categories: [transistor, cable, leather, bottle, capsule, grid]
  categories: [transistor, cable, leather, bottle, capsule, grid]
  category_roots:
    bottle: ../bottle/bottle
    cable: ../cable/cable
    capsule: ../capsule/capsule
    grid: ../grid/grid
    leather: ../leather/leather
```

`data_root` may point to either:

1. an MVTec root folder containing category subfolders (for example
   `data/raw/mvtec_ad`), or
2. a single category folder such as `transistor/`.

`data.category_roots` optionally maps individual categories to explicit
folders, so categories can live in different places without moving existing
data.

The default configuration points at the `transistor/` category folder; the other
five supported categories are mapped through `category_roots`. The project scope
is **frozen at these six categories**. The other nine MVTec AD categories are
out of scope: do not add, download, train or evaluate them. An out-of-scope
category is rejected before any artifact is loaded, and no category is ever
replaced by another category's data.

## What is not stored in this repository

- Raw dataset images and masks
- Preprocessed copies of the dataset
- Trained model binaries

All of the above are excluded in `.gitignore`. Never commit the dataset or
redistribute it.
