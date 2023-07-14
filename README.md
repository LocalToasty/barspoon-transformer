# Barspoon: A Transformer Architecture for Multilabel Predictions

Barspoon transformers are a transformer architecture for multilabel prediction
tasks for application in histopathological problems, but easily adaptable to
other domains.  It closely follows the transformer architecture described in
[Attention Is All You Need][1], slightly adapted to enable multi-label
prediction for many labels without loss of accuracy, even for a large number of
potentially noisy labels.  For more detailed information on the architecture,
refer to the [model's definition][2].

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: barspoon/model.py#L22  "Definition of the Barspoon Model Architecture"

## Installation

To install barspoon, run

```sh
pip install git+https://github.com/LocalToasty/barspoon-transformer
```

To properly leverage your GPU, you my need to manually install PyTorch as
described [on their website][3].

[3]: https://pytorch.org/get-started/locally "Start Locally | PyTorch"

## User Guide

In the following, we will give examples of how to use barspoon to do some
common-place prediction tasks in histopathology.  We assume our dataset to
consist of multiple _patients_, each of which has zero or more histopathological
_slides_ assigned to them.  For each patient, we have a series of _target
labels_ we want to train the network to predict.

We initially need the following:

 1. A table containing clinical information, henceforth the _clini table_.  This
    table has to be in either csv or excel format.  It has to have at least one
    column `patient`, which contains an ID identifying each patient, and other
    columns matching clinical information to that patient.
 2. Features extracted from each slide, generated using e.g. [KatherLab's
    end-to-end feature extraction pipeline][4].
 3. A table matching each patient to their slides, the _slide table_.  The slide
    table has two columns, `patient` and `filename`.  The `patient` column has
    to contain the same patient IDs found in the clini table.  The `filename`
    column contains the file paths to features belonging to that patient.  Each
    `filename` has to be unique, but one `patient` can be mapped to multiple
    `filename`s.

[4]: https://github.com/KatherLab/end2end-WSI-preprocessing
    "End-to-End WSI Processing Pipeline"

### Generating the Target File

```sh
barspoon-gen-target-file \
    --clini-table path/to/clini.csv \
    --category msi --category stage \
    --quantize leucocyte-fraction 3 \
    --output-file targets.toml
```

### Training a Model

```sh
barspoon-train \
    --output-dir path/to/save/results/to \
    --clini-table path/to/clini.csv \
    --slide-table path/to/slide.csv \
    --feature-dir dir/containing/features \
    --target-file path/to/target.toml
```
