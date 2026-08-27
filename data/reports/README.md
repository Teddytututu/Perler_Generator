# Report Directory

Reports are grouped by the task that produced them. New runs should be written
under the matching task directory instead of directly under `data/reports`.

```text
data/reports/
|-- data_quality/
|   |-- portrait_audits/             Source portrait audit reports
|   `-- outline_dataset_validation/  Standardized outline dataset validation
|-- outline_generation/
|   |-- baselines/                   Canny and mask-boundary comparisons
|   `-- model_runs/                  Learned outline model evaluations
`-- mask_generation/
    |-- pretrained_baseline/         Untuned pretrained segmentation results
    |-- threshold_sweeps/
    |   |-- pretrained/              Threshold sweeps before fine-tuning
    |   |-- fine_tuned/              Early fine-tuned threshold sweeps
    |   `-- fine_tuned_v1/           V1 threshold sweeps
    `-- final/                       Selected held-out test reports
```

Current selected mask-generation result:

```text
mask_generation/final/image-to-mask-v1-test/
```

Large checkpoints belong in `data/checkpoints`, and generated masks or visual
predictions belong in `data/inference`; neither should be placed here.
