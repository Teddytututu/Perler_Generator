# 987 qualified Qwen teacher images

987 native PNG teacher targets selected from the 2026-10-11 review of all1250 source identities. This revision includes the77 accepted results of the340 targeted reruns and preserves selected earlier versions, including the existing human choices. The263 redo items remain outside this qualified dataset.

Split: **904 train /41 validation /42 test**. Use only train for training and retain the other splits for independent evaluation. Source identity, original-source SHA256, teacher SHA256, license, attribution, generation version/config hash and review reasons are recorded per item.

## Download

[Verified dataset Release](https://github.com/Teddytututu/Perler_Generator/releases/tag/qwen-teachers-987-20261011)

Download `Qwen-Chen-qualified-teachers-987-20261011.zip` and extract its contents into this directory. This produces `images/<round_id>.png`, together with the included metadata. Native image bytes and pixel dimensions are preserved. The archive's exact byte count and SHA256 are in `package-integrity.json`. Images are provided as a Release asset so the repository keeps a small metadata change.

`manifest.json` and `teachers.jsonl` use paths relative to the extracted dataset root. `source_archive_member` identifies the original image inside CHEN Jing's existing dataset.zip; its SHA256 is recorded per row. Original source images are not duplicated in this teacher-target archive.

## Files and licensing

- `manifest.json`:987 qualified rows and split counts.
- `teachers.jsonl` / `teachers.csv`: machine-readable image/metadata indexes.
- `splits/`: fixed train, validation and test indexes.
- `package-integrity.json`: PNG SHA/full decode, ZIP CRC and inventory verification.
- `annotation-integrity.json` / `supervision-integrity.json`: provenance from the complete1250-image review and independent checks.

Individual image licenses and attribution apply as recorded in the manifest. Review status is a source-to-output visual classification; artifact fields retain observations about faint color-line artifacts where recorded, for downstream evaluation.
