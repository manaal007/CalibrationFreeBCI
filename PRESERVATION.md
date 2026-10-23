Original work is retained in place and in `original_snapshot_20261023.zip`.
The archive was created before rescue edits, and its Windows read-only attribute is set.
SHA-256: `B0D3BB8C40A6C2B9DF63AC43AAAD220519A19318DF7191EC5B8F24568F264043`.

Included: root scripts, historical outputs, existing package/tests/configuration.
Excluded: Git metadata, virtual environment, IDE settings, bytecode and the separate
`calibration-free-bci` directory. No historical experiment outputs have been deleted.
Read-only plus checksum provides tamper detection, not hardware-enforced immutability.
Keep a separate offline copy for archival storage. The original repository had no commits.

Before the second implementation pass, `rescue_before_final_v2.zip` also preserved
the first rescue package, tests, documentation, outputs and saved models. Its
read-only attribute is set. Subsequent v2 float32 filter-bank outputs remain in
their own run directories but are superseded by the float64/rank-corrected v3
workflow; no historical output was overwritten to make this correction.

Second snapshot SHA-256:
`33627ED96DF5ADC568CD4423C4FE311263C56A02C67EE4F8EEBB89DB6E5C163F`.

Before external evaluation, `rescue_before_external_v4.zip` preserved the v3
package, tests, configuration and portfolio documents. It is read-only, with
SHA-256 `EEC3C7A89827DBB3C59F054DA502838FB8367BF31C2BD4DA180AE7E070EB27E9`.
