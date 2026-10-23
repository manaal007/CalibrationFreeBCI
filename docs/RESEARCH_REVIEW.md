# Publication-stage research review

## What the existing evidence supports

The nine-person source LOSO comparison favors aligned CSP descriptively (0.7334
balanced accuracy versus 0.6964). It remains development evidence: the alignment
method was introduced during iterative work, so participant-separated folds do
not remove all method-development selection bias. The separate nested
filter-bank experiment was negative (0.6817 versus 0.7029 on identical outer
folds) and remains reported.

The provisional five-person PhysioNet comparison is heterogeneous. Aligned CSP
balanced accuracy ranges from 0.3512 to 0.8893. Its mean is 0.6044, with a 95%
participant-bootstrap interval of 0.4332–0.7658. The paired improvement over
RMS/CSP is 0.0430, with interval −0.0750 to 0.1634. These data do not establish
above-chance population performance or a reliable alignment advantage. A
five-person percentile bootstrap is itself imprecise; it is descriptive evidence,
not a substitute for a larger independent sample. Trial-level bootstrap would
overstate the effective sample size and is not used.

Cho2017 is historically exposed, including its single-person 0.695 result.
PhysioNet S001–S005 are now exposed, regardless of their earlier unknown history.
Neither cohort may be reused as fresh confirmation after further tuning. Low
performers were retained; labels, thresholds and prediction prevalence were not
modified in response to outcomes.

## Two distinct deployment claims

- **Strict inductive, source-only inference:** RMS/CSP fits its learned parameters
  on source participants. Per-trial RMS is a deterministic transformation and
  does not estimate a target-participant reference from other trials.
- **No supervised calibration, with unsupervised adaptation:** aligned CSP also
  estimates a whitening reference using a batch of unlabeled target-participant
  trials. It is transductive; changing the target batch can change predictions.
  It should not be advertised as requiring no target-side estimation or as an
  online single-trial decoder.

Remaining biases include small source and external cohorts, iterative method
development, fixed cohort availability, non-identical recording systems/filter
boundaries, retained artifact flags, and alignment dependence on batch composition
and class mixture. No patient, online-control or clinical efficacy claim is tested.

## Highest-value next experiment — proposed, not approved or run

Use a larger, prospectively frozen PhysioNet cohort for the **existing two frozen
models**, without another hyperparameter search. The dataset describes 109
volunteers and compatible imagined left/right-fist runs 4, 8 and 12. Participants
beyond S005 are candidates only; they are not automatically untouched.

1. Obtain an owner-reviewed exposure ledger for each candidate participant,
   including earlier notebooks, downloaded files and any analyses outside this
   repository. Only S001–S005 were found in the current local EEGMMIDB cache;
   absence from that cache does not prove absence of prior use.
2. Freeze the eligible participant IDs, source model checksums, zero thresholds,
   task runs, preprocessing, artifact policy and integrity-only exclusion rules
   before opening any new decoding outcomes. Missing channels, corrupt files or
   task incompatibility may justify documented exclusions; low accuracy may not.
   The current loader requires native 160 Hz. Inspect sampling-rate metadata
   before finalizing eligibility: the installed MOABB documentation identifies
   S088, S092 and S100 as 128-Hz recordings. Prespecify either a validated
   metadata-driven resampling extension or a sampling-rate eligibility rule;
   do not silently drop unsupported recordings after seeing their accuracy.
3. Prefer a substantially larger eligible cohort, for example at least 30
   participants if exposure history and data quality permit. This is a practical
   precision target, not a powered guarantee. Choose the final sample size using
   a prespecified precision/power calculation that respects participant-level
   variance and the uncertainty of the current five-person pilot.
4. Primary endpoint: participant-mean balanced accuracy of the prespecified
   selected model. Primary comparison: paired participant balanced-accuracy
   difference versus frozen RMS/CSP. Report participant intervals, class recalls,
   AUC, prediction prevalence and all participant outcomes. Define the inferential
   procedure and multiplicity policy in advance; do not select whichever analysis
   gives the most favorable conclusion.
5. Record unlabeled batch construction for alignment explicitly. Do not adapt
   batch sizes, artifact cutoffs, thresholds or channel selections to new labels.
6. Obtain explicit approval for dataset download size and the final evaluation.
   If prior exposure cannot be established, use provisional terminology or select
   a different compatible dataset before evaluation.

A larger cohort can narrow uncertainty and reveal whether alignment benefits
are consistent. It cannot by itself repair exposure history or selection bias.
No additional dataset was downloaded and no new experiment was run for this review.

Source: [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
Sampling-rate caveat: [MOABB PhysionetMI documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html).
