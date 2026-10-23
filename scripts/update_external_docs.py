"""Bring current portfolio claims in line with the executed provisional evaluation."""
from pathlib import Path

root=Path(__file__).resolve().parents[1]
readme=root/'README.md'
text=readme.read_text(encoding='utf-8-sig')
start=text.index('**Current evidence:**')
end=text.index('\n\nThe original experiments',start)
text=text[:start]+'''**Current external evidence:** In a frozen, provisional BNCI-to-PhysioNet evaluation on five participants, aligned CSP achieved **60.4% participant-mean balanced accuracy** (95% participant-bootstrap interval **43.3–76.6%**) versus **56.1%** for RMS/CSP. The paired improvement interval includes zero; reliable population-level improvement is **not established**. Prior participant exposure was not confirmed, so these results are not called confirmatory.

Source LOSO balanced accuracy was 73.3% for aligned CSP versus 69.6% for RMS/CSP. A previously examined Cho participant yielded exploratory balanced accuracy 69.5%. Alignment uses an **unlabeled participant batch**, not strict single-trial inference. See the [complete external report](results/physionet_external_v4/REPORT.md), including below-chance participants. No thresholds or models were tuned on PhysioNet outcomes.'''+text[end:]
start=text.index('## Independent confirmation remains gated')
end=text.index('## Outputs, verification and portfolio',start)
text=text[:start]+'''## Frozen PhysioNet evaluation

The owner authorized evaluation, and the frozen selected model plus its matched RMS/CSP baseline were run on all five cached participants (225 trials). Metric verification and repeated inference passed. The run is explicitly **provisional** because prior participant exposure remains unconfirmed. The confirmatory gate still requires owner confirmation of previously unused participants; evaluation approval does not establish that history.

To reproduce the authorized provisional run with newly created output paths:

```powershell
.\\.venv\\Scripts\\python.exe -m calibrationfreebci external --source runs/my_complete_run/alignment_source --subjects 1 2 3 4 5 --out runs/physionet_repeat --evaluation-kind provisional --exposure-record configs/physionet_exposure.provisional.json --approved-external-evaluation
.\\.venv\\Scripts\\python.exe -m calibrationfreebci verify --run runs/physionet_repeat
```

This produces predictions, participant metrics, confidence intervals, paired differences, PNG/SVG figures and a Markdown report automatically. To regenerate derived tables and figures, use `external-report --run runs/physionet_repeat`. Model weights and thresholds remain frozen. Below-chance participants are retained, without label flipping.

Confirmatory mode remains the default and requires an owner-confirmed exposure record based on `configs/physionet_exposure.template.json`. The template itself is not approval. S001–S005 have now been evaluated, so do not tune on their outcomes and subsequently call a rerun fresh confirmation. Further method development requires a separately held-out cohort for a new confirmation.

Five participants provide wide uncertainty. The existing PhysioNet results do not establish robust population generalization. See [publication instructions](docs/PUBLISHING.md) before pushing the repository.

'''+text[end:]
text=text.replace('fresh installed-wheel environment','fresh installed-wheel environment')
readme.write_text(text,encoding='utf-8')

manuscript=root/'docs/MANUSCRIPT.md'
text=manuscript.read_text(encoding='utf-8-sig')
text=text.replace('while independent confirmation remains pending approval and cohort-exposure verification.',
    'and a subsequently authorized frozen PhysioNet evaluation remained provisional because cohort exposure was unconfirmed. Across five PhysioNet participants, aligned CSP achieved mean balanced accuracy 0.604 (95% participant interval 0.433–0.766) versus 0.561 for RMS/CSP; the paired difference interval included zero. Reliable cross-dataset population improvement is not established.')
text=text.replace('PhysioNet EEGMMIDB is reserved for possible independent evaluation.',
    'PhysioNet EEGMMIDB was subsequently evaluated with frozen models after owner authorization, with provisional status because prior cohort exposure was not established.')
text=text.replace('without model predictions or decoding metrics. These checks cannot establish that the owner has never used the participants previously.',
    'before model predictions. The later evaluation included 45 trials from each participant, 225 in total. Prior use remains unknown; the absence of historical script references was not treated as proof of independence.')
text=text.replace('Eighteen automated tests','Twenty automated tests')
position=text.index('## Discussion')
section='''### Provisional frozen PhysioNet evaluation

The owner authorized the external evaluation after all source model selection was complete. A protocol with model hashes, primary endpoint, participant list and fixed thresholds was saved before target inference. The primary selected model was aligned CSP; its comparator was the previously frozen RMS/CSP model. No weights, thresholds, participants or labels were changed based on the outcomes.

| Method | Mean balanced accuracy (95% participant interval) | Mean ROC-AUC (95% interval) | Left recall | Right recall |
|---|---:|---:|---:|---:|
| Aligned CSP | 0.6044 (0.4332–0.7658) | 0.6311 (0.4325–0.8296) | 0.5528 | 0.6561 |
| RMS/CSP | 0.5614 (0.4940–0.6289) | 0.6050 (0.4534–0.7597) | 0.5834 | 0.5394 |

The paired balanced-accuracy difference was 0.0430 (95% participant-bootstrap interval −0.0750 to 0.1634). This interval includes zero, and the selected model's accuracy interval includes chance. Aligned CSP participant balanced accuracies were 0.6690, 0.8893, 0.4219, 0.6907 and 0.3512 for S001–S005. Below-chance outcomes were retained. Both methods produced both-class predictions for each participant, without forcing a prediction balance. These results demonstrate executable transfer with substantial heterogeneity, not established reliable cross-dataset performance.

The cohort is now exposed. Its prior use was unconfirmed when predictions were opened, so the results remain provisional rather than confirmatory. Repeated predictions, metrics and source model checksums are independently checked; the complete participant table is in the external report.

![Frozen PhysioNet participant outcomes](../results/physionet_external_v4/participant_comparison.png)

'''
text=text[:position]+section+text[position:]
text=text.replace('Final confirmation requires owner approval and a verified previously unused compatible cohort. The proposed five cached PhysioNet participants remain unscored. Their small sample would still provide limited precision. After final outcomes are opened, method changes would require a fresh independent cohort.',
    'The owner approved evaluation of the five cached PhysioNet participants, but their prior exposure was not confirmed. The completed evaluation is provisional and its wide intervals do not support reliable population improvement. These participants are now exposed; further method changes require a fresh independent cohort for confirmation.')
manuscript.write_text(text,encoding='utf-8')

(root/'docs/ABSTRACT.md').write_text('''We developed and verified a CPU-based motor imagery EEG transfer pipeline with corrected cue alignment, participant-separated source validation and frozen external inference. Across nine BNCI2014-001 source participants, RMS/CSP achieved mean LOSO balanced accuracy 0.696; a fixed label-free participant-alignment variant achieved 0.733. In an authorized provisional PhysioNet evaluation on five participants (225 trials), aligned CSP achieved participant-mean balanced accuracy 0.604 (95% participant-bootstrap interval 0.433–0.766), compared with 0.561 for RMS/CSP. The paired improvement interval included zero, and two aligned participant results were below chance. An earlier exploratory Cho participant result was 0.695. Alignment uses unlabeled target batches rather than strict inductive inference. Frozen model reloads, metric recomputation, repeatability checks and 20 automated tests support technical reproducibility. Prior PhysioNet exposure remains unconfirmed; reliable independent population generalization is not established.\n''',encoding='utf-8')

(root/'docs/APPLICATION_DESCRIPTION.md').write_text('''# Graduate application project description

**CalibrationFreeBCI — reproducible EEG decoding and honest transfer evaluation**

Built a tested Python pipeline for motor imagery EEG decoding across participants and recording datasets without supervised target calibration. Verified cue timing against raw sample coordinates and the original competition protocol, corrected numerical rank handling, and compared spatial-filter, covariance and filter-bank methods using participant-separated validation. A fixed label-free participant-alignment method achieved 73.3% source LOSO balanced accuracy across nine participants. Conducted a frozen, provisional five-participant PhysioNet evaluation: aligned CSP averaged 60.4% balanced accuracy versus 56.1% for RMS/CSP, with wide intervals and no established population improvement. Preserved below-chance outcomes and separated exploratory evidence from confirmatory claims. Delivered automated tests, model provenance, reproducible predictions, scientific figures, a manuscript and publication checks.

Short CV version: Developed and tested an EEG transfer research pipeline; verified event alignment, implemented participant-separated model selection and label-free alignment, and executed a frozen external comparison with participant-level uncertainty and transparent negative findings.
''',encoding='utf-8')

print('Updated README, manuscript, abstract and application description')
