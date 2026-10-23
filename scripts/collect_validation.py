"""Collect verified final artifacts without overwriting prior experiments."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from calibrationfreebci.audit import sha256
from calibrationfreebci.verification import verify_run
from calibrationfreebci.workflow import generate_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workflow',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True,exist_ok=False)
    checks={}
    for name in ('source','alignment_source','cho_exploratory','alignment_cho_exploratory'):
        checks[name]=verify_run(args.workflow/name)
    before=pd.read_csv(root/'runs/alignment_v2/predictions.csv')
    after=pd.read_csv(args.workflow/'alignment_source/predictions.csv')
    keys=['model','subject','session','run','trial','label']
    before=before.sort_values(keys).reset_index(drop=True)
    after=after.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(before[keys+['prediction']],after[keys+['prediction']])
    np.testing.assert_allclose(before.score_right,after.score_right,atol=1e-12,rtol=1e-12)
    checks['source_alignment_repeat']=dict(status='passed',rows=len(after),
        max_score_difference=float(np.max(np.abs(before.score_right-after.score_right))))
    targets=[root/'runs/cho_aligned_exploratory_v2/predictions.csv',
             args.workflow/'alignment_cho_exploratory/predictions.csv',
             root/'results/final_v3/clean_environment_cho_smoke/predictions.csv']
    hashes=[sha256(path) for path in targets]
    if len(set(hashes))!=1:
        raise ValueError('External predictions differ between repeated workflow and clean environment')
    checks['external_repeat_and_clean_environment']=dict(status='passed_byte_identical',sha256=hashes[0],runs=len(hashes))
    expected='b0d3bb8c40a6c2b9df63ac43aaad220519a19318df7191ec5b8f24568f264043'
    if sha256(root/'original_snapshot_20261023.zip')!=expected:
        raise ValueError('Original archive changed')
    checks['original_archive']='unchanged'
    for report_name,source,external in [('nested_report','source','cho_exploratory'),
                                         ('alignment_report','alignment_source','alignment_cho_exploratory')]:
        generate_report(args.workflow/source,args.workflow/external,args.out/report_name)
    evidence=args.out/'verification'
    evidence.mkdir()
    for path in [root/'results/final_v2/timing_evidence.json',root/'results/final_v2/physionet_loading_qc.json',
                 root/'.validation_clean_v3/clean_setup.json',root/'rescue_tests_final_v3.log',
                 root/'rescue_clean_setup_v3.log',root/'rescue_clean_real_smoke_v3.log']:
        shutil.copy2(path,evidence/path.name)
    (args.out/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
