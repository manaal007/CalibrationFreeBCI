"""Build current release, test installed wheel, and reproduce frozen PhysioNet outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from calibrationfreebci.audit import sha256
from calibrationfreebci.verification import load_comparisons,verify_run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=False)
    version=tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    source=root/'runs/verified_workflow_v3/alignment_source'
    clean=root/'.validation_clean_v3'
    executable=clean/'venv/Scripts/python.exe'
    wheel=root/f'dist/calibrationfreebci-{version}-py3-none-any.whl'
    load_comparisons(source)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','-e',str(root)],check=True)
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps',str(root),'--wheel-dir',str(root/'dist')],check=True)
    subprocess.run([sys.executable,'-m','pip','--python',str(executable),'install','--no-index','--no-deps','--force-reinstall',str(wheel)],check=True)
    installed=clean/'venv/Lib/site-packages/calibrationfreebci'
    for module in (root/'calibrationfreebci').glob('*.py'):
        if sha256(module)!=sha256(installed/module.name):
            raise ValueError(f'Installed module differs: {module.name}')
    subprocess.run([str(executable),'-m','pip','check'],check=True)
    command=f"import unittest,sys; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover({str(root/'tests')!r})); sys.exit(not r.wasSuccessful())"
    subprocess.run([str(executable),'-I','-c',command],cwd=clean,check=True)
    repeat=output/'clean_environment_repeat'
    subprocess.run([str(executable),'-I','-m','calibrationfreebci','external',
        '--source',str(source),'--subjects','1','2','3','4','5','--out',str(repeat),
        '--evaluation-kind','provisional','--exposure-record',str(root/'configs/physionet_exposure.provisional.json'),
        '--approved-external-evaluation'],cwd=clean,check=True)
    original=root/'results/physionet_external_v4'
    normal_repeat=root/'runs/physionet_external_v4_repeat'
    hashes={}
    for filename in ('predictions.csv','participant_metrics.csv'):
        values=[sha256(folder/filename) for folder in (original,normal_repeat,repeat)]
        if len(set(values))!=1:
            raise ValueError(f'{filename} differs across repeated environments')
        hashes[filename]=values[0]
    verification=verify_run(repeat)
    load_comparisons(source)
    if sha256(root/'original_snapshot_20261023.zip')!='b0d3bb8c40a6c2b9df63ac43aaad220519a19318df7191ec5b8f24568f264043':
        raise ValueError('Original archive changed')
    report=dict(status='passed',version=version,installed_module_hashes_match_source=True,
        tests=unittest.defaultTestLoader.discover(str(root/'tests')).countTestCases(),
        repeat_runs=3,byte_identical=hashes,metrics=verification,
        wheel_sha256=sha256(wheel),frozen_models_unchanged=True,original_archive_unchanged=True,
        evaluation_kind='provisional',cohort_independence='not established')
    (output/'release_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
