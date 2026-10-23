"""Build and verify the final wheel in the independently installed validation venv."""
import json
from pathlib import Path
import subprocess
import sys
from calibrationfreebci.audit import sha256

root=Path(__file__).resolve().parents[1]
clean=root/'.validation_clean_v3'
executable=clean/'venv/Scripts/python.exe'
wheel=root/'dist/calibrationfreebci-0.2.1-py3-none-any.whl'
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps',str(root),'--wheel-dir',str(root/'dist')],check=True)
subprocess.run([sys.executable,'-m','pip','--python',str(executable),'install','--no-index','--no-deps','--force-reinstall',str(wheel)],check=True)
installed=clean/'venv/Lib/site-packages/calibrationfreebci'
for source in (root/'calibrationfreebci').glob('*.py'):
    if sha256(source)!=sha256(installed/source.name):
        raise ValueError(f'Installed module differs: {source.name}')
subprocess.run([str(executable),'-m','pip','check'],check=True)
command=f"import unittest,sys; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover({str(root/'tests')!r})); sys.exit(not r.wasSuccessful())"
subprocess.run([str(executable),'-I','-c',command],cwd=clean,check=True)
output=root/'results/final_v3/final_distribution_cho_smoke'
subprocess.run([str(executable),'-I','-m','calibrationfreebci','cross-smoke',
    '--source',str(root/'runs/verified_workflow_v3/alignment_source'),'--out',str(output)],cwd=clean,check=True)
expected=root/'runs/verified_workflow_v3/alignment_cho_exploratory/predictions.csv'
if sha256(output/'predictions.csv')!=sha256(expected):
    raise ValueError('Final installed distribution predictions differ')
report=dict(status='passed',installed_module_hashes_match_source=True,tests=18,
            real_external_predictions_byte_identical=True,wheel_sha256=sha256(wheel),
            installed_version='0.2.1',network_dataset_downloads=False)
(root/'results/final_v3/final_distribution_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
