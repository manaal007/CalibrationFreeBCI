"""Validate a fresh venv offline using existing pip-cached wheels.

No network requests, no cache mutation. A missing wheel is a concrete blocker,
not permission to download dependencies. Invoke with the configured interpreter.
"""
import argparse
import email
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import zipfile
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--lock',type=Path,default=Path('requirements-lock.txt'))
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    wheelhouse=out/'wheels'
    wheelhouse.mkdir()
    required={}
    for line in args.lock.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name,version=line.split('==')
            if canonicalize_name(name)!='calibrationfreebci':
                required[(canonicalize_name(name),version)]=line
    cache=Path(subprocess.check_output([sys.executable,'-m','pip','cache','dir'],text=True).strip())
    compatible={str(tag) for tag in sys_tags()}
    found={}
    for path in list(cache.rglob('*.body'))+list(cache.rglob('*.whl')):
        if not zipfile.is_zipfile(path):
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                metadata_paths=[name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
                if len(metadata_paths)!=1:
                    continue
                metadata_path=metadata_paths[0]
                metadata=email.message_from_bytes(archive.read(metadata_path))
                key=(canonicalize_name(metadata['Name']),metadata['Version'])
                if key not in required or key in found:
                    continue
                wheel_path=metadata_path.rsplit('/',1)[0]+'/WHEEL'
                tags=email.message_from_bytes(archive.read(wheel_path)).get_all('Tag',[])
                matches=sorted(set(tags)&compatible)
                if not matches:
                    continue
                stem=metadata_path.split('/')[-2].removesuffix('.dist-info')
                destination=wheelhouse/f'{stem}-{matches[0]}.whl'
                shutil.copy2(path,destination)
                found[key]=destination.name
        except (KeyError,zipfile.BadZipFile):
            continue
    missing=[required[key] for key in required if key not in found]
    report=dict(available_wheels=len(found),missing_wheels=missing,network_used=False)
    (out/'cache_inventory.json').write_text(json.dumps(report,indent=2))
    if missing:
        print(json.dumps(report,indent=2))
        raise SystemExit(2)
    shutil.copy2(args.wheel,wheelhouse/args.wheel.name)
    venv.EnvBuilder(with_pip=True).create(out/'venv')
    executable=out/'venv/Scripts/python.exe' if sys.platform=='win32' else out/'venv/bin/python'
    subprocess.run([str(executable),'-m','pip','install','--no-index','--find-links',str(wheelhouse),
                    '-r',str(args.lock.resolve()),str((wheelhouse/args.wheel.name).resolve())],check=True)
    subprocess.run([str(executable),'-m','pip','check'],check=True)
    project=Path(__file__).resolve().parents[1]
    # Run from outside the source tree; -I excludes PYTHONPATH and the cwd.
    command="import calibrationfreebci, pathlib, json; from calibrationfreebci.cli import main; p=pathlib.Path(calibrationfreebci.__file__); assert 'site-packages' in str(p); assert p.with_name('default_nested.json').is_file(); print(p)"
    subprocess.run([str(executable),'-I','-c',command],cwd=out,check=True)
    command=f"import unittest,sys; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover({str(project/'tests')!r})); sys.exit(not result.wasSuccessful())"
    subprocess.run([str(executable),'-I','-c',command],cwd=out,check=True)
    report['status']='passed_fresh_environment_install_and_tests'
    (out/'clean_setup.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
