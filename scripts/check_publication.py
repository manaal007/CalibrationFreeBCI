"""Check Git-visible files for accidentally included datasets, caches or large files."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    violations=[]
    count,total=0,0
    for name in sorted(set(filter(None,paths))):
        path=root/name
        if not path.is_file():
            continue
        count+=1
        size=path.stat().st_size
        total+=size
        parts=Path(name).parts
        if path.suffix.lower() in {'.mat','.edf','.gdf','.bdf','.fif','.fdt','.set','.joblib','.zip'}:
            violations.append(dict(path=name,reason='dataset, model or local archive'))
        if any(p.startswith(('.venv','.validation_')) or p in {'.idea','__pycache__'} for p in parts):
            violations.append(dict(path=name,reason='local environment/cache'))
        if path.name=='.env' or path.name.startswith('.env.') and path.name!='.env.example':
            violations.append(dict(path=name,reason='local environment settings'))
        if size>50*1024*1024:
            violations.append(dict(path=name,reason='file exceeds 50 MiB'))
    report=dict(status='passed' if not violations else 'failed',git_visible_files=count,
        git_visible_bytes=total,violations=violations,remote_push_performed=False,
        limitation='file-type and size checks, not a guarantee that arbitrary source text contains no secrets')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(violations))


if __name__=='__main__':
    main()
