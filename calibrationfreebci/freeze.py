"""Freeze prespecified comparison models without touching external data."""
import json
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from .data import load_bnci
from .models import build
from .audit import sha256


def freeze(source, data_root=None):
    source = Path(source)
    protocol = json.loads((source/'protocol.json').read_text())
    selection = json.loads((source/'selection.json').read_text())
    names = list(dict.fromkeys(['csp_lda','oas_mdm',selection['model']]))
    x, frame, provenance = load_bnci(protocol['subjects'],data_root or protocol.get('data_root'))
    original = json.loads((source/'data_provenance.json').read_text())
    keys = lambda rows: {(r['subject'],r['session']):r['sha256'] for r in rows}
    if keys(original) != keys(provenance):
        raise ValueError('Source data differ from model selection run')
    manifest = {}
    with threadpool_limits(limits=2):
        for name in names:
            path = source/f'comparison_{name}.joblib'
            if path.exists():
                raise FileExistsError(path)
            model = build(name).fit(x, frame.label.to_numpy())
            joblib.dump(model, path)
            manifest[name] = dict(file=path.name, sha256=sha256(path), threshold=0)
    (source/'comparison_manifest.json').write_text(json.dumps(manifest, indent=2))
