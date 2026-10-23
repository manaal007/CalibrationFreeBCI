"""Re-establish sample coordinates using raw MAT fields and installed MOABB conversion."""
import inspect
import json
from pathlib import Path
import mne
import numpy as np
from scipy.io import loadmat
from moabb.datasets.bnci.base import _convert_run
from moabb.datasets import BNCI2014_001
from .data import CHANNELS,find_bnci,epoch_raw
from .audit import sha256


def verify_timing(out,root=None):
    out = Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    checks = []
    for name,path in sorted(find_bnci(root).items()):
        runs = np.atleast_1d(loadmat(path,squeeze_me=True,struct_as_record=False)['data'])
        for run_index,run in enumerate(runs):
            if not np.asarray(run.trial).size:
                continue
            converted,_ = _convert_run(run,CHANNELS+['EOG1','EOG2','EOG3'],['eeg']*22+['eog']*3,verbose=False)
            events = mne.find_events(converted,stim_channel='STI',shortest_event=1,verbose=False)
            np.testing.assert_array_equal(events[:,0],np.asarray(run.trial)-1)
            np.testing.assert_array_equal(events[:,2],run.y)
            # First run per file: compare actual epoch values against direct raw slices.
            binary = events[np.isin(events[:,2],[1,2])]
            if not any(row['file']==name for row in checks):
                epochs = epoch_raw(converted,binary)
                filtered = converted.copy().pick(CHANNELS).reorder_channels(CHANNELS)
                filtered.set_eeg_reference('average',projection=False,verbose=False)
                filtered.filter(8.,30.,verbose=False)
                for index in (0,len(epochs)-1):
                    start = int(epochs.events[index,0])+625
                    np.testing.assert_allclose(epochs.get_data()[index],filtered.get_data(start=start,stop=start+501),atol=1e-15)
            first = int(events[0,0])
            checks.append(dict(file=name,run=run_index,n_events=len(events),
                first_matlab_trial=int(np.asarray(run.trial)[0]),first_zero_based_trial=first,
                first_cue_sample=first+500,first_epoch_start=first+625,first_epoch_last=first+1125))
    if not checks:
        raise FileNotFoundError('No cached BNCI event data')
    interval = BNCI2014_001().interval
    np.testing.assert_array_equal(interval,[2,6])
    report = dict(status='passed',checks=checks,dataset_interval=interval,
                  installed_converter_source=inspect.getsourcefile(inspect.unwrap(_convert_run)),
                  converter_sha256=sha256(inspect.getsourcefile(inspect.unwrap(_convert_run))),
                  original_specification='https://www.bbci.de/competition/iv/desc_2a.pdf',
                  inference='MAT trials align to trial starts; cue offset supported by original protocol and installed interval',
                  limitation='No local original GDF file available for direct code-768/code-769 timestamp comparison')
    out.write_text(json.dumps(report,indent=2))
    print(f'Verified {len(checks)} runs across {len(set(c["file"] for c in checks))} files')
