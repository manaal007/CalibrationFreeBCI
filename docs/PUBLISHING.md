# Publishing this research repository

The Git-visible tree contains source, tests, documentation, historical evidence
and curated results. Local datasets, model artifacts, validation environments,
experiment run directories and preservation ZIPs are excluded by `.gitignore`.
Keep the preserved ZIPs and local models in your own backup; they are not part of
a normal Git clone. A fresh clone must retrain from locally available source data
to reproduce the frozen models.

Run the publication check before adding files:

```powershell
.\.venv\Scripts\python.exe scripts/check_publication.py --out results/publication_check.json
```

This checks file types, local-environment directories and 50 MiB size limits.
It does not prove arbitrary text is free of credentials; review source before
publishing. Never use `git add -f` on datasets, caches or preservation archives.

The GitHub Actions workflow installs the pinned environment and runs tests on
Windows/Python 3.12. It does not download EEG datasets or run external evaluation.
Its [first remote execution passed](https://github.com/manaal007/CalibrationFreeBCI/actions/runs/39964815136),
including pinned installation and the test suite on Windows/Python 3.12.

Use a truthful repository description, for example:

> Reproducible motor imagery EEG transfer research: verified preprocessing,
> participant-separated validation, label-free alignment and frozen external evaluation.

Do not advertise reliable cross-dataset performance as established. The provisional
five-participant PhysioNet comparison has wide uncertainty and below-chance
participants. Its cohort history remains unconfirmed. Preserve the full results.

The publication target is
[manaal007/CalibrationFreeBCI](https://github.com/manaal007/CalibrationFreeBCI).
The local verified snapshot was committed before merging the existing GitHub
README history. The project was pushed to `main` without a force-push; the original
research branch and preservation archives remain available locally. Check the
repository's Actions tab for the actual remote CI outcome.
