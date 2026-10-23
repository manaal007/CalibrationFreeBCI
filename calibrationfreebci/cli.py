import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Auditable EEG research")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--root", type=Path, default=Path.cwd())
    audit.add_argument("--out", type=Path, required=True)
    source = sub.add_parser('source', help='Offline raw EEG through source LOSO and frozen model')
    source.add_argument('--out', type=Path, required=True)
    source.add_argument('--subjects', type=int, nargs='+', default=list(range(1,10)))
    source.add_argument('--data-root', type=Path)
    external = sub.add_parser('external', help='Run frozen independent evaluation after approval')
    external.add_argument('--source', type=Path, required=True)
    external.add_argument('--out', type=Path, required=True)
    external.add_argument('--subjects', type=int, nargs='+', required=True)
    external.add_argument('--data-root', type=Path)
    external.add_argument('--approved-confirmatory-evaluation','--approved-external-evaluation', action='store_true', required=True)
    external.add_argument('--evaluation-kind',choices=['confirmatory','provisional'],default='confirmatory')
    external.add_argument('--exposure-record', type=Path, required=True)
    freeze = sub.add_parser('freeze-comparisons')
    freeze.add_argument('--source', type=Path, required=True)
    freeze.add_argument('--data-root', type=Path)
    report = sub.add_parser('report')
    report.add_argument('--source', type=Path, required=True)
    report.add_argument('--out', type=Path, required=True)
    nested = sub.add_parser('nested', help='Nested participant validation of a small filter-bank grid')
    nested.add_argument('--out', type=Path, required=True)
    nested.add_argument('--config', type=Path, default=Path(__file__).with_name('default_nested.json'))
    nested.add_argument('--data-root', type=Path)
    smoke = sub.add_parser('cross-smoke', help='Exploratory inference on previously examined Cho2017')
    smoke.add_argument('--source',type=Path,required=True)
    smoke.add_argument('--out',type=Path,required=True)
    smoke.add_argument('--subjects',type=int,nargs='+',default=[1])
    smoke.add_argument('--data-root',type=Path)
    verify = sub.add_parser('verify')
    verify.add_argument('--run',type=Path,required=True)
    timing = sub.add_parser('verify-timing')
    timing.add_argument('--out',type=Path,required=True)
    timing.add_argument('--data-root',type=Path)
    workflow = sub.add_parser('workflow',help='Nested source development through exploratory cross-dataset report')
    workflow.add_argument('--out',type=Path,required=True)
    workflow.add_argument('--config',type=Path,default=Path(__file__).with_name('default_nested.json'))
    workflow.add_argument('--data-root',type=Path)
    workflow.add_argument('--cho-root',type=Path)
    final_report = sub.add_parser('research-report')
    final_report.add_argument('--source',type=Path,required=True)
    final_report.add_argument('--external',type=Path,required=True)
    final_report.add_argument('--out',type=Path,required=True)
    qc = sub.add_parser('target-qc',help='Check real PhysioNet loading without predictions or decoding metrics')
    qc.add_argument('--out',type=Path,required=True)
    qc.add_argument('--subjects',type=int,nargs='+',default=[1,2,3,4,5])
    qc.add_argument('--data-root',type=Path)
    alignment=sub.add_parser('alignment',help='Source LOSO of a fixed label-free participant-alignment method')
    alignment.add_argument('--out',type=Path,required=True)
    alignment.add_argument('--data-root',type=Path)
    external_report=sub.add_parser('external-report',help='Generate external tables and participant figures from saved outputs')
    external_report.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    if args.command == "audit":
        from .audit import run
        run(args.root.resolve(), args.out.resolve())
    elif args.command == 'source':
        from .experiment import run
        run(args.out, args.subjects, args.data_root)
    elif args.command == 'external':
        from .external import evaluate
        evaluate(args.source, args.out, args.subjects, args.data_root, args.exposure_record,args.evaluation_kind)
    elif args.command == 'freeze-comparisons':
        from .freeze import freeze
        freeze(args.source,args.data_root)
    elif args.command == 'report':
        from .report import publish_local
        publish_local(args.source, args.out)
    elif args.command == 'nested':
        from .nested import run_nested
        run_nested(args.out, args.config, args.data_root)
    elif args.command == 'cross-smoke':
        from .verification import exploratory_smoke
        exploratory_smoke(args.source,args.out,args.subjects,args.data_root)
    elif args.command == 'verify':
        from .verification import verify_run
        print(verify_run(args.run))
    elif args.command == 'verify-timing':
        from .timing import verify_timing
        verify_timing(args.out,args.data_root)
    elif args.command == 'workflow':
        from .workflow import workflow
        workflow(args.out,args.config,args.data_root,args.cho_root)
    elif args.command == 'research-report':
        from .workflow import generate_report
        generate_report(args.source,args.external,args.out)
    elif args.command == 'target-qc':
        from .quality import check_physionet
        check_physionet(args.out,args.subjects,args.data_root)
    elif args.command == 'alignment':
        from .alignment import run_alignment
        run_alignment(args.out,args.data_root)
    elif args.command == 'external-report':
        from .external_report import generate_external_report
        generate_external_report(args.run)


if __name__ == "__main__":
    main()
