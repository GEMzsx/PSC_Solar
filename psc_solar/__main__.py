import argparse
import json
from pathlib import Path

import pandas as pd

from .data import prepare
from .model import load_bundle, predict, train


def main():
    parser = argparse.ArgumentParser(description='PSC Solar offline research pipeline')
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', help='Clean pinned raw data, split by paper, train and evaluate')
    fit.add_argument('--raw', type=Path, default=Path('data/raw/Perovskite_database_content_all_data.csv'))
    fit.add_argument('--processed', type=Path, default=Path('data/processed'))
    fit.add_argument('--output', type=Path, default=Path('artifacts/baseline'))
    fit.add_argument('--seed', type=int, default=42)
    inference = commands.add_parser('predict', help='Predict normalized input CSV')
    inference.add_argument('--model', type=Path, default=Path('artifacts/baseline/model.joblib'))
    inference.add_argument('--input', type=Path, required=True)
    inference.add_argument('--output', type=Path, required=True)
    small = commands.add_parser('small-sample', help='Rehearse 50/100/200-row training on validation papers; test set untouched')
    small.add_argument('--clean', type=Path, default=Path('data/processed/clean.csv'))
    small.add_argument('--output', type=Path, default=Path('artifacts/baseline/small_sample_results.csv'))
    ingest_command = commands.add_parser('ingest', help='Validate and merge filled experimental template; no automatic retraining')
    ingest_command.add_argument('--input', type=Path, required=True)
    ingest_command.add_argument('--output', type=Path, default=Path('data/lab'))
    args = parser.parse_args()
    if args.command == 'train':
        frame, cleaning = prepare(args.raw, args.processed)
        print(json.dumps(cleaning, ensure_ascii=False, indent=2), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'cleaning_report.json').write_text(json.dumps(cleaning, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        train(frame, args.output, cleaning, args.seed)
    elif args.command == 'small-sample':
        from .small_sample import run
        run(args.clean, args.output)
    elif args.command == 'ingest':
        from .experiments import ingest
        ingest(args.input, args.output)
    else:
        frame = pd.read_csv(args.input, keep_default_na=False)
        results = predict(load_bundle(args.model), frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([frame.reset_index(drop=True), results.reset_index(drop=True)], axis=1).to_csv(args.output, index=False)
        print(f'Saved {len(results)} predictions to {args.output}')


if __name__ == '__main__':
    main()
