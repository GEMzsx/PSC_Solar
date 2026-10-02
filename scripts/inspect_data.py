"""Inspect the original CSV without modifying it; Python standard library only."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def inspect(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    missing_tokens = {'', 'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'not reported'}
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('CSV headers are empty or duplicated')
        present = Counter()
        examples = {name: [] for name in columns}
        doi = Counter()
        architectures = Counter()
        rows = valid_pce = pce_and_doi = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Inconsistent column count at record {rows + 1}')
            rows += 1
            for name, value in row.items():
                value = value.strip()
                if value.lower() not in missing_tokens:
                    present[name] += 1
                    if len(examples[name]) < 3 and value not in examples[name]:
                        examples[name].append(value)
            reference = row.get('Ref_DOI_number', '').strip().lower()
            if reference not in missing_tokens:
                doi[reference] += 1
            architectures[row.get('Cell_architecture', '').strip()] += 1
            try:
                pce = float(row.get('JV_default_PCE', ''))
                if 0 <= pce <= 100:
                    valid_pce += 1
                    if reference not in missing_tokens:
                        pce_and_doi += 1
            except ValueError:
                pass
    return {
        'source_file': path.name, 'sha256': digest.hexdigest(),
        'row_count': rows, 'column_count': len(columns),
        'missing_tokens': sorted(missing_tokens),
        'unique_doi_count': len(doi), 'valid_pce_count': valid_pce,
        'valid_pce_with_doi_count': pce_and_doi,
        'architectures': dict(architectures),
        'columns': {name: {'present_count': present[name],
                           'missing_fraction': round(1 - present[name] / rows, 6) if rows else None,
                           'examples': examples[name]}
                    for name in columns},
        'limitations': ['Nonmissing is not equivalent to valid.',
                        'PCE range 0..100 is a parsing check, not a training filter.',
                        'Nested process values are preserved, not parsed.'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.csv_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'columns'}, ensure_ascii=False, indent=2))
