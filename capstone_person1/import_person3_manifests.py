"""Validate/import frozen question inputs without generating Person 3 evaluation data."""
import argparse
import json
from pathlib import Path

from capstone_person1.benchmark_manifests import load_protection_inputs, export_protection_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('capstone_person1/inputs_from_team'))
    parser.add_argument('--output-root', type=Path, required=True,
                        help='A diagnostic import folder or immutable run output folder')
    args = parser.parse_args()
    bundle = load_protection_inputs(args.input_dir / 'reserved_evaluation.csv',
        args.input_dir / 'reserved_evaluation_metadata.json',
        args.input_dir / 'frozen_development.csv', args.input_dir / 'frozen_development_metadata.json')
    report = export_protection_inputs(bundle, args.output_root / 'processed', args.output_root / 'reports')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
