"""Derive source shares/length summaries from saved manifests before finalization."""
import argparse
import json
from pathlib import Path
import pandas as pd
from capstone_person1.pipeline_controls import file_sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);parser.add_argument('--output')
    args=parser.parse_args();release=Path(args.release).resolve()
    output=Path(args.output).resolve() if args.output else release/'reports/manifest_source_shares_and_lengths.json'
    metadata=json.loads((release/'metadata/release_manifest.json').read_text(encoding='utf-8'))
    if metadata['immutable_after_finalization'] and release in output.parents:
        raise ValueError('Write derived statistics outside the immutable release')
    results={}
    for name,summary in metadata['manifests'].items():
        manifest=release/'manifests'/(name+'.jsonl')
        if file_sha256(manifest)!=summary['manifest_sha256']:raise ValueError('Manifest hash differs')
        frame=pd.DataFrame(json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines())
        sources={}
        for source,rows in frame.groupby('dataset_id'):
            sources[source]=dict(rows=len(rows),example_share=len(rows)/len(frame),supervised_tokens=int(rows['supervised_tokens'].sum()),
                reference_token_share=float(rows['supervised_tokens'].sum()/frame['supervised_tokens'].sum()),
                total_input_tokens=rows['total_tokens'].describe(percentiles=[.05,.5,.95]).to_dict(),
                supervised_length=rows['supervised_tokens'].describe(percentiles=[.05,.5,.95]).to_dict(),
                length_bin_counts=rows['length_bin'].value_counts().sort_index().to_dict())
        assert sum(item['rows'] for item in sources.values())==summary['rows']
        assert sum(item['supervised_tokens'] for item in sources.values())==summary['supervised_tokens']
        results[name]=dict(manifest_sha256=file_sha256(manifest),sources=sources)
    report=dict(release_id=release.name,producer_sha256=file_sha256(__file__),budget='reference labels; actual trainer counts pending',manifests=results)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(str(output))

if __name__=='__main__':main()
