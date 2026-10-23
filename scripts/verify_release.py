#!/usr/bin/env python3
"""Verify the TRACE V4 release without changing scientific artifacts."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
MANIFEST=BASE/'ARTIFACT_MANIFEST.csv'

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def emit(status,item,detail): print(f"{status:13} {item}: {detail}")

def main()->int:
    failed=False
    if not MANIFEST.exists(): emit('FAIL','manifest','missing'); return 1
    with MANIFEST.open(newline='',encoding='utf-8') as f: manifest=list(csv.DictReader(f))
    for row in manifest:
        rel=row['release_path'].strip()
        if not rel:
            emit('NOT AVAILABLE',row['artifact_id'],row['verification_status']); continue
        p=BASE/rel
        if not p.exists(): emit('FAIL',row['artifact_id'],f'missing {rel}'); failed=True; continue
        actual=sha(p)
        if actual!=row['sha256']: emit('FAIL',row['artifact_id'],'checksum mismatch'); failed=True
        else: emit('PASS',row['artifact_id'],rel)
    for name in ['pipeline.pdf','figure2_rollout_fidelity.pdf','figure3_temporal_dynamics.pdf','figure4_interval_coverage.pdf','figure5_observation_density.pdf']:
        p=BASE/'figures'/name
        if p.exists() and p.stat().st_size>1000: emit('PASS',f'figure {name}',f'{p.stat().st_size} bytes')
        else: emit('FAIL',f'figure {name}','missing or empty'); failed=True
    cfg=BASE/'configs'/'INTERNAL_MODELLING.json'
    try: json.loads(cfg.read_text(encoding='utf-8')); emit('PASS','configuration','valid JSON')
    except Exception as exc: emit('FAIL','configuration',str(exc)); failed=True
    expected=BASE/'RESULTS_VERIFICATION.csv'
    if expected.exists():
        with expected.open(newline='',encoding='utf-8') as f: vr=list(csv.DictReader(f))
        bad=[r for r in vr if r['status']!='MATCH']
        emit('PASS' if len(vr)==59 and not bad else 'FAIL','results verification',f'{len(vr)} rows; {len(bad)} non-matches')
        failed |= len(vr)!=59 or bool(bad)
    else: emit('NOT AVAILABLE','results verification','file missing')
    print('FAIL' if failed else 'PASS')
    return 1 if failed else 0

if __name__=='__main__': sys.exit(main())
