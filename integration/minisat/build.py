#!/usr/bin/env python3
"""Build pinned original and instrumented MiniSat outside this repository."""
import argparse, hashlib, pathlib, shutil, subprocess, tarfile, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
URL = 'https://github.com/stp/minisat/archive/refs/tags/releases/2.2.1.tar.gz'
SHA = '432985833596653fcd698ab439588471cc0f2437617d0df2bb191a0252ba423d'
# Applied in order: P1 per-call logging, call timestamps and phase log, buffered binary records.
PATCHES = ['bcp-instrumentation.patch', 'bcp-timestamps.patch', 'bcp-buffered-log.patch']

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--external-root', required=True, type=pathlib.Path)
    ap.add_argument('--archive', type=pathlib.Path, help='Use a previously downloaded archive')
    args = ap.parse_args()
    base = args.external_root.expanduser().resolve()
    if base == ROOT or ROOT in base.parents:
        ap.error('MiniSat source must remain outside this repository')
    if base.exists() and any(base.iterdir()):
        ap.error('Use an empty external directory; existing work is never overwritten')
    base.mkdir(parents=True, exist_ok=True)
    archive = base / 'minisat-2.2.1.tar.gz'
    if args.archive:
        shutil.copyfile(args.archive, archive)
    else:
        urllib.request.urlretrieve(URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA:
        raise ValueError('Source checksum mismatch')
    with tarfile.open(archive) as tf:
        tf.extractall(base, filter='data')
    original = base / 'minisat-releases-2.2.1'
    modified = base / 'minisat-instrumented'
    shutil.copytree(original, modified)
    for name in PATCHES:
        with (ROOT / 'integration/minisat' / name).open() as patch:
            subprocess.run(['patch', '-p1'], cwd=modified, stdin=patch, check=True)
    for source, name in [(original, 'original'), (modified, 'instrumented')]:
        target = base / ('build-' + name)
        subprocess.run(['cmake', '-S', str(source), '-B', str(target),
                        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_STANDARD=11',
                        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'], check=True)
        subprocess.run(['cmake', '--build', str(target), '--target', 'minisat_simp', '-j', '2'], check=True)
        print(target / 'minisat')

if __name__ == '__main__':
    main()
