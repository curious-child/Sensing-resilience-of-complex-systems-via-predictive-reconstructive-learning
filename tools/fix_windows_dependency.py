"""Remove torch-timeseries 0.1.10's unused POSIX-only resource import on Windows.

This reproduces the existing project's dependency compatibility adjustment;
it does not change any neural-network layer or numerical computation.
"""
from pathlib import Path
from importlib.metadata import distribution
import ast
import sys

if sys.platform != 'win32':
    print('No Windows compatibility adjustment needed.')
else:
    package=distribution('torch-timeseries')
    if package.version!='0.1.10':
        raise RuntimeError('Only torch-timeseries 0.1.10 is supported by this adjustment')
    for path in Path(package.locate_file('torch_timeseries/dataset')).glob('*.py'):
        source=path.read_text(encoding='utf-8')
        old='import resource\n'
        if old not in source.splitlines(keepends=True):
            continue
        if any(isinstance(n,ast.Name) and n.id=='resource' for n in ast.walk(ast.parse(source))):
            raise RuntimeError('resource is used by this installation; refusing to change '+path.name)
        source=''.join('# Windows compatibility: unused POSIX resource import removed.\n' if line==old else line for line in source.splitlines(keepends=True))
        path.write_text(source,encoding='utf-8')
        print('Removed unused resource import from',path.name)
    print('Windows compatibility check complete.')
