#!/usr/bin/env python3
import pkg_resources

# List all installed packages
packages = sorted([str(d) for d in pkg_resources.working_set])
for p in packages:
    if 'hops' in p.lower() or 'sklearn' in p.lower() or 'pandas' in p.lower() or 'numpy' in p.lower():
        print(p)

# Also check hopsworks version
import hopsworks
print(f"\nHopsworks version: {hopsworks.version}")
print(f"Hopsworks location: {hopsworks.__file__}")

# List all modules in hopsworks package
import os
import hopsworks as hw
print(f"\nHopsworks package location: {hw.__path__}")
for root, dirs, files in os.walk(hw.__path__[0]):
    level = root.replace(hw.__path__[0], '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith('.py') and not file.startswith('__'):
            print(f'{subindent}{file}')
