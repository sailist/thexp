import os

for root, dirs, fs in os.walk('thexp/cli/templete'):
    for f in fs:
        if f.endswith('.py'):
            f = os.path.join(root, f)
            prefn, ext = os.path.splitext(f)

            os.rename(f, '{}.py-tpl'.format(prefn))
