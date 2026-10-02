import ast, json
from pathlib import Path
n=json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())
assert n['nbformat']==4
for c in n['cells']:
    if c['cell_type']=='code':
        s=''.join(c['source'])
        ast.parse('\n'.join(x for x in s.splitlines() if not x.startswith('%')))
        assert not c['outputs']
        assert 'max_embeddings_multiples=' not in s
print('Notebook JSON, cell syntax and clean outputs OK')
