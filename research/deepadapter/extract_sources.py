from pathlib import Path
import json
import hashlib
import fitz

root = Path(__file__).resolve().parent
out = root / 'sources'
out.mkdir(exist_ok=True)
sources = {
    'spmnet': Path('C:/Users/12269/Desktop/SPNet/SPMNet Anonymous.pdf'),
    'nightadapter': Path('C:/Users/12269/Desktop/NightAdapter_Learning_a_Frequency_Adapter_for_Generalizable_Night-time_Scene_Segmentation.pdf'),
    'unext': Path('C:/Users/12269/Desktop/UNext.pdf'),
}
manifest = {}
for name, source in sources.items():
    doc = fitz.open(source)
    folder = out / name
    folder.mkdir(exist_ok=True)
    pages = []
    for i, page in enumerate(doc):
        txt = page.get_text(sort=True)
        (folder / f'page_{i+1:02d}.txt').write_text(txt, encoding='utf-8')
        pages.append(f'\n=== PDF PAGE {i+1} ===\n{txt}')
    (out / f'{name}.txt').write_text('\n'.join(pages), encoding='utf-8')
    manifest[name] = {'source': str(source), 'pages': len(doc), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    print(name, len(doc), 'pages')
    doc.close()
(out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
