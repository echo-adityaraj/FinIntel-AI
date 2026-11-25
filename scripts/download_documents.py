"""Download original FCA PDFs explicitly; no downloading on startup."""
import json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOCS={
'fg22-5.pdf':{'title':'FG22/5 — Final non-Handbook Guidance for firms on the Consumer Duty','url':'https://www.fca.org.uk/publication/finalised-guidance/fg22-5.pdf'},
'fg21-1.pdf':{'title':'FG21/1 — Guidance for firms on the fair treatment of vulnerable customers','url':'https://www.fca.org.uk/publication/finalised-guidance/fg21-1.pdf'},
'ps22-9.pdf':{'title':'PS22/9 — A new Consumer Duty','url':'https://www.fca.org.uk/publication/policy/ps22-9.pdf'},
}
def main():
    root=ROOT/'data/raw'; root.mkdir(parents=True,exist_ok=True); manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}; failures=[]
    for name,meta in DOCS.items():
        try:
            req=urllib.request.Request(meta['url'],headers={'User-Agent':'FinIntelPortfolio/1.0'})
            with urllib.request.urlopen(req,timeout=60) as response: data=response.read(50_000_001)
            if not data.startswith(b'%PDF-') or len(data)>50_000_000: raise ValueError('Response is not a valid-sized PDF')
            (root/name).write_bytes(data); manifest[name]=meta; print('Downloaded',name)
        except Exception as e: failures.append(name); print('Failed',name,str(e))
    manifest_path.write_text(json.dumps(manifest,indent=2))
    if failures: raise SystemExit('Some downloads failed. Download manually from the listed official URLs and rerun or edit the manifest.')
if __name__=='__main__': main()
