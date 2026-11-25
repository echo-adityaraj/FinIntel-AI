"""Single-process launcher and real local readiness check."""
import argparse
import json
import os
import sys
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full','pdf','starter'], default='full')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    modes = {'full': ('pdf','semantic','ollama'), 'pdf': ('pdf','tfidf','extractive'), 'starter': ('starter','tfidf','extractive')}
    os.environ.update(zip(('CORPUS','RETRIEVAL_MODE','GENERATION_MODE'),modes[args.mode]))
    from app.engine import VERSION, ROOT, get_engine
    print(f'FinIntel release {VERSION} | {ROOT} | mode={args.mode}', flush=True)
    if args.mode == 'full':
        base = os.getenv('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/')
        model = os.getenv('OLLAMA_MODEL','llama3.2:3b')
        try:
            with urlopen(base+'/api/tags', timeout=10) as r:
                available = [x['name'] for x in json.load(r)['models']]
        except Exception as exc:
            raise SystemExit(f'Ollama not reachable at {base}. Open Ollama or run ollama serve. Details: {exc}')
        if model not in available and model+':latest' not in available:
            raise SystemExit(f'Model {model} not installed. Run ollama pull {model}. Available: {available}')
        print(f'Ollama server reachable; {model} installed.', flush=True)
    print('Loading documents and retrieval index...', flush=True)
    e = get_engine()
    print(f'Loaded {len(e.passages)} passages from {len({p.title for p in e.passages})} documents.', flush=True)
    if args.mode == 'full':
        reply = e.llm.invoke([('human','Return this JSON object: {"ok":true}')]).content
        print('Real Ollama inference responded:', reply, flush=True)
    if args.mode != 'starter':
        examples = [
            ('What are the four Consumer Duty outcomes?', ['products and services','price and value','consumer understanding','consumer support']),
            ('What is fair value?', ['reasonable','benefits']),
            ('When did the rules apply to closed products?', ['31 July 2024']),
            ('What are the three cross-cutting rules of the Consumer Duty?', ['good faith','foreseeable harm','financial objectives']),
            ('Who won the 2022 FIFA World Cup?', [])]
        for q, fragments in examples:
            r = e.answer(q)
            good = (r['supported'] and all(x.lower() in r['answer'].lower() for x in fragments)) if fragments else not r['supported']
            print(f"{'PASS' if good else 'FAIL'}: {q} | {r['answer_method']}", flush=True)
            if not good:
                print(r['answer'], flush=True)
                raise SystemExit('Readiness check failed; server not started.')
        if args.mode == 'full':
            q = 'What skills should frontline staff have to support vulnerable consumers?'
            r = e.answer(q)
            good = r['supported'] and all(x in r['answer'].lower() for x in ('recognise','respond'))
            print(f"{'PASS' if good else 'WARNING'}: general Ollama evidence selection: {q}", flush=True)
            if not good: print('General evidence selection requires review:', r['answer'], flush=True)
    print('Readiness checks complete.', flush=True)
    if not args.check_only:
        import uvicorn
        from app.main import app
        uvicorn.run(app, host='127.0.0.1', port=args.port)

if __name__ == '__main__':
    main()
