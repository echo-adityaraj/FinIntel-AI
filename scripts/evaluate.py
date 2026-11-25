"""Evaluate starter corpus labels; no claims of independent human ground-truth review."""
import csv,json,statistics
from app.engine import ROOT,get_engine,REFUSAL

def main():
    engine=get_engine()
    if engine.corpus!='starter': raise SystemExit('Starter labels do not apply to PDF corpora. Create new passage labels before evaluation.')
    rows=list(csv.DictReader((ROOT/'data/evaluation/questions.csv').open()))
    def run(rows):
        results=[]
        for r in rows:
            d=engine.answer(r['question'],10); ids=[x['id'] for x in d['retrieved']]; relevant=r['relevant_id']; answerable=r['answerable']=='true'
            result={**r,'rank':ids.index(relevant)+1 if relevant in ids else 0,'supported':d['supported'],'answer':d['answer'],'latency_seconds':d['latency_seconds'],'citation_match':any(s['id']==relevant for s in d['sources']),'fragment_match':bool(answerable and r['expected_fragment'].casefold() in d['answer'].casefold()),'verbatim_evidence':bool(d['supported'] and len(d['sources'])==1 and d['answer']==d['sources'][0]['text']+' [1]')}
            results.append(result)
        return results
    results=run([r for r in rows if r['split']=='test']); pos=[r for r in results if r['answerable']=='true']; neg=[r for r in results if r['answerable']=='false']; supported=[r for r in results if r['supported']]
    mean=lambda seq: statistics.mean(seq) if seq else 0
    metrics={f'Recall@{k}':mean([0<r['rank']<=k for r in pos]) for k in [1,3,5,10]}
    metrics.update({'MRR@10':mean([1/r['rank'] if r['rank'] else 0 for r in pos]),'Expected fragment match':mean([r['fragment_match'] for r in pos]),'Relevant citation rate':mean([r['citation_match'] for r in pos]),'Unsupported refusal rate':mean([not r['supported'] for r in neg]),'Answerable response rate':mean([r['supported'] for r in pos]),'Mean latency (seconds)':mean([r['latency_seconds'] for r in results])})
    if engine.generation=='extractive':metrics['Verbatim evidence rate']=mean([r['verbatim_evidence'] for r in supported])
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps({'corpus':engine.corpus,'retrieval':engine.retrieval,'generation':engine.generation,'threshold':engine.threshold,'test_count':len(results),'development_count':len(rows)-len(results),'metrics':metrics},indent=2))
    with (out/'evaluation_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
