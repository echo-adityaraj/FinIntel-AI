"""Small, source-labelled PDF smoke benchmark. Not an independent holdout."""
import csv
import json
import statistics
from app.engine import ROOT, get_engine


def main():
    engine = get_engine()
    if engine.corpus != 'pdf':
        raise SystemExit('Set CORPUS=pdf and your desired retrieval/generation modes.')
    with (ROOT/'data/evaluation/pdf_questions.csv').open(newline='') as f:
        questions = list(csv.DictReader(f))
    results = []
    for r in questions:
        answer = engine.answer(r['question'], 10)
        targets = set(r['relevant_pages'].split('|')) - {''}
        pages = [x['id'].rsplit(':', 1)[0] for x in answer['retrieved']]
        cited = [x['id'].rsplit(':', 1)[0] for x in answer['sources']]
        expected = r['expected_fragments'].split('|')
        rank = next((i for i,p in enumerate(pages, 1) if p in targets), 0)
        match = r['answerable'] == 'true' and all(x.casefold() in answer['answer'].casefold() for x in expected)
        results.append({**r, 'rank':rank, 'supported':answer['supported'], 'fragment_match':match,
                        'citation_page_match':bool(targets.intersection(cited)), 'answer_method':answer['answer_method'],
                        'latency_seconds':answer['latency_seconds'], 'answer':answer['answer'],
                        'sources_json':json.dumps(answer['sources'], ensure_ascii=False), 'manual_groundedness':'unreviewed'})
        print(r['question_id'], answer['answer_method'], flush=True)
    pos = [r for r in results if r['answerable']=='true']
    neg = [r for r in results if r['answerable']=='false']
    mean = statistics.mean
    metrics = {f'Page hit@{k}':mean(0<r['rank']<=k for r in pos) for k in (1,3,5,10)}
    metrics.update({'Expected fragment match':mean(r['fragment_match'] for r in pos),
                    'Labelled citation page match':mean(r['citation_page_match'] for r in pos),
                    'Unsupported refusal rate':mean(not r['supported'] for r in neg),
                    'Answerable response rate':mean(r['supported'] for r in pos),
                    'Mean latency (seconds)':mean(r['latency_seconds'] for r in results)})
    out = ROOT/'results'/'pdf'/f'{engine.retrieval}-{engine.generation}'; out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps({'corpus':'pdf','retrieval':engine.retrieval,'generation':engine.generation,'test_count':len(results),'note':'Exploratory PDF smoke benchmark; not independent or exhaustive. Page matches and fragments do not establish groundedness.','metrics':metrics},indent=2))
    with (out/'evaluation_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    print(json.dumps(metrics,indent=2))

if __name__ == '__main__': main()
