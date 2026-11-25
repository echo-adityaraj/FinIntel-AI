import json
from app.engine import Passage, verify_quotes
P=Passage('p:1:0','Test',None,'PDF page 1','The rules apply to closed products from 31 July 2024. Other rules apply from 2022.','original PDF')

def candidate(quote,citation=1):
    return json.dumps({'evidence':[{'citation':citation,'quote':quote}]})

def test_exact_quote_passes():
    result=verify_quotes(candidate('The rules apply to closed products from 31 July 2024.'),[(P,.8)],'When did the rules apply to closed products?')
    assert result[0][1].text=='The rules apply to closed products from 31 July 2024.'
    assert result[0][1].location=='PDF page 1'

def test_invented_date_refused():
    assert not verify_quotes(candidate('The rules apply to closed products from 31 July 2025.'),[(P,.8)],'When did the rules apply?')

def test_wrong_citation_refused():
    assert not verify_quotes(candidate(P.text,2),[(P,.8)],'When did the rules apply?')

def test_date_question_needs_year():
    assert not verify_quotes(candidate('The rules apply to closed products'),[(P,.8)],'When did the rules apply?')

def test_whitespace_normalised():
    assert verify_quotes(candidate('The rules apply to closed products\nfrom 31 July 2024.'),[(P,.8)],'When?')

def test_invalid_response_refused():
    for v in ('answer [1]','{}','{"evidence": []}','{"evidence": [null]}'):
        assert not verify_quotes(v,[(P,.8)],'When?')

def test_unrelated_date_refused():
    q='When did the rules apply to closed products?'
    assert not verify_quotes(candidate('Other rules apply from 2022.'),[(P,.8)],q)

def test_model_selection_end_to_end():
    from types import SimpleNamespace
    from app.engine import Engine, REFUSAL
    e=Engine(passages=[P],threshold=0)
    e.generation='ollama'
    e.llm=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content=candidate('The rules apply to closed products from 31 July 2024.')))
    r=e.answer('When did the rules apply to closed products?')
    assert r['supported'] and r['answer_method']=='verified_excerpt'
    assert r['answer']==r['sources'][0]['text']+' [1]'
    e.llm.invoke=lambda messages:SimpleNamespace(content=candidate('The rules apply to closed products from 31 July 2025.'))
    assert e.answer('When did the rules apply to closed products?')['answer']==REFUSAL

def test_hybrid_search_prefers_diverse_pages():
    import numpy as np
    from types import SimpleNamespace
    from app.engine import Engine
    p=[Passage('a:1:0','Test',None,'PDF page 1','closed products rules','original PDF'),Passage('a:1:1','Test',None,'PDF page 1','closed products rules and dates','original PDF'),Passage('a:2:0','Test',None,'PDF page 2','closed products apply from 31 July 2024','original PDF')]
    e=Engine(passages=p,threshold=.5)
    e.retrieval='semantic'
    e.model=SimpleNamespace(encode=lambda *a,**k:[[1.,0.]])
    e.index=SimpleNamespace(search=lambda q,k:(np.array([[.9,.89,.85]]),np.array([[0,1,2]])))
    result=e.search('closed products rules',2)
    assert len({x.location for x,_ in result})==2
