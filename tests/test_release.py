import pytest
from app.engine import Engine, Passage, load_passages, VERSION, excerpt_candidates, verify_selection, split_page
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope='module')
def pdf_engine():
    return Engine(corpus='pdf',retrieval='tfidf',generation='extractive')

@pytest.mark.parametrize('question,expected,page',[
    ('What are the four Consumer Duty outcomes?', ['products and services','price and value','consumer understanding','consumer support'],'fg22-5:3:'),
    ('What is fair value?', ['reasonable','benefits'],'ps22-9:119:'),
    ('When did the rules apply to closed products?', ['31 July 2024'],'ps22-9:11:'),
    ('What are the three cross-cutting rules of the Consumer Duty?', ['good faith','foreseeable harm','financial objectives'],'fg22-5:28:'),
])
def test_actual_pdf_http(pdf_engine, monkeypatch, question,expected,page):
    monkeypatch.setattr('app.main.get_engine',lambda:pdf_engine)
    with TestClient(app) as c:
        r=c.post('/ask',json={'question':question})
        assert r.status_code==200
        result=r.json()
        assert result['supported'] and result['version']==VERSION
        assert all(x.lower() in result['answer'].lower() for x in expected)
        assert result['sources'][0]['id'].startswith(page)
        assert result['sources'][0]['text']

def test_real_pdf_refusal(pdf_engine):
    for q in ['Who won the 2022 FIFA World Cup?','What is the population of Mars?','What is the recipe for chocolate cake?']:
        assert not pdf_engine.answer(q)['supported']

def test_pdf_model_selects_ids_without_transcribing():
    from types import SimpleNamespace
    p=Passage('a:1:0','Example',None,'PDF page 1','Staff must recognise and respond to vulnerability.', 'original PDF')
    engine=Engine(passages=[p],corpus='pdf',threshold=0)
    engine.generation='ollama'
    engine.llm=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content='{"ids":[1]}'))
    r=engine.answer('How should staff respond to vulnerability?')
    assert r['supported'] and r['sources'][0]['text']==p.text
    engine.llm.invoke=lambda messages:SimpleNamespace(content='{"ids":[99]}')
    assert not engine.answer('How should staff respond to vulnerability?')['supported']

def test_selection_invalid_ids():
    p=Passage('a:1:0','Example',None,'PDF page 1','Staff must recognise and respond to vulnerability.', 'original PDF')
    c=excerpt_candidates([(p,.8)],'How should staff respond to vulnerability?')
    assert verify_selection('{"ids":[1]}',c,'How should staff respond to vulnerability?')
    for raw in ('{"ids":[true]}','{"ids":[1,1]}','{"ids":[0]}','{"ids":[]}','no'):
        assert not verify_selection(raw,c,'How should staff respond to vulnerability?')

def test_chunking_stable():
    text='word '*1000
    parts=list(split_page(text)); assert len(parts)>1 and all(len(x)<=700 for x in parts)
    assert all(x in text for x in parts)


def test_pending_metrics():
    with TestClient(app) as c:
        r=c.get('/metrics?corpus=pdf&retrieval=tfidf&generation=ollama')
        assert r.status_code==200 and r.json()['status']=='pending' and r.json()['metrics']=={}
