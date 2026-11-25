import pytest
from app.engine import Engine, Passage, REFUSAL, load_passages

@pytest.fixture(scope='module')
def engine(): return Engine()

def test_corpus():
    p=load_passages(); assert len(p)==16 and len({x.id for x in p})==len(p)
    assert all(x.url.startswith('https://') and x.text for x in p)

def test_retrieval(engine):
    assert engine.search('What is fair value?',3)[0][0].id=='value'
    scores=[s for _,s in engine.search('consumer duty',5)]
    assert scores==sorted(scores,reverse=True)

def test_citations(engine):
    r=engine.answer('What are the four Consumer Duty outcomes?')
    assert r['supported'] and r['sources'][0]['id']=='outcomes'
    assert r['answer']==r['sources'][0]['text']+' [1]'
    assert r['latency_seconds']>=0

@pytest.mark.parametrize('question',['Who won the 2022 FIFA World Cup?','What is the exact fine for breaching Consumer Duty?','What is my bank account balance?','zzzzzzzz'])
def test_refusal(engine,question):
    r=engine.answer(question); assert r['answer']==REFUSAL and not r['sources'] and not r['supported']

def test_k_clamped(engine): assert len(engine.search('consumer',100))==len(engine.passages)

def test_invalid_config():
    with pytest.raises(ValueError): Engine(retrieval='invalid')
    with pytest.raises(ValueError): Engine(passages=[])
