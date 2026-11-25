import json
from types import SimpleNamespace
from app.engine import Engine, Passage, date_evidence_candidates, verify_quotes, REFUSAL

Q='When did the rules apply to closed products?'
# Short synthetic fixtures; not a substitute for running actual downloaded PDFs.
CORRECT='For closed products or services the rules come into force on 31 July 2024.'
DECOY='Closed products manufactured before 29 July 2022 must comply with the relevant standards.'
P=Passage('a:11:0','Policy',None,'PDF page 11','Implementation overview','original PDF',page_text='• Rules for open products come into force on 31 July 2023. • '+CORRECT)
D=Passage('b:126:0','Other policy',None,'PDF page 126',DECOY,'original PDF',page_text=DECOY)

def test_date_candidates_find_correct_page_outside_top_k():
    result=date_evidence_candidates([D,P],Q)
    assert result and result[0][0].location=='PDF page 11'
    assert result[0][0].text==CORRECT
    assert result[0][1] is None

def test_date_query_with_different_topic():
    text='The reporting standards for insurance firms take effect on 15 March 2027.'
    p=Passage('z:2:0','Other',None,'PDF page 2',text,'original PDF')
    assert date_evidence_candidates([P,p],'When do reporting standards for insurance firms take effect?')[0][0].text==text

def test_out_of_scope_has_no_candidate():
    assert not date_evidence_candidates([P,D],'When did Argentina win the World Cup?')

def test_reject_unrelated_date_for_applicability():
    raw=json.dumps({'evidence':[{'citation':1,'quote':DECOY}]})
    assert not verify_quotes(raw,[(D,None)],Q)

def test_date_candidates_are_deduplicated():
    from dataclasses import replace
    assert len(date_evidence_candidates([P,replace(P,id='a:11:1')],Q))==1

def test_direct_date_even_if_top_k_is_wrong():
    engine=Engine(passages=[D,P],threshold=0)
    engine.corpus='pdf'; engine.generation='ollama'
    engine.search=lambda *a,**k:[(D,.9)]
    def invoke(messages):
        raise AssertionError('A direct source date must not call Ollama')
    engine.llm=SimpleNamespace(invoke=invoke)
    response=engine.answer(Q)
    assert response['supported'] and response['sources'][0]['location']=='PDF page 11'
    assert response['sources'][0]['score'] is None
    assert response['answer_method']=='source_date'
