from app.engine import Engine, Passage, direct_date_evidence, REFUSAL
from types import SimpleNamespace

Q='When did the rules apply to closed products?'
# Transcribed from the user's diagnostic screenshot, not a live PDF download.
ACTUAL=('FCA 2022/31 Page 57 of 68 Material to which the transitional provision applies '
        'Transitional Provision Transitional Provision: dates in force Handbook provision: '
        'coming into force 4 Principle 12, PRIN 2A R Except to the extent specified in '
        'PRIN TP5 and TP6, the provisions listed in column 2 only apply to a closed product '
        'from 31 July 2024.')

def p(text, page=1):
    return Passage(f'policy:{page}:0','Policy','https://example.org/policy.pdf',f'PDF page {page}',text,'original PDF',page_text=text)

def test_screenshot_clause():
    status, result=direct_date_evidence([p(ACTUAL)], Q)
    assert status=='unique'
    assert result[0][1].text==ACTUAL
    assert result[0][1].location=='PDF page 1'

def test_different_date_topic():
    text='The reporting standards for insurance firms take effect on 15 March 2027.'
    assert direct_date_evidence([p(text)], 'When do reporting standards for insurance firms take effect?')[0]=='unique'

def test_conflicting_dates():
    assert direct_date_evidence([p(ACTUAL),p(ACTUAL.replace('2024','2025'),2)], Q)==('ambiguous',[])

def test_same_date_different_sources():
    assert direct_date_evidence([p(ACTUAL),p(ACTUAL,2)],Q)[0]=='unique'

def test_unrelated_manufacturing_date():
    assert direct_date_evidence([p('Rules apply to closed products manufactured on 29 July 2022.')],Q)==('none',[])

def test_negated_applicability():
    assert direct_date_evidence([p('Rules do not apply to closed products from 31 July 2024.')],Q)==('none',[])

def test_multi_date_clause_not_auto_selected():
    assert direct_date_evidence([p('Rules apply to open products from 31 July 2023 and closed products from 31 July 2024.')],Q)==('none',[])

def test_direct_response_no_llm():
    e=Engine(passages=[p(ACTUAL)],corpus='pdf',threshold=0)
    e.generation='ollama'
    def fail(*args): raise AssertionError('Unexpected model call')
    e.llm=SimpleNamespace(invoke=fail)
    r=e.answer(Q)
    assert r['answer_method']=='source_date' and r['supported']
    assert '31 July 2024' in r['answer'] and r['sources'][0]['text']==ACTUAL
