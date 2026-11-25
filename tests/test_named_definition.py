from app.engine import Engine, Passage, REFUSAL

QUESTION = 'What are the four Consumer Duty outcomes?'
# Synthetic page text with the actual list labels and a paraphrased lead-in.
PAGE = ('1.3 The Duty is comprised of these components. The ‘four outcomes’ which are '
        'rules for four areas: – the governance of products and services – price and value '
        '– consumer understanding, and – consumer support\n1.4 Reasonableness applies.')

def engine(text=PAGE):
    return Engine(passages=[
        Passage('ps22-9:6:0', 'PS22/9', None, 'PDF page 6',
                'Consumer Duty outcomes: fair value, suitability and treatment, confidence, access.', 'original PDF'),
        Passage('fg22-5:3:0', 'FG22/5', 'https://www.fca.org.uk/publication/finalised-guidance/fg22-5.pdf',
                'PDF page 3', 'Introduction', 'original PDF', page_text=text),
    ], corpus='pdf')

def test_correct_definition_even_when_top_search_hit_is_wrong():
    e = engine()
    assert e.search(QUESTION, 1)[0][0].id == 'ps22-9:6:0'
    r = e.answer(QUESTION, 1)
    assert r['supported'] and r['answer_method'] == 'source_definition'
    assert all(x in r['answer'] for x in ['products and services','price and value','consumer understanding','consumer support'])
    assert 'confidence' not in r['answer']
    assert r['sources'][0]['location'] == 'PDF page 3'
    assert r['sources'][0]['id'] == 'fg22-5:3:definition-1.3'
    assert r['sources'][0]['score'] is None
    assert 'page_text' not in r['sources'][0]
    assert 'consumer support' in r['sources'][0]['text']

def test_paraphrased_query():
    assert engine().answer('List the 4 outcomes under the FCA Consumer Duty')['supported']

def test_missing_or_incomplete_definition_refuses():
    e = engine(PAGE.replace('– consumer support',''))
    assert e.answer(QUESTION)['answer'] == REFUSAL

def test_not_a_stored_answer():
    assert 'alternative item' in engine(PAGE.replace('consumer support','alternative item')).answer(QUESTION)['answer']

def test_other_queries_not_routed():
    e = engine()
    assert e.named_definition('What are the four Consumer Duty outcomes for vulnerable customers?') is None
    assert e.named_definition('What is fair value?') is None
