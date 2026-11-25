import pytest
from app.engine import Engine, Passage, ROOT, explicit_definition_evidence


def passage(text, ident='test:1:0'):
    return Passage(ident, 'Test', None, 'PDF page 1', text, 'original PDF', page_text=text)


def test_general_definition():
    p = passage('A reserve is discussed. Liquidity means the ability to meet payment obligations.')
    assert explicit_definition_evidence([p], 'What is liquidity?').text == 'Liquidity means the ability to meet payment obligations.'


def test_generic_regulatory_condition():
    p = passage('A product provides adequate protection where the coverage meets the stated needs.')
    assert explicit_definition_evidence([p], 'What is adequate protection?').text == p.text


def test_mentions_not_definitions():
    assert explicit_definition_evidence([passage('Firms must assess fair value annually.')], 'What is fair value?') is None


def test_conflicting_definitions_not_automatically_selected():
    a = passage('Liquidity means readily available cash.')
    b = passage('Liquidity means access to external funding.', 'test:2:0')
    assert explicit_definition_evidence([a,b], 'Define liquidity') is None


def test_actual_fca_fair_value():
    PdfReader = pytest.importorskip('pypdf').PdfReader
    pytest.importorskip('cryptography')
    paths = sorted((ROOT/'data/raw').glob('*.pdf'))
    if len(paths) < 3:
        pytest.skip('Original PDFs not present')
    pages = []
    for path in paths:
        for i, page in enumerate(PdfReader(path).pages,1):
            text = page.extract_text() or ''
            if text.strip():
                pages.append(Passage(f'{path.stem}:{i}:0', path.stem,None,f'PDF page {i}',text,'original PDF',page_text=text))
    engine = Engine(passages=pages, corpus='pdf')
    engine.generation = 'ollama'
    # No mock answer: source extraction must succeed without calling an LLM.
    for q in ('What is fair value?', 'Define fair value', 'What does fair value mean?'):
        result = engine.answer(q)
        assert result['supported'] and result['answer_method'] == 'source_definition'
        assert result['sources'][0]['id'].startswith('ps22-9:119:')
        assert 'reasonable relative to the benefits' in result['answer']
