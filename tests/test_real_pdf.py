"""Optional real-document regression; no model downloads required."""
import json
import pytest
from app.engine import ROOT, Passage, direct_date_evidence, Engine


def test_actual_pdf_date_and_definition():
    PdfReader = pytest.importorskip('pypdf').PdfReader
    pytest.importorskip('cryptography')
    manifest_file = ROOT / 'data/raw/manifest.json'
    if not manifest_file.exists():
        pytest.skip('Download original PDFs first')
    manifest = json.loads(manifest_file.read_text())
    passages = []
    for name, page_numbers in [('ps22-9.pdf', [8, 11, 149]), ('fg22-5.pdf', [3])]:
        path = ROOT / 'data/raw' / name
        if not path.exists():
            pytest.skip('Download original PDFs first')
        reader = PdfReader(path)
        for page_no in page_numbers:
            text = reader.pages[page_no-1].extract_text()
            passages.append(Passage(f'{path.stem}:{page_no}:0', manifest[name]['title'], manifest[name]['url'], f'PDF page {page_no}', text, 'original PDF', page_text=text))
    status, evidence = direct_date_evidence(passages, 'When did the rules apply to closed products?')
    assert status == 'unique'
    assert '31 July 2024' in evidence[0][1].text
    assert evidence[0][1].id.startswith('ps22-9:11:')
    engine = Engine(passages=passages, corpus='pdf')
    response = engine.answer('What are the four Consumer Duty outcomes?')
    assert response['supported'] and response['answer_method'] == 'source_definition'
    for fragment in ['products and services', 'price and value', 'consumer understanding', 'consumer support']:
        assert fragment in response['answer']
