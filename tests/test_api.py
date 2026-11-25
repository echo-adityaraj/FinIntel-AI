import pytest
from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health():
    r=client.get('/health'); assert r.status_code==200 and r.json()['status']=='healthy'
def test_home():
    r=client.get('/'); assert r.status_code==200 and 'FinIntel AI' in r.text

def test_openapi(): assert '/ask' in client.get('/openapi.json').json()['paths']
def test_ask():
    r=client.post('/ask',json={'question':'What is fair value?'}); assert r.status_code==200 and r.json()['supported']
def test_unsupported():
    r=client.post('/ask',json={'question':'What is the capital of Japan?'}); assert r.status_code==200 and not r.json()['supported']
@pytest.mark.parametrize('body',[{}, {'question':''},{'question':'   '},{'question':'a'*2001},{'question':'test','k':0},{'question':'test','k':21}])
def test_validation(body): assert client.post('/ask',json=body).status_code==422

def test_metrics():
    r=client.get('/metrics'); assert r.status_code==200 and r.json()['test_count']==44

def test_error_handling(monkeypatch):
    def fail(): raise RuntimeError('internal secret')
    monkeypatch.setattr('app.main.get_engine',fail)
    for r in [client.get('/health'),client.post('/ask',json={'question':'test'})]:
        assert r.status_code==503 and 'internal secret' not in r.text
