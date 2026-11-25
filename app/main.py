from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from app.engine import get_engine, ROOT, VERSION
import json
import logging
logger=logging.getLogger(__name__)

app=FastAPI(title='FinIntel AI — Financial Document Intelligence',version=VERSION)
class Question(BaseModel):
    question: str=Field(min_length=1,max_length=2000)
    k: int=Field(default=5,ge=1,le=20)
    @field_validator('question')
    @classmethod
    def nonblank(cls,v):
        if not v.strip(): raise ValueError('Question must not be blank')
        return v.strip()

@app.get('/',include_in_schema=False)
def home(): return FileResponse(ROOT/'app/static/index.html')
@app.get('/health')
def health():
    try:
        engine=get_engine()
        return {'status':'healthy','version':VERSION,'retrieval':engine.retrieval,'generation':engine.generation,'corpus':engine.corpus,'passages':len(engine.passages),'documents':len({p.title for p in engine.passages})}
    except Exception as e:
        logger.exception('Engine initialisation failed')
        raise HTTPException(503,'Engine unavailable; check configuration and server logs.') from e
@app.post('/ask')
def ask(request:Question):
    try: return get_engine().answer(request.question,request.k)
    except Exception as e:
        logger.exception('Answer generation failed')
        raise HTTPException(503,'Answer service unavailable; check model, data and server logs.') from e
@app.get('/metrics')
def metrics(corpus: str = 'starter', retrieval: str = 'tfidf', generation: str = 'extractive'):
    if corpus not in ('starter', 'pdf'):
        raise HTTPException(400, 'Unknown evaluation corpus.')
    if retrieval not in ('tfidf','semantic') or generation not in ('extractive','ollama'):
        raise HTTPException(400,'Unknown evaluation configuration.')
    path=ROOT/'results/metrics.json' if corpus == 'starter' else ROOT/'results/pdf'/f'{retrieval}-{generation}'/'metrics.json'
    if not path.exists():
        return {'status':'pending','corpus':corpus,'retrieval':retrieval,'generation':generation,'test_count':0,'metrics':{},'note':'No measurements for this configuration. Run the evaluation script.'}
    return json.loads(path.read_text())

