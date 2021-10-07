from typing import List

import spacy
from fastapi import FastAPI
from models import Payload, Entities

app = FastAPI()
nlp = spacy.load("en_core_web_sm")


@app.post('/ner-service')
async def get_ner(payload: Payload):
    tokenize_content: List[spacy.tokens.doc.Doc] = [
        nlp(content.content) for content in payload.data
    ]
    document_entities = []
    for doc in tokenize_content:
        document_entities.append(
            [{'text': ent.text, 'entity_type': ent.label_} for ent in doc.ents]
        )
    return [
        Entities(post_url=post.post_url, entities=ents)
        for post, ents in zip(payload.data, document_entities)
    ]


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
