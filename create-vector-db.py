from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import sys
from sentence_transformers import SentenceTransformer
import json
import pprint

CREATE_COLLECTION = True
COLLECTION_NAME = "patent-abstracts"
BATCH_SIZE = 128

client = QdrantClient(url="http://172.17.0.2:6333")

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False, width=100)

with open("out.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

rows = rows[0:100000]

docs = [ f'{row["title"]}: {row["abstract"]}' for row in rows ]

#pp.pprint(docs)


model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
#model.half()


if CREATE_COLLECTION:
    client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

document_embeddings = model.encode(docs, batch_size=BATCH_SIZE, show_progress_bar=True,
             convert_to_numpy=True, normalize_embeddings=True)

#print(document_embeddings[0].shape)


print("Inserting docs: ")

#similarity = model.similarity(query_embeddings, document_embeddings)
#print(similarity)

for y in range(0, len(document_embeddings), BATCH_SIZE):
    range_end = y + BATCH_SIZE
    batch = document_embeddings[y:range_end]

    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[ PointStruct(id=(y + i), vector=embed, payload={"id": y + i, "info": docs[y + i] } ) for i, embed in enumerate(batch) ]
    )

    print(operation_info)

print("Test search: ")
query_embeddings = model.encode("semiconductor", prompt_name="query")

search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embeddings,
    with_payload=True,
    limit=3
).points

print(search_result)

#for point in search_result:
#    print(point.id, point.score, point.payload)
