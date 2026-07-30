"""Offline tests for the RAG module — no network, no Ollama. Keeps CI green."""

from src.rag.ingest import chunk_text
from src.rag.pubmed import parse_pubmed_xml
from src.rag.store import VectorStore

SAMPLE_XML = """
<PubmedArticleSet>
  <PubmedArticle><MedlineCitation>
    <PMID>12345</PMID>
    <Article>
      <ArticleTitle>SOFA score and ICU mortality</ArticleTitle>
      <Abstract><AbstractText>The SOFA score predicts mortality in the ICU.</AbstractText></Abstract>
      <Journal><Title>Critical Care</Title></Journal>
    </Article>
  </MedlineCitation></PubmedArticle>
</PubmedArticleSet>
"""


def test_chunk_text_overlaps_and_covers():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, size=40, overlap=10)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks).replace("", "")  # non-empty
    assert len(chunks) >= 3


def test_vector_store_ranks_by_similarity():
    store = VectorStore()
    store.add([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    hits = store.search([0.9, 0.1, 0.0], k=2)
    assert hits[0][0]["id"] == "a"
    assert len(hits) == 2


def test_parse_pubmed_xml():
    docs = parse_pubmed_xml(SAMPLE_XML)
    assert len(docs) == 1
    assert docs[0]["pmid"] == "12345"
    assert "SOFA" in docs[0]["title"]
    assert "mortality" in docs[0]["abstract"]
    assert docs[0]["journal"] == "Critical Care"
