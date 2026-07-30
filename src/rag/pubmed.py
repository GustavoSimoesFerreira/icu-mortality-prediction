"""Fetch abstracts from PubMed via NCBI E-utilities (open, no API key needed).

Two steps: esearch returns matching PubMed IDs, efetch returns the records as
XML which we parse into simple dicts.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, retmax: int, email: str) -> list[str]:
    """Return a list of PubMed IDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "email": email,
    }
    resp = requests.get(ESEARCH, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str], email: str) -> list[dict]:
    """Fetch and parse the abstract records for a list of PubMed IDs."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": email,
    }
    resp = requests.get(EFETCH, params=params, timeout=120)
    resp.raise_for_status()
    return parse_pubmed_xml(resp.text)


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML into dicts with pmid, title, abstract, journal, year.

    Abstracts with no text are skipped. Kept pure so it can be unit-tested
    without any network access.
    """
    root = ET.fromstring(xml_text)
    docs = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID", default="") or ""
        title = art.findtext(".//ArticleTitle", default="") or ""
        parts = [(e.text or "") for e in art.findall(".//Abstract/AbstractText")]
        abstract = " ".join(p for p in parts if p).strip()
        journal = art.findtext(".//Journal/Title", default="") or ""
        year = art.findtext(".//JournalIssue/PubDate/Year", default="") or ""
        if abstract:
            docs.append(
                {
                    "pmid": pmid,
                    "title": title.strip(),
                    "abstract": abstract,
                    "journal": journal.strip(),
                    "year": year,
                }
            )
    return docs
