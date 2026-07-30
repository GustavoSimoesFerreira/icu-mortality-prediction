# ICU Mortality Prediction — Life Sciences AI & Advanced Analytics Accelerator

> An end-to-end, **config-driven accelerator** for healthcare & life-sciences
> analytics, built around a **rare-disease** use case. It combines classic
> **predictive modeling** (patient risk, identification, segmentation,
> forecasting) with a **Generative-AI evidence layer** (RAG + semantic search)
> and an **agentic workflow** that ties them together into a stakeholder-ready
> output — all with fairness auditing and model governance baked in.
>
> ⚠️ **Research / portfolio use only.** Not a medical device, not clinical
> advice. Tabular modeling uses the **Open Access** PhysioNet Indwelling Arterial
> Catheter dataset (de-identified); other modules use synthetic data. No patient
> data is committed to this repo.

![Python](https://img.shields.io/badge/python-3.11-blue)
![SQL](https://img.shields.io/badge/SQL-DuckDB%20%7C%20Spark-orange)
![GenAI](https://img.shields.io/badge/GenAI-RAG%20%2B%20Agents-8A2BE2)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://img.shields.io/badge/CI-github--actions-blue)

---

## Why this project

Life-sciences analytics teams don't just build one model — they build
**reusable assets** that turn clinical, commercial, and real-world-evidence
(RWE) questions into scalable, production-ready solutions. This repo is designed
as exactly that: a modular **accelerator** where each capability is independent,
config-driven, and reusable across therapeutic areas.

The demonstration therapeutic area is a **rare disease** (swappable via config),
chosen because it showcases the industry-critical problem of **patient
identification** — finding likely under-diagnosed patients hidden in real-world
data.

## Table of Contents
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Roadmap (by priority)](#roadmap-by-priority)
- [Results](#results)
- [Responsible AI & Governance](#responsible-ai--governance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Live Demo](#live-demo)
- [Limitations](#limitations)
- [License](#license)

## Capabilities

Each module maps to real life-sciences analytics work:

1. **Patient Identification** — combine clinical rules with an ML classifier to
   flag likely under-diagnosed patients from RWD. *(clinical rules → scalable
   solution)*
2. **Patient Segmentation** — unsupervised clustering to build data-driven
   patient segments. *(segmentation frameworks)*
3. **Forecasting** — time-series forecast of diagnosed patient volume.
   *(forecasting solutions)*
4. **Predictive Risk Model** — gradient-boosted model predicting **28-day ICU
   mortality** on the Open Access PhysioNet Indwelling Arterial Catheter dataset,
   with SHAP explanations and a subgroup fairness audit. *(predictive modeling,
   ensemble methods, responsible AI)*
5. **Evidence Assistant (GenAI / RAG)** — retrieval-augmented Q&A over medical
   literature and drug labels, with **citations** and a **groundedness /
   faithfulness** evaluation. *(GenAI, LLMs, RAG, semantic search, model
   evaluation)*
6. **Agentic Workflow** — an agent that orchestrates the pipeline: run the risk
   model → retrieve supporting evidence → generate a structured summary for a
   stakeholder. *(Agentic AI, workflow automation, operationalization)*
7. **Data & Platform Layer** — SQL (DuckDB) + a PySpark pipeline, optionally run
   on Databricks. *(Python + SQL, distributed computing, data platforms)*
8. **Governance Layer** — model cards, data-drift monitoring, and a responsible-
   AI note. *(model evaluation, monitoring, and governance)*

## Architecture

```
                       ┌─────────────────────────┐
   Synthetic RWD  ─────▶  Data & Platform Layer   │  (SQL / PySpark, config-driven)
   (Synthea)           └───────────┬─────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
   Patient ID     Segmentation  Forecasting  Risk Model   (features shared)
        │              │           │           │
        └──────────────┴─────┬─────┴───────────┘
                             ▼
                    ┌──────────────────┐      ┌───────────────────────┐
                    │  Agentic Workflow │◀────▶│  Evidence Assistant   │
                    │  (LangGraph)      │      │  (RAG + Semantic Search│
                    └────────┬─────────┘       │  over PubMed/openFDA)  │
                             ▼                 └───────────────────────┘
                    Stakeholder Report  ──▶  FastAPI + Streamlit demo
```

## Data Sources

| Layer | Source | Access | Notes |
|-------|--------|--------|-------|
| Tabular RWD (primary) | **PhysioNet Indwelling Arterial Catheter** (`mimic2-iaccd`) | **Open** (ODC-By) | 1,776 ICU patients; 28-day mortality; no credentialing |
| Tabular benchmark (optional) | Diabetes 130-US Hospitals (UCI) | Open | Alternative predictive-modeling baseline |
| Synthetic RWD | **Synthea** | Open | For the patient-identification / segmentation modules |
| Credentialed RWD (optional, not default) | MIMIC-IV | PhysioNet + CITI | Only if you later add real clinical notes |
| Literature | **PubMed** abstracts (NCBI E-utilities) | Open API | RAG corpus |
| Drug labels | **openFDA** / DailyMed | Open API | RAG corpus |
| Trials | **ClinicalTrials.gov** API | Open API | RAG corpus (check current API version) |
| Imaging (optional) | MIMIC-CXR / CheXpert | Credentialed / Open | Only for clinical-imaging variant |

> No patient data is committed to this repo. The PhysioNet dataset is downloaded
> locally into gitignored `data/`; synthetic data is generated from a config.

## Tech Stack (100% free & local)

Every component runs on your own machine at **zero cost** — no paid API keys, no
cloud bills. Running the LLM locally is not just about cost: it also satisfies
PhysioNet's zero-data-retention policy for credentialed data (see
[Governance](#responsible-ai--governance)).

| Layer | Tool | Cost |
|-------|------|------|
| LLM (RAG + agent) | Ollama + an open model (Llama / Mistral / Qwen) | Free, local |
| Embeddings | sentence-transformers (Hugging Face) | Free, local |
| Vector store | FAISS or Chroma | Free, local |
| Agent orchestration | LangGraph / LlamaIndex | Free (OSS) |
| SQL | DuckDB | Free, local |
| Distributed processing | PySpark (local mode) | Free |
| Experiment tracking | MLflow (self-hosted) | Free, local |
| Drift monitoring | Evidently | Free (OSS) |
| Demo / UI | Streamlit / Gradio | Free (OSS) |
| Demo hosting (optional) | Hugging Face Spaces (CPU tier) | Free |
| Heavy compute (optional) | Google Colab / Kaggle notebooks | Free GPU tier |

> **Cost traps to avoid:** commercial LLM/embedding APIs (OpenAI, Anthropic)
> bill per call — use local models instead. Snowflake is a 30-day trial only;
> Databricks has a forever-free *Free Edition* but is optional (PySpark covers
> distributed computing locally).

## Roadmap (by priority)

Prioritized for the target role — you do **not** need every module. The top four
close the most job-specific gaps.

- [x] **P0 — Accelerator scaffold:** config-driven repo, CI, tests.
- [x] **P0 — Predictive core:** risk model (LogReg + XGBoost) + SHAP + fairness audit.
- [ ] **P1 — Evidence Assistant (RAG):** semantic search + cited answers +
      groundedness eval.
- [ ] **P1 — Agentic workflow:** orchestrate model + RAG + report generation.
- [ ] **P2 — Segmentation & Forecasting** modules.
- [ ] **P2 — Governance:** model cards, drift monitoring, responsible-AI doc.
- [ ] **P3 — Deployment:** FastAPI + Streamlit demo on Hugging Face Spaces.
- [ ] **P3 — (Optional) Databricks run** and/or **imaging** module.

## Results

> Fill in per module, on held-out, patient-level splits.

**Risk / Patient-ID model**

| Model | AUROC | AUPRC | Brier ↓ | Recall @ fixed FPR |
|-------|-------|-------|---------|--------------------|
| Logistic Regression | — | — | — | — |
| XGBoost | — | — | — | — |

**Evidence Assistant (RAG)**

| Metric | Score |
|--------|-------|
| Retrieval hit-rate @k | — |
| Faithfulness / groundedness | — |
| Answer relevance | — |

Include: PR & calibration curves, SHAP summary, subgroup fairness table, and
example RAG answers with citations.

## Responsible AI & Governance

- **Fairness audit** across age and sex subgroups (race is not available in this dataset).
- **Interpretability:** SHAP (tabular), source citations (RAG).
- **Leakage control:** patient-level splits; outcome/censoring columns excluded from features.
- **Governance:** model card per model, **data-drift** check (e.g. Evidently),
  and a short responsible-AI / evaluation policy doc.
- **Privacy:** de-identified Open Access data; synthetic elsewhere; no PHI in the repo.
- **Attribution (ODC-By):** cite Raffa, J. (2016), *Clinical data from the MIMIC-II
  database for a case study on indwelling arterial catheters*, PhysioNet.
- **Credentialed-data / LLM policy:** when a module touches credentialed data
  (e.g. MIMIC), PhysioNet's responsible-use policy applies — credentialed data
  must **not** be sent to third-party LLM APIs without a zero-data-retention
  guarantee. This project therefore routes any credentialed-data step through a
  **local LLM** (or a ZDR-compliant endpoint), while the open RAG corpus
  (PubMed, openFDA, ClinicalTrials.gov) has no such restriction. See PhysioNet's
  [responsible use of LLMs](https://physionet.org/news/post/llm-responsible-use/).

## Project Structure

```
rwe-patient-identification/
├── configs/             # therapeutic area, data, model configs (YAML)
├── data/                # (gitignored) generated/synthetic data
├── src/
│   ├── data/            # Synthea gen, SQL/Spark loading, splits
│   ├── patient_id/      # rules + ML classifier
│   ├── segmentation/    # clustering
│   ├── forecasting/     # time-series
│   ├── risk/            # predictive model + SHAP + fairness
│   ├── rag/             # ingestion, embeddings, retrieval, eval
│   ├── agent/           # LangGraph orchestration
│   └── governance/      # model cards, drift monitoring
├── app/                 # FastAPI + Streamlit demo
├── tests/               # pytest
├── .github/workflows/   # CI
├── requirements.txt
├── Dockerfile
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/rwe-patient-identification.git
cd rwe-patient-identification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download the Open Access PhysioNet dataset (see Data Sources) into data/raw/,
#    then load & explore it:
python -m src.data.load_diabetes --config configs/data.yaml

# 2. Train & evaluate the risk model (LogReg + XGBoost, SHAP, fairness):
python -m src.models.train --config configs/model.yaml

# Later-stage modules (roadmap):
#   python -m src.rag.ingest      (RAG evidence assistant)
#   python -m src.agent.run       (agentic workflow)
```

## Live Demo

Streamlit app: enter a synthetic patient, get a risk score with a SHAP
explanation, plus an evidence-backed summary generated by the agent.
**[Link once deployed]**

## Limitations

- The tabular cohort is small (~1,776 patients) and single-center ICU data —
  results illustrate methodology, not generalizable performance.
- Not validated for clinical use; **research/educational use only**.
- LLM outputs can be wrong; the RAG layer mitigates but does not eliminate this.

## License

MIT — see [LICENSE](LICENSE).