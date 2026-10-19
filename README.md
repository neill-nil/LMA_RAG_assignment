# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

A comparison of two retrieval pipelines for Retrieval-Augmented Generation (RAG) over one long, rule-heavy PDF: the **IFAB Laws of the Game 2026/27** (football/soccer rulebook, 260 pages).

Everything is in a single notebook: [`rag_reranking.ipynb`](rag_reranking.ipynb).

## Pipelines

Both pipelines use the same chunked corpus, the same LLM and the same prompt. The only difference is which 3 chunks reach the LLM.

```
                         ┌──────────────── Pipeline 1 (Baseline) ────────────────┐
query ─► Bi-encoder ─► FAISS top 10 ─► top 3 ───────────────────────► LLM ─► Answer 1
                              │
                              └─► Cross-encoder re-scores all 10 ─► top 3 ─► LLM ─► Answer 2
                         └────────────── Pipeline 2 (Re-ranking) ────────────────┘
```

| Component | Choice |
|---|---|
| PDF parsing | PyMuPDF (block-level text, footers/page numbers removed, sections tagged from the table of contents) |
| Chunking | ~160 words per chunk, ~40-word overlap, never crossing a section boundary (~300 chunks) |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS `IndexFlatIP` on L2-normalised vectors (= cosine similarity) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini (`gemini-3.5-flash-lite`), temperature 0 |

The LLM is told to answer **only** from the retrieved passages, to cite them, and otherwise to reply *"Not found in the provided context."* Without that rule, the model's own football knowledge would hide retrieval failures. Asked with no context, it gives the outdated goalkeeper "6-second" rule.

## Why this document

- It's long (260 pages), so many chunks look alike. That's where bi-encoders and cross-encoders start to differ.
- It's full of exceptions ("except when…"), negations ("may not", "cannot") and overlapping vocabulary (*goal kick* vs. *kick-off*, *penalty kick* vs. *penalty shoot-out*).
- The 2026/27 edition adds new rules the LLM is unlikely to know, so the answers have to come from the document.

## What the notebook contains

1. **Setup**: config and model names
2. **PDF download**: fetched from the IFAB website (not committed, see below)
3. **Parsing**
4. **Chunking**, with chunk statistics
5. **Pipeline 1**: bi-encoder embeddings and FAISS index
6. **Pipeline 2**: cross-encoder re-ranker
7. **LLM**: grounded prompt
8. **`ask(query)`**: runs both pipelines and shows a rank table (bi-encoder rank and score vs. cross-encoder rank and score) plus Answer 1 vs. Answer 2
9. **Failure-mode queries**: keyword traps, exceptions, negations and buried chunks, with a summary table of where the correct chunk ranks before and after re-ranking
10. **Latency profiling**: bi-encoder retrieval vs. re-ranking vs. LLM call, plus the cost of running the cross-encoder over the whole corpus
11. **Try your own question**

## Repository structure

```
.
├── rag_reranking.ipynb   # the whole project (executed, with outputs)
├── 2025202027_report.pdf # assignment report
├── requirements.txt
├── .gitignore
└── README.md
```

These files are created locally and are not committed:

| File | Purpose |
|---|---|
| `data/LOTG_2026-27_single.pdf` | Downloaded by the notebook on first run |
| `.env` | Your Gemini API key |
| `llm_cache.json` | Cached LLM answers, so re-runs don't spend API quota |

## Setup

Requires Python 3.10+. Everything runs on CPU; no GPU or training is needed.

```bash
git clone https://github.com/neill-nil/LMA_RAG_assignment.git
cd LMA_RAG_assignment

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# CPU-only PyTorch keeps the install small
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
```

## Running

Open `rag_reranking.ipynb` in Jupyter or VS Code, select the `.venv` kernel and run all cells.

- The first run downloads the PDF (~23 MB) and the two MiniLM models (~90 MB each).
- Parsing, embedding and re-ranking take seconds on CPU.
- Most of the run time is LLM calls, which are spaced out to respect free-tier rate limits.

To ask your own question, use the last cell:

```python
_ = ask("Can a player be offside if they receive the ball directly from a throw-in?")
```

## Notes

- **Free-tier limits:** the free tier allows ~5 requests per minute and a small daily quota per model. The notebook spaces out calls, retries on rate-limit and overload errors (`429`/`503`), and caches every answer in `llm_cache.json`. To switch models, change `LLM_MODEL` in the setup cell.
- **Copyright:** the Laws of the Game booklet may not be reproduced, so the PDF is downloaded from [theifab.com](https://www.theifab.com/laws-of-the-game-documents/) instead of being stored in this repo.
- **Deleted text in the PDF:** the "Law changes 2026/27" section shows deleted wording with a line through it. Text extraction can't detect this, so deleted rules end up in the corpus as normal text. The notebook includes a query that exposes this.
