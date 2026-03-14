# 🔍 Founder Research Agent

An autonomous, LangGraph-powered agent that researches a founder or CEO by browsing the web, following links, extracting facts, maintaining memory, and producing a structured Markdown + JSON report.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                      │
│                                                                  │
│   ┌──────┐   ┌────────┐   ┌────────┐   ┌─────────┐             │
│   │ plan │──▶│ search │──▶│ scrape │──▶│ analyse │             │
│   └──────┘   └────────┘   └────────┘   └────┬────┘             │
│                  ▲                           │                   │
│                  │         ┌─────────────────▼──────────┐       │
│                  │         │       memory_write          │       │
│                  │         │  (persist facts + summarise)│       │
│                  │         └─────────────┬───────────────┘       │
│                  │                       │                        │
│                  │         ┌─────────────▼──────────┐            │
│                  │         │  gaps found?            │            │
│                  └─────────│  → loop back to search  │            │
│                            │  else → finalise        │            │
│                            └─────────────┬───────────┘           │
│                                          │                        │
│                            ┌─────────────▼──────────┐            │
│                            │        finalise         │            │
│                            │  (build profile JSON)   │            │
│                            └─────────────┬───────────┘           │
│                                          │                        │
│                            ┌─────────────▼──────────┐            │
│                            │         report          │            │
│                            │  (render MD + JSON)     │            │
│                            └────────────────────────-┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Node responsibilities

| Node | What it does |
|---|---|
| **plan** | LLM generates research goals and 10-15 initial search queries |
| **search** | Executes 3 queries/iteration via Tavily (or DuckDuckGo fallback) |
| **scrape** | Fetches and cleans HTML from new URLs; skips LinkedIn bot-blocks |
| **analyse** | LLM extracts structured facts; generates follow-up queries for gaps |
| **memory_write** | Stores facts in `MemoryStore`; produces rolling LLM summary |
| **finalise** | Synthesises everything into a canonical JSON profile |
| **report** | Renders Markdown + saves JSON; prints paths |

---

## Quickstart

### 1. Clone and configure

```bash
git clone <this-repo>
cd founder-research-agent

cp .env.example .env
# Edit .env and add your API keys
```

### 2. Option A – Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py "Sam Altman" --context "CEO of OpenAI" --iterations 3 --print-report
```

### 3. Option B – Run with Docker

```bash
# Build image
docker build -t founder-research-agent .

# Research a founder (reports land in ./output/)
docker run --rm \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  founder-research-agent \
  "Elon Musk" --context "CEO of Tesla and SpaceX" --iterations 4
```

### 4. Option C – Docker Compose

```bash
# Edit docker-compose.yml to change the target name, then:
docker compose up --build

# Or override inline:
docker compose run research-agent "Jensen Huang" --context "CEO of NVIDIA" --iterations 3
```

---

## CLI Reference

```
python main.py <NAME> [OPTIONS]

Arguments:
  NAME                    Full name of the founder/CEO

Options:
  --context TEXT          Extra context, e.g. "CEO of OpenAI"
  --iterations INT        Max search/scrape loops (default: 4)
  --output-dir PATH       Where to save reports (default: output/)
  --log-level TEXT        DEBUG | INFO | WARNING (default: INFO)
  --print-report          Print Markdown report to stdout after completion
```

---

## Output

Each run produces two files in `output/`:

| File | Format | Contents |
|---|---|---|
| `<name>_<timestamp>.md` | Markdown | Human-readable report with all sections |
| `<name>_<timestamp>.json` | JSON | Full profile + all extracted facts + sources |

### Report sections

- **Summary** – 2–3 sentence bio
- **Education** – institutions, degrees, years
- **Career Timeline** – chronological roles
- **Companies** – founded/led, status, descriptions
- **Funding Highlights** – rounds, amounts, companies
- **Key Achievements**
- **Philosophy & Vision**
- **Notable Quotes** with sources
- **Controversies**
- **Recent News** with links
- **Sources** – all URLs consulted

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key |
| `TAVILY_API_KEY` | Optional | Tavily search API key (free tier available at [tavily.com](https://tavily.com)) |
| `OPENAI_MODEL` | Optional | Model name (default: `gpt-4o`) |
| `LOG_LEVEL` | Optional | Logging verbosity |

> **No Tavily key?** The agent automatically falls back to DuckDuckGo (no key required), though result quality may be slightly lower.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
founder-research-agent/
├── main.py                  # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── agent/
│   ├── graph.py             # LangGraph graph + routing logic
│   ├── nodes.py             # All node implementations
│   └── state.py             # Shared ResearchState TypedDict
│
├── tools/
│   ├── search.py            # Tavily + DuckDuckGo search
│   └── scraper.py           # HTML scraper with source-type detection
│
├── memory/
│   └── store.py             # In-memory + JSON-persistent fact store
│
├── tests/
│   └── test_agent.py        # Unit tests (pytest)
│
└── output/                  # Generated reports land here
```

---

## Design Decisions

**Why LangGraph?**
LangGraph models the research loop as an explicit state machine, making the flow auditable, interruptible, and easy to extend (e.g. adding a human-in-the-loop review step between `analyse` and `finalise`).

**Why a memory store instead of a vector DB?**
For a single-run research task, a lightweight key-value store with LLM summarisation avoids the operational complexity of spinning up ChromaDB or Pinecone. The `MemoryStore` can be swapped for a vector store if similarity search over thousands of facts is needed.

**Why Tavily + DuckDuckGo?**
Tavily provides structured, research-optimised search results. DuckDuckGo requires no API key and works as a zero-config fallback, so the agent is functional even without any paid keys.

**Scraper strategy**
Source-type detection lets us apply targeted CSS selectors for Wikipedia and news sites, significantly improving content quality. LinkedIn is explicitly skipped (bot-blocked) and the gap is covered by other sources.

---

## Extending the Agent

- **Add a LinkedIn scraper**: Use Proxycurl or Apify's LinkedIn API and add a `linkedin` branch in `scrape_node`.
- **Add vector memory**: Replace `MemoryStore` with a LangChain `Chroma` or `FAISS` retriever for semantic deduplication.
- **Add human-in-the-loop**: Insert a `HumanMessage` interrupt between `analyse` and `memory_write` using LangGraph's `interrupt_before`.
- **Stream progress**: Use `graph.astream_events()` for real-time progress updates in a UI.
- **Add Assignment 2**: Wire this agent's output into the Dev Workflow agent to research the engineering culture of a target company.
