<div align="center">
  <h1>🧠 AI Consciousness Research Assistant</h1>
  <p>
    RAG-based AI assistant that helps explore, filter, and chat with scholarly papers on consciousness from arXiv.org.
    <br />
    <a href="#getting-started"><strong>Get Started »</strong></a>
    <br /><br />
    <a href="https://github.com/sidsharmaa/ai-consciousness-project-major">View Demo</a>
    ·
    <a href="https://github.com/sidsharmaa/ai-consciousness-project-major/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/sidsharmaa/ai-consciousness-project-major/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## 📄 About The Project

This project is an end-to-end agentic pipeline for exploring **AI consciousness** in academic literature. It scrapes arXiv papers on AI, filters them using NLP, chunks and embeds them using Sentence Transformers, and enables local Q\&A via Mistral (Ollama) with source citations.

### Key Features

* ✍️ Automated ArXiv paper scraping and keyword filtering
* 🔬 LLM-based embedding with FAISS index
* 🫠 Local chatbot powered by Ollama (Mistral)
* 🔹 Source-aware answers from academic papers
* 📊 Dashboard-ready structure for future visualization

### Built With

* Python 3.11+
* LangChain 0.2+
* SentenceTransformers
* FAISS
* Ollama (for local LLMs)
* arXiv API
* pandas, tqdm, matplotlib

---

## ⚙️ Getting Started

### Prerequisites

* Python >= 3.10
* Ollama installed: [https://ollama.com](https://ollama.com)
* Create `.env` from template:

  ```bash
  cp .env.example .env
  ```

### Installation

```bash
git clone https://github.com/sidsharmaa/ai-consciousness-project-major.git
cd ai-consciousness-project-major
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Fetch & Filter Papers

```bash
python scripts/pipeline.py
```

### 2. Embed for RAG

```bash
python rag/load_and_embed.py
```

### 3. Start Ollama (in a new terminal)

```bash
ollama run mistral
```

### 4. Ask Questions

```bash
python rag/query_bot.py
```

---

## 🚧 Roadmap

* [x] Fetch papers from arXiv
* [x] Filter relevant papers
* [x] Embed using Sentence Transformers
* [x] Query chatbot via Mistral
* [ ] Streamlit dashboard for filtering logic + stats
* [ ] Evaluate answer quality + chunking strategies
* [ ] Add David Chalmers TED Talk transcript to sources
* [ ] Guardrails + prompt engineering for AI safety

---

## 💪 Contributing

We welcome PRs and suggestions! See `CONTRIBUTING.md` (coming soon).

1. Fork it
2. Create a branch: `feature/my-feature`
3. Commit your changes
4. Push and create a PR

---


## 📢 Contact

**Siddhant Sharma** — [siddhantsharma4520@gmail.com](mailto:siddhantsharma4520@gmail.com)
GitHub: [@sidsharmaa](https://github.com/sidsharmaa)

Project Link: [https://github.com/sidsharmaa/ai-consciousness-project-major](https://github.com/sidsharmaa/ai-consciousness-project-major)

---

## 📖 Resources used

* LangChain & SentenceTransformers
* Ollama team for lightweight LLMs
* David Chalmers' research & talks
* [arXiv API](https://arxiv.org/help/api/)

<p align="right">(<a href="#top">back to top</a>)</p>
