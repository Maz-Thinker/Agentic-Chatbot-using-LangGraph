# 🤖 Advanced Agentic Chatbot with LangGraph & Streamlit

An advanced, enterprise-grade **Agentic Chatbot** built using **LangGraph**, **LangChain**, and **OpenRouter (GPT-4o-mini)**. The assistant features multi-tool orchestration, Retrieval-Augmented Generation (RAG), persistent chat memory backed by **MongoDB**, Human-in-the-Loop (HITL) approval workflows, a modern **Streamlit UI**, containerization via **Docker**, and an automated **CI/CD pipeline** targeting **AWS EC2** with **Docker Hub** integration.

---

## 🌟 Key Highlights

- **🧠 Advanced Agentic State Graph**: Powered by LangGraph `StateGraph`, enabling dynamic tool condition routing, multi-turn reasoning, and message streaming.
- **🖥️ Streamlit Web Interface**: Features live streaming text generation, thread switching sidebar, document drag-and-drop, and real-time HITL authorization cards.
- **💾 Memory Persistence**: Uses `MongoDBSaver` checkpointer to save full state history across distinct conversation threads in MongoDB.
- **🛡️ Human-in-the-Loop (HITL)**: Uses LangGraph `interrupt()` and `Command(resume=...)` to pause execution for sensitive tool calls (e.g., executing transactions) until explicit user authorization (Approve/Reject).
- **🐳 Dockerized Architecture**: Containerized app using a lightweight `python:3.11-slim` base image configured for FAISS CPU execution.
- **🚀 CI/CD & AWS EC2 Deployment**: Automated GitHub Actions workflow for linting/testing, building and pushing images to **Docker Hub**, and deploying to an **AWS EC2** instance via a self-hosted runner.

---

## 🛠️ Integrated Tools List

| Tool Name | Class / Function | Description | HITL Enabled |
| :--- | :--- | :--- | :---: |
| **RAG Tool** | `rag_tool` | Ingests uploaded PDFs using `PyPDFLoader` & `RecursiveCharacterTextSplitter`, embeds text via `FastEmbed` (`bge-small-en-v1.5`), and retrieves relevant context with local `FAISS` vector store. | ❌ |
| **Web Search Tool** | `search_tool` | Performs web searches for real-time information and current events via `TavilySearch` API (advanced depth). | ❌ |
| **Calculator Tool** | `calculator` | Evaluates mathematical expressions safely using constrained `eval` scope (`math`, `abs`, `round`, `min`, `max`, `sum`). | ❌ |
| **Stock Price Tool** | `get_stock_price` | Fetches real-time stock market metrics for ticker symbols (e.g. `AAPL`, `TSLA`) via Alpha Vantage API. | ❌ |
| **Weather Tool** | `get_current_weather` | Resolves city coordinates via OpenWeather Geocoding and retrieves live weather parameters (temp, humidity, wind, pressure). | ❌ |
| **Stock Purchase Tool** | `purchase_stock` | Simulates stock transaction placement. Triggers a **Human-in-the-Loop interrupt** to request user approval before execution. | ✅ **Yes** |

---

## 📐 System Architecture

```mermaid
flowgraph TD
    User([👤 User / Streamlit UI]) <--> StreamlitApp[🖥️ Streamlit App app.py]
    StreamlitApp <--> LangGraphEngine[⚙️ LangGraph StateGraph backend.py]
    
    subgraph State Machine & Tools
        LangGraphEngine --> ChatNode[🧠 Chat Node / LLM Router]
        ChatNode -- Tool Call Required --> ToolsNode[🔧 Tool Node]
        
        ToolsNode --> RAG[📄 RAG Tool FAISS]
        ToolsNode --> Search[🌐 Tavily Search]
        ToolsNode --> Calc[🧮 Calculator]
        ToolsNode --> Weather[☀️ OpenWeather API]
        ToolsNode --> StockPrice[📈 Alpha Vantage API]
        
        ToolsNode -- HITL Interrupt --> HITLCard[🔐 Human Approval Request]
        HITLCard -- Approved / Rejected --> ResumeCmd[⚡ Command resume]
        ResumeCmd --> ChatNode
        
        ToolsNode --> ChatNode
    end
    
    LangGraphEngine <--> MongoDBSaver[(💾 MongoDB Checkpointer)]
```

---

## 🔐 Human-in-the-Loop (HITL) Workflow

When the agent attempts to execute sensitive actions—such as `purchase_stock`:
1. LangGraph triggers `interrupt()` which halts execution and persists current state in MongoDB.
2. Streamlit detects the pending interrupt (`get_pending_interrupt()`) and presents an interactive **Approval Request** banner with **Approve (✅)** and **Reject (❌)** buttons.
3. Upon user interaction, Streamlit sends `Command(resume="yes")` or `Command(resume="no")`, resuming the execution graph smoothly.

---

## 🚢 CI/CD Pipeline & AWS EC2 Deployment

The project includes an automated deployment pipeline configured under `.github/workflows/cicd.yaml`:

```mermaid
flowgraph LR
    Push[📌 Git Push main] --> CI[1. Continuous Integration Lint & Tests]
    CI --> DockerBuild[2. Build & Push Docker Image Docker Hub]
    DockerBuild --> CD[3. Continuous Deployment AWS EC2 Runner]
    CD --> HealthCheck[4. Health Check http://localhost:8501/_stcore/health]
```

### Pipeline Overview:
1. **Continuous Integration**: Validates workspace codebase.
2. **Build & Push Docker Registry**: Logs into **Docker Hub**, builds multi-arch container image with Docker Buildx, and tags images with `:main` and `:sha-<git_commit>`.
3. **Continuous Deployment to AWS EC2**: Executes on an AWS EC2 **Self-Hosted GitHub Runner**:
   - Pulls the latest container image from Docker Hub.
   - Gracefully stops and removes existing container (`stapp`).
   - Runs new container with application secret injections (`OPENAI_API_KEY`, `OPENWEATHER_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_*`, etc.).
   - Executes readiness checks via Streamlit core health endpoint (`/_stcore/health`).
   - Prunes dangling Docker images automatically.

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# OpenRouter / OpenAI API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key

# Tools API Keys
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

# Memory Persistence
MONGODB_URI=mongodb://localhost:27017/ or mongodb+srv://<user>:<password>@cluster.mongodb.net/

# Optional: LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=agentic-chatbot
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.11+
- MongoDB instance (local service or MongoDB Atlas)

### 2. Installation
```bash
# Clone repository
git clone https://github.com/Maz-Thinker/Agentic-Chatbot-using-LangGraph.git
cd Agentic-Chatbot-using-LangGraph

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker

### Build Image
```bash
docker build -t agentic-chatbot:latest .
```

### Run Container
```bash
docker run -d \
  --name agentic-chatbot \
  -p 8501:8501 \
  --env-file .env \
  agentic-chatbot:latest
```

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── cicd.yaml           # GitHub Actions CI/CD workflow (Docker Hub -> AWS EC2)
├── .dockerignore
├── Dockerfile                  # Python 3.11-slim containerization recipe
├── app.py                      # Streamlit frontend UI & HITL handler
├── backend.py                  # LangGraph agent graph, MongoDB checkpointer & tools definitions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📜 License

Distributed under the MIT License.