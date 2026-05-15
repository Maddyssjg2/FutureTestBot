*   Project Name: Eigenbird (referred to as FutureTestBot in the user prompt, but the repository name is Eigenbird).
    *   Goal: Write a `README.md` based *only* on the provided source code.

    *   `config.py`: Handles environment variables, LLM provider selection (NVIDIA NIM, Google Gemini, LM Studio), and directory paths.
    *   `app.py`: The main FastAPI application. It defines several API endpoints/payloads:
        *   `ConnectorPayload`, `ConnectorTestPayload`: For managing connectors.
        *   `QueryPayload`: For user questions (modes: `auto`, `answer`, `execute`).
        *   `HumanInputPayload`, `FeedbackPayload`, `EvalPayload`: For human-in-the-loop and evaluation.
        *   Services imported: `ai_brain`, `evals`, `knowledge_base`, `codebase_search`, `request_understanding`, `symphony_memory`, `symphony_orchestrator`, `symphony_test_runner`, `symphony_workflow`, `query_engine`, `task_executor`, `code_agent`.
    *   `connectors/base.py`: Base class for all connectors.
    *   `connectors/jira.py`: Jira connector (actions: `create_issue`, `update_issue`, `add_comment`, `transition_issue`).
    *   `connectors/github.py`: GitHub connector (actions: `create_branch`, `upsert_file`, `open_pr`).
    *   `connectors/slack.py`: Slack connector (actions: `post_message`).
    *   `connectors/workspace.py`: Workspace connector (actions: `sync`, `read_repo`, `read_file`, `write_file`).

    *   It's an AI-driven agentic system (implied by `SymphonyOrchestrator`, `ai_brain`, `task_executor`, `code_agent`).
    *   It uses connectors to interact with external tools (GitHub, Jira, Slack, Workspace).
    *   It can answer questions, execute tasks, and manage workflows.
    *   It supports multiple LLM providers (NVIDIA NIM, Google Gemini).
    *   It includes features for codebase search, knowledge base management (RAG), and evaluation.

    *   *Title:* # Eigenbird
    *   *Overview:* Eigenbird is an AI-driven orchestration platform that uses specialized connectors to interact with development tools like GitHub, Jira, and Slack. It leverages advanced LLM providers to understand user requests, execute workspace tasks, and manage complex workflows through an agentic architecture.
    *   *Features:*
        *   Multi-connector support (GitHub, Jira, Slack, Local Workspace).
        *   Agentic task execution (Code agents, task planning, and workflow management).
        *   Knowledge Base & RAG (Retrieval-Augmented Generation) capabilities.
        *   Codebase search and analysis.
        *   Human-in-the-loop feedback and evaluation suites.
        *   Multi-LLM provider support (NVIDIA NIM, Google Gemini).
    *   *Tech Stack:*
        *   Python
        *   FastAPI
        *   Pydantic
        *   Dotenv
        *   Certifi
        *   (Implicitly: NVIDIA NIM, Google Gemini via API)
    *   *Project Structure:*
        *   `app.py`: Main FastAPI application and API definitions.
        *   `config.py`: Configuration and environment variable management.
        *   `connectors/`: Implementation of external tool integrations (GitHub, Jira, Slack, Workspace).
        *   `services/`: Core logic including AI brain, orchestrator, task executor, and knowledge base.
        *   `data/`: Local database and data storage.
        *   `frontend/`: Frontend application files.
    *   *Getting Started:*
        *   Install dependencies (implied by imports like `fastapi`, `pydantic`, `python-dotenv`).
        *   Configure environment variables in a `.env` file (based on `config.py`).
        *   Run the application (implied by `app.py` being a FastAPI app).

    *   Check `config.py` for specific env vars: `EIGENBIRD_LLM_PROVIDER`, `EIGENBIRD_NVIDIA_NIM_API_KEY`, `EIGENBIRD_GOOGLE_GEMINI_API_KEY`, `EIGENBIRD_HOST`, `EIGENBIRD_PORT`.
    *   Check `app.py` for specific service names: `SymphonyOrchestrator`, `WorkflowManager`, `TestRunner`, `CodeAgent`.

    *   *Title:* # Eigenbird
    *   *Overview:* Eigenbird is an agentic orchestration platform designed to automate tasks across development ecosystems. It utilizes a suite of connectors to interact with GitHub, Jira, Slack, and local workspaces, enabling AI-driven code execution, workflow management, and knowledge retrieval.
    *   *Features:*
        *   **Agentic Orchestration:** Uses a `SymphonyOrchestrator` and `WorkflowManager` to plan and execute complex user requests.
        *   **Multi-Tool Connectors:** Integrated support for GitHub (branching, file updates, PRs), Jira (issue management, comments, transitions), Slack (messaging), and local workspaces (file reading/writing).
        *   **AI-Powered Code Intelligence:** Includes a `CodeAgent` for code-related tasks and `codebase_search` for repository analysis.
        *   **Knowledge Management:** Features a RAG-based knowledge base with capabilities for rebuilding and exporting retraining datasets.
        *   **Evaluation Framework:** Built-in evaluation suites and human-in-the-loop feedback mechanisms to refine model performance.
        *   **Flexible LLM Integration:** Supports multiple providers including NVIDIA NIM and Google Gemini.
    *   *Tech Stack:*
        *   FastAPI
        *   Pydantic
        *   Python
        *   Dotenv
        *   Certifi
    *   *Project Structure:*
        *   `app.py`: Entry point containing the FastAPI application and API schemas.
        *   `config.py`: Environment configuration and LLM provider logic.
        *   `connectors/`: Directory containing connector implementations (`github.py`, `jira.py`, `slack.py`, `workspace.py`, `base.py`).
        *   `services/`: Core logic modules including `ai_brain`, `symphony_orchestrator`, `task_executor`, `code_agent`, and `knowledge_base`.
        *   `data/`: Directory for the SQLite database (`eigenbird.db`) and other data files.
        *   `frontend/`: Frontend source code and distribution files.
    *   *Getting Started:*
        1.  Clone the repository.
        2.  Create a `.env` file in the root directory.
        3.  Configure required environment variables such as `EIGENBIRD_LLM_PROVIDER`, `EIGENBIRD_NVIDIA_NIM_API_KEY` or `GOOGLE_GEMINI_API_KEY`, and connector credentials.
        4.  Install dependencies via pip.
        5.  Run the FastAPI server using a compatible ASGI server.

    *   No placeholders? Yes.
    *   No generic descriptions? Yes.
    *   Only what is in the code? Yes.
    *   Valid markdown? Yes.# Eigenbird

Eigenbird is an agentic orchestration platform designed to automate complex workflows across development ecosystems. It utilizes a suite of specialized connectors to interact with tools like GitHub, Jira, and Slack, enabling AI-driven task execution, codebase analysis, and knowledge management.

## Features

- **Agentic Orchestration**: Uses a `SymphonyOrchestrator` and `WorkflowManager` to plan and execute user requests through various modes (`auto`, `answer`, `execute`).
- **Multi-Tool Connectors**:
    - **GitHub**: Supports creating branches, upserting files, and opening pull requests.
    - **Jira**: Enables issue creation, updates, adding comments, and transitioning tickets.
    - **Slack**: Facilitates automated messaging via bot accounts.
    - **Workspace**: Provides direct local filesystem interaction, including reading repositories and writing files.
- **AI-Powered Code Intelligence**: Includes a `CodeAgent` for specialized coding tasks and a `codebase_search` service for repository exploration.
- **Knowledge Management**: Features a RAG-based (Retrieval-Augmented Generation) knowledge base with capabilities to rebuild the index and export retraining datasets.
- **Evaluation & Feedback**: Built-in evaluation suites and human-in-the-loop feedback mechanisms to refine agent performance and accuracy.
- **Flexible LLM Support**: Configurable to use multiple providers, including NVIDIA NIM and Google Gemini.

## Tech Stack

- **Python**
- **FastAPI** (Web Framework)
- **Pydantic** (Data Validation)
- **python-dotenv** (Environment Management)
- **Certifi** (SSL/TLS Verification)

## Project Structure

- `app.py`: The main FastAPI application containing API endpoints, request schemas, and service orchestration.
- `config.py`: Centralized configuration management, environment variable parsing, and LLM provider selection logic.
- `connectors/`: Implementation of tool integrations:
    - `base.py`: Abstract base class for all connectors.
    - `github.py`: GitHub integration logic.
    - `jira.py`: Jira integration logic.
    - `slack.py`: Slack integration logic.
    - `workspace.py`: Local filesystem/workspace integration.
- `services/`: Core business logic and agentic components:
    - `ai_brain.py`: Request planning.
    - `symphony_orchestrator.py`: Workflow orchestration.
    - `task_executor.py`: Execution of workspace and Jira tasks.
    - `code_agent.py`: Specialized code manipulation.
    - `knowledge_base.py`: RAG and data management.
    - `codebase_search.py`: Repository searching.
- `data/`: Local storage directory for the `eigenbird.db` database.
- `frontend/`: Frontend application source and distribution files.

## Getting Started

1. **Environment Setup**: Create a `.env` file in the project root.
2. **Configuration**: Define required environment variables in `.env`, including:
   - `EIGENBIRD_LLM_PROVIDER` (e.g., `nvidia_nim` or `google_gemini`)
   - `EIGENBIRD_NVIDIA_NIM_API_KEY` or `GOOGLE_GEMINI_API_KEY`
   - Connector credentials (e.g., `GITHUB_ACCESS_TOKEN`, `JIRA_OAUTH_ACCESS_TOKEN`)
3. **Installation**: Install the required Python dependencies.
4. **Execution**: Run the FastAPI application using an ASGI server.
