## 🏗️ Architecture

This project uses LangGraph to orchestrate the RAG workflow.

### Workflow

1. **AI Assistant** — Receives and analyzes the user query.
2. **Vector Retriever** — Retrieves relevant documents from the vector store.
3. **Document Grader** — Evaluates whether the retrieved documents are relevant.
4. **Query Rewriter** — Rewrites the query when retrieved documents are not relevant.
5. **Output Generator** — Generates the final response using the relevant context.

![LangGraph DAG](images/langgraph_dag.png)