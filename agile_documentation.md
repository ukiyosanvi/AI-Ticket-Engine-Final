Agile Project Documentation

1. Product Backlog
The following features were prioritized for the AI-Powered Knowledge Engine:

User Module: Secure Login/Signup and history tracking.

RAG Core: Integration of FAISS and Sentence-Transformers for document retrieval.

LLM Integration: Connecting Llama 3.2 via Ollama for resolution generation.

Notification System: Real-time Slack Webhook integration for support alerts.

Analytics Dashboard: KPI tracking (Confidence scores, ticket volume, and knowledge gaps).

2. User Stories
These stories guided the development of the features:

Support Agent: "As a support agent, I want the AI to suggest resolutions from our SOPs so that I can reduce my response time for common technical issues."

IT Admin: "As an admin, I want to see a Knowledge Gap Heatmap so I can identify which technical topics need better documentation."

System Developer: "As a developer, I want all sensitive credentials (like Slack URLs) to be stored in environment variables to ensure project security."

3. Sprint Cycles
We followed two 1-week sprints to achieve a Minimum Viable Product (MVP).

Sprint 1: The Foundation (AI & Data)
Goal: Establish the RAG pipeline.

Tasks: * Set up Flask backend.

Implemented PDF ingestion and FAISS vector indexing.

Verified Llama 3.2 local execution.

Sprint 2: The Interface & Integration
Goal: Build the user experience and alerting.

Tasks: * Developed the Responsive Admin Dashboard.

Implemented the Slack Alerting service.

Integrated confidence-based color coding (Red/Orange/Green alerts).

4. Scrum Ceremonies
Sprint Planning: Defined the scope of the knowledge engine and selected Llama 3.2 as the core LLM.

Daily Standups: Tracked progress on vector database integration and UI styling.

Sprint Review: Demonstrated the live Slack alert triggering during ticket submission.
