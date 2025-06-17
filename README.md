# Thai HR RAG Chatbot 🤖🇹🇭

A smart Q&A system for enterprise HR documents, built with Streamlit, LangChain, and Google's Gemini API. This chatbot allows users to ask questions in natural Thai language and get answers based on a private knowledge base.

![App Screenshot](https://placehold.co/800x450/4F46E5/FFFFFF?text=Thai+HR+Chatbot+UI)
_(Replace this with a real screenshot of your running application before sharing.)_

---

## ✨ Features

- **Natural Language Q&A:** Ask questions about company policies in plain Thai.
- **Context-Aware Answers:** Utilizes Retrieval-Augmented Generation (RAG) to provide answers strictly from the provided documents.
- **Handles Out-of-Scope Questions:** Gracefully responds when it doesn't have the information.
- **User-Friendly Interface:** Built with a clean and simple UI using Streamlit.
- **Deployable:** Ready to be deployed on Streamlit Community Cloud.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **AI/LLM Framework:** [LangChain](https://www.langchain.com/)
- **LLM Model:** [Google Gemini](https://ai.google.dev/)
- **Embeddings & Vector Store:** Google AI Embeddings & [ChromaDB](https://www.trychroma.com/)

---

## 🚀 Getting Started (Local Setup)

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.8+ (to run the application scripts)
- [Git](https://git-scm.com/) (to clone the project repository)
- A **Google Gemini API Key** (to access the LLM). You can get one from [Google AI for Developers](https://ai.google.dev/).

---

### 💻 Installation & Setup

Choose the instructions for your operating system.

<details>
<summary><strong>macOS / Linux</strong></summary>

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Praciller/thai-hr-rag-chatbot.git](https://github.com/Praciller/thai-hr-rag-chatbot.git)
    cd thai-hr-rag-chatbot
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**

    - Create a file named `.env` in the project's root directory.
    - Add your API key to the file like this:
      ```
      GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
      ```

5.  **Run the application:**

    ```bash
    streamlit run ui.py
    ```

    Your web browser should automatically open with the application running.

</details>

<details>
<summary><strong>Windows</strong></summary>

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Praciller/thai-hr-rag-chatbot.git](https://github.com/Praciller/thai-hr-rag-chatbot.git)
    cd thai-hr-rag-chatbot
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**

    - Create a file named `.env` in the project's root directory.
    - Add your API key to the file like this:
      ```
      GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
      ```

5.  **Run the application:**
    ```bash
    streamlit run ui.py
    ```
    Your web browser should automatically open with the application running.

</details>

---

## ☁️ Deployment

This application is designed to be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

1.  Push your code to a **public** GitHub repository.
2.  Sign in to [Streamlit Community Cloud](https://share.streamlit.io/signup) and choose 'Deploy from an existing repo'.
3.  **Important:** Remember to add your `GOOGLE_API_KEY` to the app's secrets in the advanced settings during deployment.
