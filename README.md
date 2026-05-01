# 🤖 AI Tutor – RAG Based Question Answering System

An intelligent AI Tutor system that answers user questions using **Semantic Search (Machine Learning)** and a **Local Language Model (TinyLlama)**.

This project demonstrates a **Retrieval-Augmented Generation (RAG)** approach, combining ML-based similarity search with LLM-generated responses.

---

## 🚀 Features

* 🧠 Understands questions using semantic embeddings
* 🔍 Retrieves most relevant answers from dataset
* ✨ Generates human-like explanations using a local LLM
* 📚 Learns new questions over time (incremental dataset)
* 💻 Works locally (no API required)

---

## 🏗️ System Architecture

```
User Question
      ↓
Sentence Transformer (Embeddings)
      ↓
Cosine Similarity Search
      ↓
Retrieve Best Match
      ↓
TinyLlama (LLM)
      ↓
Final Answer
```

---

## 🛠️ Tech Stack

* Python
* Flask
* Sentence Transformers (NLP / ML)
* TinyLlama (Local LLM via Ollama)
* JSON (Dataset storage)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-tutor-rag-system.git
cd ai-tutor-rag-system
```

---

### 2. Install dependencies

```bash
pip install sentence-transformers torch flask requests
```

---

### 3. Install Ollama

Download from: https://ollama.com/download

---

### 4. Run TinyLlama

```bash
ollama run tinyllama
```

---

### 5. Train the model (prepare dataset)

```bash
python modelTraining.py
```

---

### 6. Run the application

```bash
python main.py
```

---

## 🧪 Example Queries

* What is Artificial Intelligence?
* Explain photosynthesis
* What is gravity?
* What is football?

---

## 🧠 Machine Learning Concepts Used

* Semantic Embeddings
* Cosine Similarity
* Retrieval-Based Learning
* Natural Language Processing (NLP)

---

## ⚙️ How It Works

1. User enters a question
2. System converts it into embeddings
3. Finds most similar question using cosine similarity
4. Retrieves stored answer
5. TinyLlama enhances the response
6. If unknown → generates answer and stores it for learning

---

## ⚠️ Limitations

* Small dataset may reduce accuracy
* TinyLlama is lightweight (not as powerful as large models)
* Automatic learning may store incorrect answers

---

## 🔮 Future Improvements

* Add user feedback system (correct/incorrect answers)
* Improve dataset quality
* Use more advanced lightweight LLMs
* Add chat history & memory
* Build modern UI

---

## 📄 Resume Highlight

> Developed an AI Tutor using semantic search and local LLM (TinyLlama) implementing a Retrieval-Augmented Generation (RAG) architecture with incremental learning capability.

---

## 👨‍💻 Author

**Aryendra Singh**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
