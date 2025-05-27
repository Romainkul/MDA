---
title: EU Explorer (MDA Assignment)   
emoji: 🤖                         
colorFrom: purple                 
colorTo: indigo                   
sdk: docker                       
app_port: 4444                    
pinned: false
---
Hugginface spaces setup

# Interactive Retrieval-Augmented Generation for Semantic Exploration of Horizon Europe Research Data

**A Web Application for Question Answering and Research Trend Analysis**

This project presents a scalable system that leverages Retrieval-Augmented Generation (RAG) to provide semantic access to the Horizon Europe research project database (CORDIS). Combining dense and sparse retrieval methods with advanced multilingual language models, the system enables users to ask natural language questions and receive document-grounded answers, complete with citations.

The backend, built using FastAPI and integrated with tools like FAISS, Whoosh, and LangChain, supports both semantic and keyword search, hybrid retrieval, and re-ranking. A user-facing web application and chatbot interface make the system interactive and intuitive, allowing researchers, policymakers, and the public to explore EU-funded research projects in an intelligent, multilingual, and conversational manner.

## Table of Contents

- [Overview](#overview)
- [Dataset: Horizon Europe Projects](#dataset-horizon-europe-projects)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Application](#web-application)
  - [API Endpoints](#api-endpoints)
- [Predictive Modelling](#predictive-modelling)
- [Retrieval-Augmented Generation Pipeline](#retrieval-augmented-generation-pipeline)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

This repository contains an application for semantic exploration and trend analysis of the Horizon Europe research dataset. It enables:

- Multilingual question answering over the CORDIS database.
- Research trend analysis.
- Document-grounded answers with citations.
- Both semantic and keyword search.

## Dataset: Horizon Europe Projects

The system is built around data from the Horizon Europe research program (CORDIS), including metadata and deliverables for EU-funded projects. Data processing scripts and notebooks are provided for cleaning and transforming the CSV datasets into efficient formats (e.g., parquet).

## Features

- **Retrieval-Augmented Generation (RAG):** Combines dense and sparse retrieval for robust search.
- **Multilingual Support:** Uses advanced language models for question answering in multiple languages.
- **Hybrid Search:** Supports semantic (vector-based) and keyword (Whoosh) retrieval, including hybrid and re-ranking.
- **Web Interface & Chatbot:** Intuitive UI for interactive exploration.
- **API Access:** RESTful endpoints for programmatic access.

## System Architecture

- **Backend:** FastAPI-based, integrating FAISS (vector search), Whoosh (keyword search), and LangChain (RAG pipeline).
- **Frontend:** Web app and chatbot for user queries and result visualization.
- **Data Pipeline:** Notebooks and scripts for ingesting, cleaning, and transforming CORDIS data.

## Technologies Used

- FastAPI
- FAISS
- Whoosh
- LangChain
- Polars
- Python 3.10+
- Docker, Cloud deployment tools

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Romainkul/MDA.git
    cd MDA
    ```

2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Prepare datasets:
    - Place Horizon Europe (CORDIS) CSV files in the appropriate data directory.
    - Run provided Jupyter notebooks/scripts (e.g., `DataExploration.ipynb`) to clean and convert data.

5. Start the backend API:
    ```bash
    cd backend
    uvicorn main.app --host ::1 --reload
    ```

6. Run the frontend:
    ```bash
    cd frontend
    npm run dev
    ```

7. Alternatively it can be launched as a Docker Image:
```bash
docker build -t mda_eu_project:latest .
```

## Usage

### Web Application

- Start the backend API as above.
- Access the web UI via your browser at `http://localhost:8000`.
- With Docker, the 8000 becomes 4444/api due to the reverse proxy.

### API Endpoints

- Documentation is available at `http://localhost:8000/docs` (FastAPI Swagger UI).
- Example endpoints include:
    - `/api/rag`
    - `/api/projects`
    - `/api/filters`
    - `/api/project/id/organizations`
    - `/api/stats`
 
## Predictive Modelling

This script provides an end-to-end pipeline for status prediction in the MDA project. It features:

- **Data Preparation**: Cleans and engineers features, including handling multi-label and text fields.
- **Text Embedding**: Uses Sentence Transformers with SVD for dimensionality reduction.
- **ML Pipeline**: Builds a scikit-learn pipeline with preprocessing, anomaly detection, resampling, feature selection, and model calibration.
- **Model Training & Tuning**: Supports Optuna-based hyperparameter optimization.
- **Evaluation & Explanation**: Outputs classification metrics, SHAP explanations, and monitors data drift using Evidently.
- **Scoring**: Loads saved models to predict and explain results on new data.
  
Run the script to train the model, evaluate it, save artifacts, and score incoming data.

## Retrieval-Augmented Generation Pipeline

- **Data Ingestion:** Clean and preprocess CORDIS project and deliverable datasets.
- **Indexing:** Build FAISS (dense) and Whoosh (sparse) indexes.
- **Hybrid Retrieval:** Combine results from both indexes, optionally re-rank.
- **Generation:** Use a multilingual language model to generate grounded answers with citations.

## Limitations and Future Work

- Current language model and retrieval performance may be improved.
- Improve predictive modelling
- UI/UX enhancements planned.
- Additional analytics and trend visualizations under development.
- Support for more languages and larger datasets.

---

## Acknowledgements

- European Union Open Data Portal (CORDIS)
- Open-source contributors and projects (FastAPI, FAISS, Whoosh, LangChain, Polars)
- Course and teachers of Modern Data Analytics which/who made this project possible
