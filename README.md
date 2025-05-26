---
title: EU Explorer (MDA Assignment)   
emoji: 🤖                         
colorFrom: purple                 
colorTo: indigo                   
sdk: docker                       
app_port: 4444                    
pinned: false
---

# Interactive Retrieval-Augmented Generation for Semantic Exploration of Horizon Europe Research Data

**A Cloud-Native Web Application for Multilingual Question Answering and Research Trend Analysis**

This project presents a scalable, cloud-native system that leverages Retrieval-Augmented Generation (RAG) to provide semantic access to the Horizon Europe research project database (CORDIS). Combining dense and sparse retrieval methods with advanced multilingual language models, the system enables users to ask natural language questions and receive document-grounded answers, complete with citations.

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
- [Retrieval-Augmented Generation Pipeline](#retrieval-augmented-generation-pipeline)
- [Limitations and Future Work](#limitations-and-future-work)

