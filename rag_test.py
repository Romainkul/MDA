import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import spacy
import re
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import matplotlib.pyplot as plt

# --- Load Models and Data ---
@st.cache_resource
def load_models_and_data():
    nlp = spacy.load("en_core_web_sm")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    projects_df = pd.read_parquet("cordis_projects.parquet")
    with open("topic_summaries.pkl", "rb") as f:
        topic_summaries_df = pickle.load(f)

    project_index = faiss.read_index("project_chunks.faiss")
    topic_index = faiss.read_index("topic_summary_index.faiss")

    return nlp, embedding_model, projects_df, topic_summaries_df, project_index, topic_index

nlp, embedding_model, projects_df, topic_summaries_df, project_index, topic_index = load_models_and_data()

# --- RAG Components ---

class SessionContext:
    def __init__(self):
        self.last_query_type = None
        self.last_entity = None
        self.last_projects_df = None
        self.last_project_id = None

    def update(self, query_type, entity, df=None, project_id=None):
        self.last_query_type = query_type
        self.last_entity = entity
        self.last_projects_df = df
        self.last_project_id = project_id

session_context = SessionContext()

def classify_query_type(query: str) -> str:
    query = query.lower()
    if re.search(r'\b(project|grant agreement|projectid|rcn|ga)\b', query):
        return "project"
    if re.search(r'\b(topic|eurovoc|euro-scivoc)\b', query):
        return "topic"
    if re.search(r'\b(organization|institution|company|beneficiary)\b', query):
        return "organization"
    if re.search(r'\b(legalbasis|legislation|h2020|fp7)\b', query):
        return "legalBasis"
    return "general"

def extract_entities_custom(query: str) -> dict:
    entities = {"project_id": None, "organization": None, "topic": None}
    doc = nlp(query)
    match = re.search(r"\b\d{6,8}\b", query)
    if match:
        entities["project_id"] = match.group(0)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["organization"] = ent.text
        if ent.label_ == "MISC":
            entities["topic"] = ent.text
    return entities

def retrieve_project_chunks_by_id(project_id: str) -> list:
    subset = projects_df[projects_df['project_id'] == project_id]
    return subset['chunk_text'].tolist()

def retrieve_topic_summary_from_project(project_id: str) -> str:
    row = projects_df[projects_df['project_id'] == project_id]
    if row.empty:
        return ""
    topic = row['topic_path'].values[0]
    match = topic_summaries_df[topic_summaries_df['topic'] == topic]
    return match['summary'].values[0] if not match.empty else ""

def get_kpi_context() -> str:
    return (
        "Average project duration: 780 days\n"
        "Termination rate: 12.3%\n"
        "Top countries by project count: Germany, France, Italy\n"
        "Common termination reasons: coordination failure, underperformance"
    )

def run_reasoning(question: str, df: pd.DataFrame) -> str:
    q = question.lower()
    if "average" in q and "funding" in q:
        avg = df['ecMaxContribution'].mean()
        return f"The average funding is €{avg:,.2f}."
    return ""

def generate_follow_up_suggestions(query_type: str) -> str:
    if query_type == "project":
        return "Would you like more info on related topics or organizations involved?"
    if query_type == "organization":
        return "Want to explore more projects from this organization?"
    if query_type == "topic":
        return "Would you like to see top projects under this topic?"
    return ""

def is_follow_up_question(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in [
        "they", "those", "what do they", "what are they about", "explain them", "what's the topic"
    ])

def summarize_projects(df: pd.DataFrame, field="objective", top_n=5) -> str:
    texts = df[field].dropna().tolist()[:top_n]
    if not texts:
        return "No objectives or summaries available for these projects."
    combined_text = "\n\n".join(texts)
    prompt = f"Summarize what these projects are generally about:\n\n{combined_text}"
    return OpenAI(temperature=0)(prompt)

rag_template = PromptTemplate(
    input_variables=["project_info", "topic_summary", "kpi_context", "programmatic", "question", "followup"],
    template="""
You are a research assistant for a European funding agency. Use the provided context to answer the user query.

--- Project Info ---
{project_info}

--- Topic Summary ---
{topic_summary}

--- KPI Context ---
{kpi_context}

--- Data Insights ---
{programmatic}

--- Question ---
{question}

--- Answer ---

Also consider:
{followup}
"""
)

def chat_with_context(question: str, session: SessionContext) -> str:
    if is_follow_up_question(question) and session.last_projects_df is not None:
        return summarize_projects(session.last_projects_df)

    query_type = classify_query_type(question)
    entities = extract_entities_custom(question)
    project_info = topic_summary = programmatic = ""
    relevant_df = projects_df.copy()
    project_id = None

    if entities["project_id"]:
        project_id = entities["project_id"]
        project_info = "\n".join(retrieve_project_chunks_by_id(project_id))
        topic_summary = retrieve_topic_summary_from_project(project_id)
        relevant_df = projects_df[projects_df['project_id'] == project_id]

    elif query_type == "organization":
        org = entities.get("organization")
        if org:
            relevant_df = projects_df[projects_df['list_name'].str.contains(org, case=False, na=False)]
            project_info = "\n".join(relevant_df['chunk_text'].head(3))

    programmatic = run_reasoning(question, relevant_df)
    kpi_context = get_kpi_context()
    followup = generate_follow_up_suggestions(query_type)

    prompt = rag_template.format(
        project_info=project_info,
        topic_summary=topic_summary,
        kpi_context=kpi_context,
        programmatic=programmatic,
        question=question,
        followup=followup
    )

    session.update(query_type, entities.get("organization") or entities.get("project_id"), relevant_df, project_id)

    return OpenAI(temperature=0)(prompt)

# --- Streamlit UI ---

st.set_page_config(page_title="EU Funding Explorer", layout="wide")
st.title("🇪🇺 EU Projects Dashboard")

tabs = st.tabs(["📊 Dashboard", "📁 Projects + Chatbot"])

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.subheader("Funding Overview")
    funding_by_year = projects_df.groupby("startYear")["ecMaxContribution"].sum().reset_index()
    plt.figure(figsize=(10,4))
    plt.bar(funding_by_year["startYear"], funding_by_year["ecMaxContribution"] / 1e6)
    plt.ylabel("Total Funding (€M)")
    st.pyplot(plt)

    top_orgs = projects_df["list_name"].value_counts().head(10)
    st.bar_chart(top_orgs)

# --- Tab 2: Projects + Chatbot ---
with tabs[1]:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Browse Projects")
        page_size = 10
        page_num = st.number_input("Page", min_value=0, max_value=(len(projects_df) // page_size), step=1)
        paginated = projects_df.iloc[page_num * page_size : (page_num + 1) * page_size]
        st.dataframe(paginated[["title", "list_name", "ecMaxContribution", "startYear", "status"]], use_container_width=True)

    with right:
        st.subheader("Ask the Chatbot")
        user_input = st.text_input("Ask a question...")
        if st.button("Ask") and user_input:
            answer = chat_with_context(user_input, session_context)
            st.markdown(answer)
