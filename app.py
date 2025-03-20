import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Charger les variables d’environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Vérifier que la clé API est bien récupérée
if not openai_api_key:
    st.error("❌ Erreur : La clé API OpenAI n'est pas définie. Vérifiez votre fichier .env.")
    st.stop()  # Arrête l'exécution si la clé est absente

# Interface Streamlit
st.title("🔍 Analyse de Rapports Juridiques avec LangChain")

# Chargement du fichier PDF
uploaded_file = st.file_uploader("📂 Téléchargez un rapport juridique (PDF)", type="pdf")

if uploaded_file is not None:
    st.write("📄 **Fichier chargé :**", uploaded_file.name)

    # Sauvegarder temporairement le fichier
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extraction du texte
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Vérifier si des pages ont été extraites
    if not pages:
        st.error("❌ Impossible de lire le fichier PDF. Vérifiez qu'il contient du texte lisible.")
        st.stop()

    # Diviser le texte en segments pour analyse
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)

    # Vérifier si du texte a été extrait
    if not texts:
        st.error("❌ Aucune donnée textuelle n'a pu être extraite.")
        st.stop()

    # Afficher un extrait du texte
    st.subheader("📜 Aperçu du document :")
    st.write(texts[0].page_content)  # Affiche un extrait

    # Embedding avec OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = InMemoryVectorStore.from_documents(texts, embeddings)

    # Système de recherche
    query = st.text_input("🔍 Posez une question sur le document 📖")
    if query:
        results = vector_store.similarity_search(query, k=3)
        
        if results:
            st.subheader("📌 Résultats les plus pertinents :")
            for i, res in enumerate(results):
                st.write(f"🔹 **Extrait {i+1}** : {res.page_content}")
        else:
            st.warning("⚠️ Aucun résultat trouvé pour cette requête.")
     # Questions financières par défaut
        st.subheader("Questions prédéfinies")
        default_questions = [
            "Quel est le chiffre d'affaires total de cette année ?",
            "Quels sont les segments les plus performants ?",
            "Quelle est l'évolution des bénéfices par rapport à l'année précédente ?",
            "Quels sont les principaux coûts opérationnels ?",
            "Quel est le ratio de solvabilité (CET1) ?",
        ]
        selected_question = st.selectbox("Choisissez une question prédéfinie :", default_questions)

        # Bouton pour les questions prédéfinies
        if st.button("Obtenir une réponse pour la question prédéfinie"):
            st.write("🔍 Recherche en cours pour la question prédéfinie...")
            docs = vector_store.similarity_search(selected_question, k=2)
            if docs:
                for doc in docs:
                    st.write(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}...')
            else:
                st.warning("⚠️ Aucune réponse trouvée dans le document.")

        # Option pour poser une question personnalisée
        st.subheader("Posez une question spécifique")
        user_question = st.text_input("Entrez votre propre question :", "")

        # Bouton pour les questions spécifiques
        if st.button("Obtenir une réponse pour la question spécifique"):
            if user_question.strip():
                st.write("🔍 Recherche en cours pour la question spécifique...")
                docs = vector_store.similarity_search(user_question, k=2)
                if docs:
                    for doc in docs:
                        st.write(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}...')
                else:
                    st.warning("⚠️ Aucune réponse trouvée dans le document.")
            else:
                st.warning("⚠️ Veuillez entrer une question spécifique avant de cliquer.")