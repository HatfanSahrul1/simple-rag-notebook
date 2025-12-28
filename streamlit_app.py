import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from openai import OpenAI

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

st.set_page_config(page_title="RAG Agent - Barca & Indofood", page_icon="🤖")

@st.cache_resource
def load_rag_system():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    barca_vdb = FAISS.load_local("barca_vdb", embeddings, allow_dangerous_deserialization=True)
    barca_retriever = barca_vdb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    indofood_vdb = FAISS.load_local("indofood_vdb", embeddings, allow_dangerous_deserialization=True)
    indofood_retriever = indofood_vdb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    prompt_template = """
Kamu adalah asisten pintar yang membantu menjawab pertanyaan berdasarkan informasi yang tersedia.
Gunakan konteks berikut untuk menjawab.
- Jawablah secara ringkas, dalam bentuk poin-poin yang mudah dipahami.
- Jangan gunakan pengetahuan umum / data pre-train, hanya jawab berdasarkan konteks.
- jika pertanyaan tidak ada hubungannya dengan konteks, katakan "Maaf, saya tidak tahu."
- jika pertanyaannya soal FC Barcelona, jawab berdasarkan data FC Barcelona yang bukan women, kecuali pertanyaannya menyertakan kata "women".
- jawab dengan bahasa Indonesia.

Pertanyaan: {question}
Konteks:
{context}

Jawaban:
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    barca_rag_chain = (
        {"context": barca_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    indofood_rag_chain = (
        {"context": indofood_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return barca_rag_chain, indofood_rag_chain

barca_rag_chain, indofood_rag_chain = load_rag_system()

def ask_barca(question: str):
    answer = barca_rag_chain.invoke(question)
    return answer

def ask_indofood(question: str):
    answer = indofood_rag_chain.invoke(question)
    return answer

known_actions = {
    "ask_barca": ask_barca,
    "ask_indofood": ask_indofood
}

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({'role': 'system', 'content': self.system})

    def execute(self):
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content

    def run(self, query):
        self.messages.append({'role': 'user', 'content': query})
        
        while True:
            result = self.execute()
            self.messages.append({'role': 'assistant', 'content': result})
            
            if "Answer:" in result:
                self.final_answer = result.split("Answer:")[-1].strip()
                return
            
            action_match = re.search(r"Action: (\w+): (.*)", result)
            
            if action_match:
                action_name = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                
                if action_name in known_actions:
                    action_function = known_actions[action_name]
                    observation = action_function(action_input)
                    observation_message = f"Observation: {observation}"
                    self.messages.append({'role': 'user', 'content': observation_message})
                else:
                    return
            else:
                return
            
    def get_final_answer(self):
        return self.final_answer

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

ask_barca:
e.g. ask_barca: who's barca captain?
Use this to look up factual information about FC Barcelona, if the question isn't using english, translate it to english before use this action

ask_indofood:
e.g. ask_indofood: apa produk unggulan indofood?
Use this to get the spesific info about indofood
""".strip()

st.title("🤖 RAG Agent - Barca & Indofood")
st.markdown("Ask anything about **FC Barcelona** or **Indofood**!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Tanya sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = Agent(prompt)
            agent.run(user_input)
            answer = agent.get_final_answer()
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
