import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted, PermissionDenied
import utils as Utils

load_dotenv()

class NewsChat:
    store = {}
    session_id = ''
    rag_chain = None

    def __init__(self, article_id: str):
        self.session_id = article_id

        g_key = os.getenv("GEMINI_API_KEY")
        if not g_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=g_key
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=g_key
        )

        db = Chroma(
            persist_directory=Utils.DB_FOLDER,
            embedding_function=embeddings,
            collection_name='indian_laws_collection'
        )
        retriever = db.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Given a chat history and the latest user question, "
                "formulate a standalone question understandable without chat history. "
                "Do NOT answer; just reformulate if needed."
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an assistant for question-answering tasks. "
                "Use the retrieved context to answer the question. "
                "If unknown, say so. Keep it concise (max 3 sentences). "
                "Retrieved context:\n{context}"
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        self.rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def ask(self, question: str) -> str:
        try:
            response = self.rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": self.session_id}},
            )["answer"]
        except ResourceExhausted:
            return "Quota exceeded for Gemini API. Please try again later."
        except PermissionDenied:
            return "API key invalid or insufficient permissions."
        except GoogleAPICallError as e:
            return f"Google API error: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        return response
