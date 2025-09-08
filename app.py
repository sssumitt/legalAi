import os
import streamlit as st
import agent as Agent
import utils as Utils
import embed


def create_chat(session_id: str):
    chat_container = st.container()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["id"] == session_id:
            chat_container.chat_message(message["role"]).write(message["content"])

    if "newschat" not in st.session_state or st.session_state.newschat.session_id != session_id:
        st.session_state.newschat = Agent.NewsChat(session_id)

    user_input = st.chat_input(
        placeholder="Ask me about Indian legal topics (e.g., IT Act, AI Regulations)", key=session_id
    )

    if user_input:
        chat_container.chat_message("user").write(user_input)

        with st.spinner("Thinking..."):
            assistant_response = st.session_state.newschat.ask(user_input)
            chat_container.chat_message("assistant").write(assistant_response)

        st.session_state.messages.append({
            "id": session_id,
            "role": "user",
            "content": user_input
        })
        st.session_state.messages.append({
            "id": session_id,
            "role": "assistant",
            "content": assistant_response
        })


if __name__ == "__main__":
    if not os.path.exists(Utils.DB_FOLDER):
        document_name = "Indian IT Act 2000"
        document_description = "Information Technology Act 2000 - India"
        pdf_url = "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf"

        text = embed.pdf_to_text(pdf_url)
        if text:
            embed.embed_text_in_chromadb(text, document_name, document_description)
        else:
            print("Failed to extract text from PDF.")

    create_chat("chat1")
