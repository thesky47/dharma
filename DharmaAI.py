import streamlit as st
import openai
from langchain.memory import StreamlitChatMessageHistory
from bot import LegalBot
import time

openai.api_key=st.secrets["OPENAI_API_KEY"]
message_history =  StreamlitChatMessageHistory(key="chat_messages") 
legal_bot = LegalBot(message_history=message_history)


def main():  # sourcery skip: use-named-expression
    if message_history.messages == []:
        message_history.add_ai_message(
            "Hello!, I am DharmaAI. I'm here to help answer your questions and guide you through the legal process. Your information is confidential. How can I assist you today?",
        ) 

    for message in legal_bot.memory.buffer_as_messages:
        message_type = message.type
        message_content = message.content

        with st.chat_message("Assistant" if message_type == "ai" else "User"):
            st.write(message_content)
    
    query = st.chat_input(placeholder="Hi I am dharmaAI, ask you legal queries")
    if query:
        start_time = time.time()
        
        with st.chat_message("User"):
            st.write(query)
            message_history.add_user_message(query)
   
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = legal_bot.generate_response(question=query) #ask(agent_chain, query)
                res = response[0]
                end_time = time.time()
                message_history.add_ai_message(res)
                stage = response[1][1].strip(' \'"')    
                st.write(res) 
                st.write(f"Stage: {stage} \n Execution time: {(end_time - start_time):.3f} seconds")


if __name__=='__main__':
    main()
