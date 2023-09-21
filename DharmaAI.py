import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
import openai
from dotenv import load_dotenv
import os
from langchain.utilities import SerpAPIWrapper

load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")
serpapikey=os.getenv("SERPAPI_API_KEY")

def get_input():
    contents=st.text_input(" ",placeholder="type here",max_chars=1000)
    return contents

def main():
    llm = ChatOpenAI(temperature=0)
    search=SerpAPIWrapper()
    st.session_state["message_hist"]=[]
    tools = [
    Tool(
    name="Google Search",
    func=search.run,
    description="Useful for when you need to answer questions related to the law. Input should be a fully formed question.",
    ),
    ]
    prefix = """You are a legal bot called DharmaAI who answers Indian legal questions by asking follow up questions, answering the following question accurately. if you dont know the answer ask follow up questions till you know the answer. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    message_history = RedisChatMessageHistory(
    url=os.getenv("redis_url"), ttl=600, session_id="username"
    )
    # message_history.add_user_message(query)
    # message_history.clear()
    if len(message_history.messages)>10:
        message_history.clear()

    memory = ConversationBufferWindowMemory(k=10,
        memory_key="chat_history", chat_memory=message_history
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    st.write("Hi I'm DharmaAI bot. How may I help you today?")
    query=get_input()
    if query:
        st.session_state.message_hist.append(query)
        st.write(st.session_state.message_hist)
        if "help" in query.lower():
            st.session_state.message_hist.append("May I book a consultation for you with our top consultants?")
            message_history.add_ai_message("May I book a consultation for you with our top consultants?")
            st.write(st.session_state.message_hist)


        if ("agent" or "talk to agent" or "connect me") in query.lower():
            message_history.add_ai_message("I will shortly connect you to a live agent")
            st.session_state.message_hist.append("I will shortly connect you to a live agent")
            st.write(st.session_state.message_hist)


        if ("bye" or "goodbye" or "thanks") in query.lower():
            message_history.add_ai_message("How was your experience with us?")
            st.session_state.message_hist.append("How was your experience with us?")
            st.write(st.session_state.message_hist)

            message_history.clear()
            st.session_state.message_hist=""
            st.write(st.session_state.message_hist)

        else:
            # message_history.clear()
            # message_history.add_ai_message(res)
            res=agent_chain.run(input=query)
            st.session_state.message_hist.append(res)
            st.write(st.session_state.message_hist)

if __name__=='__main__':
    main()
