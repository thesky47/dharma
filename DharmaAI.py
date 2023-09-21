import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
import openai
from dotenv import load_dotenv
import os
from langchain.utilities import SerpAPIWrapper
# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.chat_models import HumanInputChatModel
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")
serpapikey=os.getenv("SERPAPI_API_KEY")
llm = ChatOpenAI(temperature=0)
llm2=OpenAI(temperature=0)
message_history = RedisChatMessageHistory(
url=os.getenv("redis_url"), ttl=600, session_id="username"
)
# st.session_state["message_hist"]="Hi I'm DharmaAI bot. How may I help you today?"


# def get_input():
#     # if 'message_hist' not in st.session_state:
#     #     st.session_state.message_hist="Hi I'm DharmaAI bot. How may I help you today?"

#     # con=st.text_input(thought,placeholder="Type here",max_chars=1000)
#     # content=llm(con)
#     # return content
#     con=st.text_input(label="Could you explain the query in more detail?",placeholder="Type here",max_chars=1000)
#     contents=llm2(con)
#     # st.session_state.message_hist.append("Could you explain the query in more detail?",con,content)
#     message_history.add_ai_message("Could you explain the query in more detail?")
#     message_history.add_ai_message(contents)

#     message_history.add_user_message(con)

#     return contents

def main():
    search=SerpAPIWrapper()
    tools = [
    Tool(
    name="Google Search",
    func=search.run,
    description="Useful for when you need to answer questions related to the law. Input should be a fully formed question.",
    ),
    # Tool(name="human",func=get_input(),description="Useful when asked questions with little context")
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
    query=st.text_input(" ",placeholder="Type here",max_chars=1000)
    if query:
        # if 'message_hist' not in st.session_state:
        #     st.session_state.message_hist="Hi I'm DharmaAI bot. How may I help you today?"

        # st.session_state.message_hist.append(query)
        # st.write(st.session_state.message_hist)
        if "help" in query.lower():
            message_history.add_user_message(query)
            # st.session_state.message_hist.append("May I book a consultation for you with our top consultants?")
            message_history.add_ai_message("May I book a consultation for you with our top consultants?")
            st.write("May I book a consultation for you with our top consultants?")

        elif ("agent" or "talk to agent" or "connect me") in query.lower():
            message_history.add_user_message(query)
            message_history.add_ai_message("I will shortly connect you to a live agent")
            # st.session_state.message_hist.append("I will shortly connect you to a live agent")
            st.write("I will shortly connect you to a live agent")

        elif ("bye" or "goodbye" or "thanks") in query.lower():
            message_history.add_user_message(query)
            message_history.add_ai_message("How was your experience with us?")
            # st.session_state.message_hist.append("How was your experience with us?")
            st.write("How was your experience with us?")

            message_history.clear()
            # st.session_state.message_hist=""
            # st.write(st.session_state.message_hist)

        else:
            # message_history.clear()
            # message_history.add_ai_message(res)
            res=agent_chain.run(input=query)
            # st.session_state.message_hist.append(res)
            st.write(res)

if __name__=='__main__':
    main()
