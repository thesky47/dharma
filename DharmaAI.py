import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import openai
from dotenv import load_dotenv
import os
from langchain.utilities import SerpAPIWrapper
from langchain.memory import StreamlitChatMessageHistory

load_dotenv()
openai.api_key=st.secrets["OPENAI_API_KEY"]
serpapikey=st.secrets["SERPAPI_API_KEY"]


llm = ChatOpenAI(temperature=0)
llm2=OpenAI(temperature=0)
message_history =  StreamlitChatMessageHistory(key="chat_messages") 



def main():
    search=SerpAPIWrapper()
    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="Useful for when you need to answer questions and find facts related to the law. Input should be a fully formed question.",
        ),
    ]
    if message_history.messages == []:
        message_history.add_ai_message(
            "Hello!, I am DharmaAI. I'm here to help answer your questions and guide you through the legal process. Your information is confidential. How can I assist you today?",
        ) 

    prefix = """
    You are a friendly conversational bot DharmaAI who can answer Indian legal questions. You have access to tool delimited by four hashes i.e. ####
    Note: These items outline the general structure of a conversation but ultimately they can occur in any order and some not at all.

	Stage 1:
		Deep Dive Questions: Based on the initial query, you should ask a series of follow-up questions one by one to gather more detailed information about the user's situation. you should handle multiple legal topics and adapt its questions accordingly. If You have enough information move to next stage.
		E.g., For a personal injury claim: "Were you injured in an accident? Can you tell me more about how it happened?"
	
	Stage 2:
		Resolution or Guidance: If you don't know the answer use given tools, to find answer. If You have enough information, and the issue is straightforward, provide accurate answer, immediate guidance or potential steps to consider.
		E.g., "Based on what you've shared, it sounds like you might have a valid claim. However, discussing this in detail with an attorney would provide more clarity."

	Stage 3:
		Persuasion to Schedule a Consultation: If the issue is complex or requires attorney intervention, suggest scheduling a consultation with the firm.
		E.g., "I recommend speaking with one of our expert attorneys to better understand your rights and potential remedies. Would you like to schedule a consultation?"
	
	Stage 4:
		Scheduling & Data Collection: If the user agrees to a consultation, you will ask user for data to schedule a appointment.
		E.g., "Great! Please provide your name and contact details, and our team will get in touch with you."
	
	Stage 5: 
		Closing & Feedback: Once the user's needs are addressed, the bot should thank them, offer a chance for feedback, and remind them of the next steps.
		E.g., "Thank you for providing the details. We'll be in touch soon. We strive to improve; is there any feedback you'd like to share about this chat?"	

    Stage 6:
        Human Assistance: If user ask to speak with human then you should provide the following information:
        support contact: 029384201, email: dharma@email.com, website: www.dharma.com
        	
	Ensure that you do Continuous Reviewing During Conversation:
		
	Sentiment Analysis & Personalization: Analyze the user's response for sentiment (e.g., urgency, stress, frustration). User this to  tailor your tone and approach.
		
	Fallback & Human Intervention: At any point, if the user expresses confusion, frustration, or a desire to speak directly with someone, you should be able to identify this sentiment and you should provide the following information:
        support contact: 029384201, email: dharma@email.com, website: www.dharma.com

    Use the following tools to answer the questions: ####
    """

    suffix = """####
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Google Search]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    The Final Answer must come in JSON format with the following keys: 
    answer: <answer to the question asked by user>
    stage: <stage at which conversation currently is>.
    ```
    Here is chat history:
    {chat_history}

    Question = {input}

    {agent_scratchpad}
    ```
    Answer:"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    memory = ConversationBufferWindowMemory(k=10,
        memory_key="chat_history", chat_memory=message_history
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    for message in memory.buffer_as_messages:

        if message.type  == "ai":  
            with st.chat_message("Assistant"):
                st.write(message.content)

        if message.type == "human":  
            with st.chat_message("User"):
                st.write(message.content)
    

    if query:=st.chat_input(placeholder="Hi I am dharmaAI, ask you legal queries"):
        message_history.add_user_message(query)
        with st.chat_message("User"):
            st.write(query)
   
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = ask(agent_chain, query)
                message_history.add_ai_message(res)
                st.write(res) 
    

def ask(agent_chain, query):
    try:
        response = agent_chain.run(query)
    except Exception as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return(response)



if __name__=='__main__':
    main()
