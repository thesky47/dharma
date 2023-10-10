import streamlit as st
import openai
from dotenv import load_dotenv
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
import prompts as pmt
import ast
load_dotenv()

def convert_string_to_tuple(string):
    """Converts a string to a tuple.

    Args:
      string: A string in the format "(2, 'Initial Query Collection')".

    Returns:
      A tuple.
    """
    return string.strip(' \'"').strip("()").split(",")


class LegalBot:
    def __init__(self, message_history):
        self.llm = ChatOpenAI(temperature=0)
        self.search = SerpAPIWrapper(serpapi_api_key=st.secrets["SERPAPI_API_KEY"])
        self.message_history = message_history
        self.tools = [
            Tool(
                name="Google Search",
                func=self.search.run,
                description="Useful for when you need to answer questions and find facts related to the law. Input should be a fully formed question.",
            ),
        ]

        self.memory = self.get_memory(chat_history=message_history)

    def get_memory(self, chat_history=None):
        if not chat_history:
            chat_history = self.message_history
        return ConversationBufferWindowMemory(
            k=10, memory_key="chat_history", chat_memory=chat_history
        )

    def get_current_stage(self, query):
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=pmt.prompt_template.format(
                chat_history=self.memory.buffer_as_str, human_input=query),
            temperature=0,
        )
        return response.choices[0].text

    def select_prompt(self, stage):
        # Select the correct prompt for the given stage.
        prompts = {
            "welcome_&_introduction": "answer in friend and welcoming way",
            "deep_dive_questions": pmt.deep_dive_questions,
            "resolution_or_guidance": pmt.resolution_or_guidance,
            "persuasion_to_schedule_consultation": pmt.persuasion_to_schedule_consultation,
            "scheduling_and_data_collection": pmt.scheduling_and_data_collection,
            "closing_and_feedback": pmt.scheduling_and_data_collection,
            "human_assistance": pmt.human_assistance
        }

        return prompts[stage]

    def ask(self, agent_chain, query):
        try:
            response = agent_chain.run(query)
        except Exception as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            response = response.removeprefix(
                "Could not parse LLM output: `").removesuffix("`")
        return (response)

    def generate_response(self, question):
        stage = self.get_current_stage(question)
        stage = convert_string_to_tuple(stage)
        instruction = self.select_prompt(stage[1].strip(' \'"'))
        prefix = pmt.prefix.format(prompt_by_stage=instruction)

        prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=prefix,
            suffix=pmt.suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],  
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True, memory=self.memory
        )

        return (self.ask(agent_chain, question), stage)
