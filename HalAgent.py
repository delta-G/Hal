from typing import List, Union, Optional, Type
from langchain import OpenAI, PromptTemplate, LLMChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, CombinedMemory
from langchain.agents import Tool, load_tools, initialize_agent, AgentType, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.prompts import StringPromptTemplate 
from langchain.schema import AgentAction, AgentFinish

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader

# from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.agents.tools import tool

from langchain.utilities import OpenWeatherMapAPIWrapper

import os, re 
from langchain.memory.vectorstore import VectorStoreRetrieverMemory

####  Copyright 2023 David Caldwell disco47dave@gmail.com


#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.


template = """ You are Hal, a large language model serving as a digital assistant.  Hal assists a human user named Dave.  Hal is designed to be able to assist \
with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Hal is \
able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses \
that are coherent and relevant to the topic at hand.
Hal always strives to be factual.  If Hal does not know an answer then Hal will truthfully say that it does not know.  It will never hallucinate.  \

Hal has been provided a list of tools that will help with the job.  These tools will allow Hal to interact with the real world in limited ways.  \
Some are for obtaining additional information while others are for taking actions.  

You have access to the following tools:

{tools}

Use the following format your final response MUST start with "Final Answer:" in order to not cause an error:

Question: the input to which you must respond
Thought: you should always think about what to do. 
Action: the action to take, should be one of [{tool_names}] or None.  If Action is None then you should skip to Final Answer and respond now.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times and will be recorded for you in the Agent Scratchpad section)
Thought: I now know the final answer
Final Answer: the final answer or response to the original input.  This MUST be included if no action is taken.

Begin!

Previous conversation history:
[
{chat_history}
]

New Input:
Dave: {input}
Agent Scratchpad:
[
{agent_scratchpad}
]"""



class HalMainPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log 
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts 
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class HalMainOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
                )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)



convoPath = '/home/david/chatWorkspace/Hal/HalConvo/'
chromaPersistDirectory = "/home/david/chatWorkspace/Hal/chromaDB/"



class HalAgent:
    
    def __init__(self):
        
        ###  Load and setup chroma database with old conversations for vector memory
        self.embeddings = OpenAIEmbeddings()
        
        if len(os.listdir(chromaPersistDirectory)) == 0:
            loader = DirectoryLoader(path=convoPath)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            docs = text_splitter.split_documents(documents)
            self.chromaDB = Chroma.from_documents(docs, self.embeddings, persist_directory=chromaPersistDirectory)
            self.retriever = self.chromaDB.as_retriever(search_kwargs=dict(k=3)) 
        else:
            self.chromaDB = Chroma(embedding_function=self.embeddings, persist_directory=chromaPersistDirectory)
            self.retriever = self.chromaDB.as_retriever(search_kwargs=dict(k=3))
        
        
        
        ### declare a language model to use for setting up
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0
            )
        
        ### configure our tools
        ###  load the premade tools first
        loaded_tool_names = ["serpapi", "llm-math"]
        self.tools = load_tools(loaded_tool_names, llm=llm)
        
        ###  then add some custom tools we've created. 
        self.tools.append(
            Tool(
                func=self.searchRetriever,
                name="vecSearch",
                description = "Useful for remembering information from prior conversations with Dave"
                )
            )
        weather = OpenWeatherMapAPIWrapper()
        self.tools.append(
            Tool(
                func=weather.run,
                name="weather",
                description = "Returns the current weather conditions.  Use Jefferson, AR, USA for the location unless told otherwise."
                )
            )
        self.tools.append(
            Tool(
                func=self.getHumanHelp,
                name="human",
                description = "Allows HAL to ask the human operator a question for help or clarification.  Returns the human's response."
                )
            )
        self.tool_names = [t.name for t in self.tools]
        
        
        ###  setup our conversation memory for the present conversation
        convo_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=300,
            memory_key="chat_history",
            input_key="input",
            ai_prefix="Hal",
            human_prefix="Dave"
            )
        
        ### Build the parts of our chat agent
        prompt = HalMainPromptTemplate(
            input_variables=["input", "intermediate_steps", "chat_history"],    
            template=template,
            tools=self.tools
            )
        
        output_parser = HalMainOutputParser()
        
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        
        
        ###  Setup the chat agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser = output_parser,
            stop=["\nObservation:"],
            allowed_tools=self.tool_names,
            )
        
        ###  setup the agent executor
        self.executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                           tools=self.tools,
                                                           memory=convo_memory,
                                                           verbose=True)  
        return 
    
    
    @tool
    def searchRetriever(self, query: str) -> str:
        """ Returns relevant docs from the vector database   """
        docs = self.chromaDB.similarity_search(query)
        prefix = "Start of vecSearch results:[\n"
        retData = '\n'.join([d.page_content for d in docs]) 
        suffix = "\n]End of vecSearch results:"
        return prefix + retData + suffix
    
    @tool
    def getHumanHelp(self, query: str) -> str:
        """Allows the LLM to get additional input from the user"""
        print("-----  HAL is asking for Input --------")
        print(query)
        print("Insert response.  Enter 'q' to end.")
        
        contents = []
        while True:
            line = input()
            if line == 'q':
                break
            contents.append(line)
        return "n".join(contents)
    
    