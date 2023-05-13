

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

import HalAgent

import os, re 
from langchain.memory.vectorstore import VectorStoreRetrieverMemory

convoPath = '/home/david/chatWorkspace/Hal/HalConvo/'
chromaPersistDirectory = "/home/david/chatWorkspace/Hal/chromaDB/"

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

#####Relevant parts of past conversations (You don't need to use these pieces of information if not relevant.):
##### [
##### {vec_history}
##### ]



####  I think this was missing something about func.  I think I need func=self._run in there.  Not sure.  Need to look at BaseTool
# class VecSearchTool(BaseTool):
#     name = "vecSearch"
#     description = "useful for getting information from prior conversations with Dave"
#
#     def _run(self, query: str) -> str:
#         return searchRetriever(query)
#
#     async def _arun(self, query: str) -> str:
#         raise NotImplementedError("custom_search does not support async")


# if len(os.listdir(chromaPersistDirectory)) == 0:
#     rebuildChatVectorMemory()
    
# vector_memory = VectorStoreRetrieverMemory(retriever=retriever, input_key="input", memory_key="vec_history")

           


# memory = CombinedMemory(memories=[convo_memory, vector_memory])

# agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     verbose=True,
#     memory=memory
#     )


class HAL:
    
    def __init__(self, convo_file):
        
        self.agent = HalAgent.HalAgent()
        self.convo_file = convo_file 
        self.exit_flag = False
        
        return
    
    def dualOutput(self, aString):
        print(aString)
        self.convo_file.write(aString)
        
        return 
    
    
    def getUserInput(self):
        print("\n*******************************\n\nDave:  ")
        self.human_input = input("")
        if self.human_input == "exit":
            self.exit_flag = True
        return 
    
    def printResponse(self):
        
        if self.ai_output is not None:
            self.dualOutput("\n*******************************\n\nHal:  ")
            self.dualOutput(self.ai_output)
        
        return 
    
    def getResponse(self):
            
        if self.human_input is not None:
            self.ai_output = self.agent.executor.run(self.human_input) 
            self.convo_file.write("\n*******************************\n\nDave:  ")
            self.convo_file.write(self.human_input)                              
        
        return 
    
    def run(self):
        
        global db
        self.getUserInput()
        while not self.exit_flag:
            self.getResponse()
            self.printResponse()
            self.getUserInput()    
        return 
    

if not os.path.exists(convoPath):
    os.mkdir(convoPath)
    
lastNum = 0
for (root, dirs, files) in os.walk(convoPath):
    for f in files:
        num = int(re.search('Halconvo(\d*)', f).group(1))
        lastNum = num if num > lastNum else lastNum

lastNum = lastNum + 1
convoFileName = f"{convoPath}Halconvo{lastNum:04d}.txt"

with open(convoFileName, 'x') as convoFile:
    
    hal = HAL(convoFile)
    hal.run()
