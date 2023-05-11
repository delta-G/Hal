

####  Copyright 2023 David Caldwell disco47dave@gmail.com

# openai.api_key = os.environ['OPENAI_API_KEY']

import os, re 
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from networkx.generators import line

convoPath = '/home/david/chatWorkspace/chat/HalConvo/'
chromaPersistDirectory = "/home/david/chatWorkspace/chat/chromaDB/"

from typing import List, Union, Optional, Type
from langchain import OpenAI, PromptTemplate, LLMChain, SerpAPIWrapper
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
Action: the action to take, should be one of [{tool_names}] if Action is None then you should skip to Final Answer and respond now
Action Input: the input to the action.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer or response to the original input.  This MUST be included if no action is taken.

Begin!

Relevant parts of past conversations (You don't need to use these pieces of information if not relevant.):
{vec_history}


Previous conversation history:
{chat_history}

New Input:
Dave: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
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
    
class CustomOutputParser(AgentOutputParser):
    
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
    

    
embeddings = OpenAIEmbeddings()

###  See https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html for how to persist
db = Chroma(embedding_function=embeddings, persist_directory=chromaPersistDirectory)
retriever = db.as_retriever(search_kwargs=dict(k=3))

def rebuildChatVectorMemory():
    global db
    global retriever
    loader = DirectoryLoader(path=convoPath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embeddings, persist_directory=chromaPersistDirectory)
    retriever = db.as_retriever(search_kwargs=dict(k=3))
    return



@tool
def searchRetriever(query: str) -> str:
    """ Returns relevant docs from the vector database   """
    docs = db.similarity_search(query)
    prefix = "Start of vecSearch results:[\n"
    retData = '\n'.join([d.page_content for d in docs]) 
    suffix = "\n]End of vecSearch results:"
    return prefix + retData + suffix

def getHumanHelp(query: str) -> str:
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
            
    

weather = OpenWeatherMapAPIWrapper()



# class VecSearchTool(BaseTool):
#     name = "vecSearch"
#     description = "useful for getting information from prior conversations with Dave"
#
#     def _run(self, query: str) -> str:
#         return searchRetriever(query)
#
#     async def _arun(self, query: str) -> str:
#         raise NotImplementedError("custom_search does not support async")


llm = OpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0
    )

loaded_tool_names = ["serpapi", "llm-math"]
tools = load_tools(loaded_tool_names, llm=llm)

tools.append(
    Tool(
        func=searchRetriever,
        name="vecSearch",
        description = "Useful for remembering information from prior conversations with Dave"
        )
    )
tools.append(
    Tool(
        func=weather.run,
        name="weather",
        description = "Returns the current weather conditions.  Use Jefferson, AR, USA for the location unless told otherwise."
        )
    )
tools.append(
    Tool(
        func=getHumanHelp,
        name="human",
        description = "Allows HAL to ask the human operator a question for help or clarification.  Returns the human's response."
        )
    )
tool_names = [t.name for t in tools]


convo_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=300,
    memory_key="chat_history",
    input_key="input",
    ai_prefix="Hal",
    human_prefix="Dave"
    )

if len(os.listdir(chromaPersistDirectory)) == 0:
    rebuildChatVectorMemory()
    
vector_memory = VectorStoreRetrieverMemory(retriever=retriever, input_key="input", memory_key="vec_history")

           


memory = CombinedMemory(memories=[convo_memory, vector_memory])


prompt = CustomPromptTemplate(
    input_variables=["input", "intermediate_steps", "vec_history", "chat_history"],    
    template=template,
    tools=tools
    )

output_parser = CustomOutputParser()

llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)

# agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     verbose=True,
#     memory=memory
#     )


class HAL:
    
    def __init__(self, convo_file):
        
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
            self.ai_output = agent_executor.run(self.human_input) 
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
        db.persist()
        db = None
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
