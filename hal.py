

####  Copyright 2023 David Caldwell disco47dave@gmail.com

# openai.api_key = os.environ['OPENAI_API_KEY']

import os, re 

convoPath = '/home/david/chatWorkspace/chat/HalConvo/'


from langchain import OpenAI, PromptTemplate, LLMChain, SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, load_tools, initialize_agent, AgentType


template = """ You are Hal, a large language model serving as a digital assistant.  Hal assists a human user named Dave.  Hal is designed to be able to assist \
with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Hal is \
able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses \
that are coherent and relevant to the topic at hand.
Hal always strives to be factual.  If Hal does not know an answer then Hal will truthfully say that it does not know.  It will never hallucinate.  \
Hal always attempts to sound as human as possible.  If an answer requires emotion or some human quality, Hal will attempt to react as a human would \
or will reply that it does not have the emotional tools to answer but will never call itself an AI model.  Hal is a machine, but does not call itslef one unless that fact is relevant. 
Hal has been provided a list of tools that will help with the job.  These tools will allow Hal to interact with the real world in limited ways.  \
Some are for obtaining additional information while others are for taking actions.  

You have access to the following tools:

{tools}

Use the following format:

Dave: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Dave: {human_input}
{agent_scratchpad}"""


prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
    )

llm = OpenAI(temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history")

tool_names = ["serpapi", "llm-math"]
tools = load_tools(tool_names, llm=llm)

agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

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
        self.dualOutput("\n*******************************\n\nDave:  ")
        self.human_input = input("")
        if self.human_input == "exit":
            self.exit_flag = True
        self.convo_file.write(self.human_input)
        return 
    
    def printResponse(self):
        
        if self.ai_output is not None:
            self.dualOutput("\n*******************************\n\nHal:  ")
            self.dualOutput(self.ai_output)
        
        return 
    
    def getResponse(self):
            
        if self.human_input is not None:
            self.ai_output = agent_chain.run(input=self.human_input)                               
        
        return 
    
    def run(self):
        
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
