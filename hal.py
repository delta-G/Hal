

####  Copyright 2023 David Caldwell disco47dave@gmail.com

# openai.api_key = os.environ['OPENAI_API_KEY']

import os, re 

convoPath = '/home/david/chatWorkspace/chat/HalConvo/'

from typing import List, Union
from langchain import OpenAI, PromptTemplate, LLMChain, SerpAPIWrapper
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool, load_tools, initialize_agent, AgentType, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.prompts import StringPromptTemplate 
from langchain.schema import AgentAction, AgentFinish


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

Question: the input to which you must respond
Thought: you should always think about what to do. 
Action: the action to take, should be one of [{tool_names}] if Action is None then you should skip to Final Answer and respond now
Action Input: the input to the action.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer or response to the original input.  This MUST be included if no action is taken.

Begin!

Previous conversation history:
{history}

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
    

    

llm = OpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0
    )

tool_names = ["serpapi", "llm-math"]
tools = load_tools(tool_names, llm=llm)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,
    # memory_key="chat_history",
    ai_prefix="Hal",
    human_prefix="Dave"
    )

prompt = CustomPromptTemplate(
    input_variables=["input", "intermediate_steps", "history"],    
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
            self.ai_output = agent_executor.run(self.human_input)                               
        
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
