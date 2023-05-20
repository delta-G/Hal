

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




####  TODO:   As of right now this is just a copy of ConversationSummaryBufferMemory.
####   TODO:  add code so this will save lines from the conversation to the vector store before closing. 


from typing import Any, Dict, List

from pydantic import root_validator, Field

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.summary import SummarizerMixin
from langchain.schema import BaseMessage, get_buffer_string
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document


class HalConversationMemory(BaseChatMemory, SummarizerMixin):
    """Buffer with summarizer for storing conversation memory."""

    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    memory_key: str = "history"
    retriever: VectorStoreRetriever = Field(exclude=True)
    dataBase:  Any

    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer = self.buffer
        if self.moving_summary_buffer != "":
            first_messages: List[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer)
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
            )
        return {self.memory_key: final_buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.prune()

    def prune(self) -> None:
        """Prune buffer if it exceeds max token limit"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                # if (buffer[0].type == "human") and (buffer[1].type == "ai"):
                #     page_content = "\n".join([buffer[0], buffer[1]])
                #     newdoc = Document(page_content=page_content)
                #     self.retriever.add_documents([newdoc])
                #     print("*@*@*@*@*@*@*@*@* ADDED DOCS  @*@*@*@*@*@*@*@*")
                #     print(newdoc.page_content)
                #     pruned_memory.append(buffer.pop(0))
                #     pruned_memory.append(buffer.pop(0))
                # else:
                #### Let's always keep question and answer pairs together
                ####  So we will always pop two messages at a time.
                pruned_memory.append(buffer.pop(0)) 
                pruned_memory.append(buffer.pop(0))                    
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            # print("*@*@*@*@*@*@*@*@* Summarizing  @*@*@*@*@*@*@*@*")
            docs = []
            for i in range(0,len(pruned_memory),2):                
                firstType = pruned_memory[i].type 
                firstMessage = pruned_memory[i].content 
                secondType = pruned_memory[i+1].type 
                secondMessage = pruned_memory[i+1].content 
                text = f"{firstType}: {firstMessage}\n{secondType}: {secondMessage}"
                docs.append(Document(page_content=text))
            self.retriever.add_documents(docs)                
                
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )
            
    def saveRemainingConversation(self):
        
        buffer = self.chat_memory.messages 
        docs = []
        while (len(buffer) >= 2):
            first = buffer.pop(0)
            second = buffer.pop(0)
            text = f"{first.type}: {first.content}\n{second.type}: {second.content}"
            docs.append(Document(page_content=text))
        ## if there's a last message get it too
        if(len(buffer)):
            text = f"{buffer[0].type}: {buffer[0].content}"
            docs.append(Document(page_content=text))
        if len(docs) > 0:
            self.retriever.add_documents(docs)        
        
        return

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.moving_summary_buffer = ""
        
        
        
        
        
        
        
        
        