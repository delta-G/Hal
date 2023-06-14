
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


import tkinter as tk

import time



daveDelim = "\n******************************\n\nDave:  "
halDelim = "\n******************************\n\nHal:  "

colors = {
            'red' : '#990303',
            'green' : '#00FF00',
            'yellow' : '#FFFF00',
            'highlight' : 'red'
            }

highlightColor = 'red'

commonConfig = {'bg':'black'}
highlightConfig = {'highlightthickness':1, 'highlightbackground':colors['highlight']}
framePadding = {'padx':10, 'pady':10}

class HalGui(tk.Frame):
    
    def __init__(self, aAgent, aConvoFile):
        
        self.parent = tk.Tk()
        # self.parent = aParent 
        self.agent = aAgent
        self.convoFile = aConvoFile
        
        img = tk.PhotoImage(file='/home/david/chatWorkspace/Hal/hal.png') 
        self.parent.tk.call('wm', 'iconphoto', self.parent._w, img)
        
        
        tk.Frame.__init__(self, self.parent, **framePadding, bg=colors['red'])
        
        self.topFrame = tk.Frame(self, **commonConfig)
        
        ###   Create a frame to hold the input/output section:
        self.conversationFrame = tk.Frame(self.topFrame, **commonConfig, **highlightConfig)
        self.sideFrame = tk.Frame(self.topFrame, **commonConfig)
        
        ###  Create Frames, Text boxes, and scroll bars for the input and output
        self.outputFrame = tk.Frame(self.conversationFrame, **framePadding, **commonConfig, **highlightConfig)
        self.inputFrame = tk.Frame(self.conversationFrame, **framePadding, **commonConfig, **highlightConfig) 
        
        self.outText = tk.Text(self.outputFrame, fg='white', state=tk.DISABLED, takefocus=0, width=200, height = 20, padx=5, pady=5, **commonConfig, **highlightConfig)        
        self.inText = tk.Text(self.inputFrame, fg='white', width=200, height = 20, padx=5, pady=5, **commonConfig, **highlightConfig)
        
        self.outLabel = tk.Label(self.outputFrame, fg='white', text="OUTPUT - HAL:", **commonConfig)
        self.inLabel = tk.Label(self.inputFrame, fg='white', text="INPUT - Dave:", **commonConfig)
        
        self.outputScroll = tk.Scrollbar(self.outputFrame, command=self.outText.yview, **commonConfig) 
        self.inputScroll = tk.Scrollbar(self.inputFrame, command=self.inText.yview, **commonConfig) 
        
        self.outputScroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.inputScroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.outLabel.pack(side=tk.TOP, anchor='w')
        self.outText.pack()
        
        self.inLabel.pack(side=tk.TOP, anchor='w')
        self.inText.pack()
        
        self.outputFrame.pack(side=tk.TOP)
        self.inputFrame.pack(side=tk.TOP)
        
        ###  A frame for the tools toggles
        
        self.toggleFrame = tk.Frame(self.sideFrame, **framePadding, **commonConfig, **highlightConfig) 
        
        self.toggleLabel = tk.Label(self.sideFrame, fg='white', text="Tools", **commonConfig)
        self.toggleLabel.pack(side=tk.TOP, anchor='w')        
        self.toggles = {}
        for tool in self.agent.tools:
            self.toggles[tool.name] = tk.BooleanVar()
            toggle = tk.Checkbutton(self.toggleFrame, fg='white', selectcolor='black', text=tool.name, highlightthickness=0, bd=0, variable=self.toggles[tool.name], **commonConfig)
            toggle.pack(side=tk.TOP, anchor='w')
        
        ###  A frame for the bottom with the buttons        
        self.buttonFrame = tk.Frame(self, **framePadding, bg=colors['red'])
        self.sendButton = tk.Button(self.buttonFrame, fg='white', text="Send", command=self.submit, **commonConfig, **highlightConfig)
        self.sendButton.pack()
        
        self.topFrame.pack(side=tk.TOP)
        self.conversationFrame.pack(side=tk.LEFT) 
        self.sideFrame.pack(side=tk.LEFT)
        self.toggleFrame.pack(side=tk.TOP)
        self.buttonFrame.pack(side=tk.TOP)
        
        self.pack()
        
        return 
    
    def addText(self, aStr):        
        self.inText.insert(tk.END, aStr)
        return
    
    
    def submit(self):
        formInput = self.inText.get("1.0", tk.END)
        
        if formInput is not None:
            
            selectedTools = [toolName for toolName in self.toggles if self.toggles[toolName].get() == True] 
            self.agent.prompt.tools = [tool for tool in self.agent.tools if tool.name in selectedTools]
            self.ai_output = self.agent.run(formInput) 
            if self.convoFile is not None:
                self.convoFile.write(daveDelim)
                self.convoFile.write(formInput)
                
            self.outText.config(state=tk.NORMAL)
            self.outText.insert(tk.END, daveDelim)
            self.outText.insert(tk.END, formInput)  
            self.outText.see(tk.END)          
            self.outText.config(state=tk.DISABLED)
            
            if self.ai_output is not None:
                if self.convoFile is not None:
                    self.convoFile.write(halDelim)
                    self.convoFile.write(self.ai_output) 
                self.outText.config(state=tk.NORMAL)
                self.outText.insert(tk.END, halDelim)
                self.outText.insert(tk.END, self.ai_output)
                self.outText.see(tk.END)
                self.outText.config(state=tk.DISABLED)
            self.inText.delete("1.0", tk.END)

        return 


    def run(self):
        self.parent.update_idletasks()
        self.parent.update() 
        time.sleep(0.01)
        return 
    
    def runMainLoop(self):
        self.parent.mainloop()
        
        
        
        
        
        
        
        
        