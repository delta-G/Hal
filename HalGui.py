
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

commonConfig = {'bg':'black', 'highlightthickness':1, 'highlightbackground':colors['highlight']}
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
        
        ###   Create a frame to hold the input/output section:
        self.conversationFrame = tk.Frame(self, **commonConfig)
        
        ###  Create Frames, Text boxes, and scroll bars for the input and output
        self.outputFrame = tk.Frame(self.conversationFrame, **framePadding, **commonConfig)
        self.inputFrame = tk.Frame(self.conversationFrame, **framePadding, **commonConfig) 
        
        self.outText = tk.Text(self.outputFrame, fg='white', state=tk.DISABLED, takefocus=0, width=200, height = 20, padx=5, pady=5, **commonConfig)        
        self.inText = tk.Text(self.inputFrame, fg='white', width=200, height = 20, padx=5, pady=5, **commonConfig)
        
        self.outputScroll = tk.Scrollbar(self.outputFrame, command=self.outText.yview, **commonConfig) 
        self.inputScroll = tk.Scrollbar(self.inputFrame, command=self.inText.yview, **commonConfig) 
        
        self.outputScroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.inputScroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.outText.pack()
        self.inText.pack()
        
        self.outputFrame.pack(side=tk.TOP)
        self.inputFrame.pack(side=tk.TOP)
        
        ###  A frame for the bottom with the buttons        
        self.buttonFrame = tk.Frame(self, **framePadding, bg=colors['red'])
        self.sendButton = tk.Button(self.buttonFrame, fg='white', text="Send", command=self.submit, **commonConfig)
        self.sendButton.pack()
        
        self.conversationFrame.pack(side=tk.TOP) 
        self.buttonFrame.pack(side=tk.TOP)
        
        self.pack()
        
        return 
    
    def addText(self, aStr):        
        self.inText.insert(tk.END, aStr)
        return
    
    def submit(self):
        formInput = self.inText.get("1.0", tk.END)
        
        if formInput is not None:
            self.ai_output = self.agent.run(formInput) 
            if self.convoFile is not None:
                self.convoFile.write(daveDelim)
                self.convoFile.write(formInput)
                
            self.outText.config(state=tk.NORMAL)
            self.outText.insert(tk.END, daveDelim)
            self.outText.insert(tk.END, formInput)            
            self.outText.config(state=tk.DISABLED)
            
            if self.ai_output is not None:
                if self.convoFile is not None:
                    self.convoFile.write(halDelim)
                    self.convoFile.write(self.ai_output) 
                self.outText.config(state=tk.NORMAL)
                self.outText.insert(tk.END, halDelim)
                self.outText.insert(tk.END, self.ai_output)
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
        
        
        
        
        
        
        
        
        