

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
import HalGui

import os, re 

convoPath = '/home/david/chatWorkspace/Hal/HalConvo/'
chromaPersistDirectory = "/home/david/chatWorkspace/Hal/chromaDB/"
    

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
    
    # hal = HAL(convoFile)
    agent = HalAgent.HalAgent()
    gui = HalGui.HalGui(agent, convoFile)   
    
    
    try:
        gui.parent.mainloop()
    except Exception as e:
        print("**********  HAL EXCEPTION  **********")
        print(e) 
    finally:
        agent.closingOut()
    
