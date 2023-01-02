#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import statistics

random_grid = [
    [random.randint(0, 9) for t in range(0, 15)]
    for i in range(0, 11)]
for row in random_grid:
    print(row)


# In[9]:


# Task 1 Question 1 Solution


# In[10]:


def isValidMove(i, j, m, n):
    if (i > m-1 or j > n-1):
        return False
    return True

# Function that returns all adjacent elements
def getAdjacent(i, j, m, n):
# Initialising a vector array
    v1 = 99
    v2 = 99
    # Checking for all the possible adjacent positions
    if (isValidMove(i + 1, j, m, n)):
        v1=(mat[i + 1][j])
    if (isValidMove(i, j + 1, m, n)):
        v2=(mat[i][j + 1])

    # Returning the vector
    return v1, v2

def moveDecider(i, j, m, n):
    if (i > d[0] or j > d[1]):
        return
    path.append((i,j))
    values.append(mat[i][j])
    
    v1, v2 = getAdjacent(i, j, m, n)
    if v1 <= v2 :
        moveDecider(i+1, j, m, n)
    if v1 > v2:
        moveDecider(i, j+1, m, n)
    return
    
def calsum(l):
    # returning sum of list using List comprehension
    return  sum([int(i) for i in l if type(i)== int or i.isdigit()])     

# Driver Code
if __name__ == "__main__":
  mat = random_grid
  # Size of given 2d array
  m = len(mat)
  n = len(mat[0])
  # print(m, n)
  s = [0, 0]
  d = [m-1, n-1]
  path = []
  values = []
  moveDecider(s[0], s[1], len(mat), len(mat[0]))
  print(path)
  print(values)
  print(calsum(values))


# In[11]:


# Task 1 Question 2 Solution: Implementing Djikstra (using a simple priority queue)


# In[12]:


import random
from heapq import heappush, heappop # Importing Priority Queue using heap module

def shortestPathfinder(grid):
	h = [] # initialise empty priority queue
	startNode = (grid[0][0], (0,0)) # starting node : time cost, coordinates of a node
	heappush(h, startNode) # initialise priority queue with pushing startNode
	directions = [(1,0), (0,1), (-1,0), (0,-1) ] # 4-way directions since diagonal moves not allowed
	costsVisited = {(0,0): grid[0][0]} # stores min time costs to get to all nodes
	moveFrom = {(0,0): None} # stores to source of moves to directions
	while h:
		cost, node = heappop(h) # getting time cost, node coordinates from queue
		x, y = node # coordinates of a node
		if x == len(grid)-1 and y == len(grid[0])-1: # destination found
			break

		# exploring nextNode
		for dir in directions:
			nextX, nextY = x+dir[0], y+dir[1] # next nodes coordinates to check
			# check boundaries
			if 0 <= nextX <= len(grid)-1 and 0 <= nextY <= len(grid[0])-1:
				nextNodeCost, nextNode = grid[nextX][nextY], (nextX, nextY) # exploring next node's time cost
				newCost = cost + nextNodeCost # calculating move's total cost
				# check if time costs needs to be updated
				if ( nextNode not in costsVisited or (nextNode in costsVisited and costsVisited[nextNode] > newCost) ):
					costsVisited[nextNode] = newCost # storing next node and its time cost to visited nodes list
					heappush(h, (newCost, nextNode)) # updating queue
					moveFrom[nextNode] = node # updating source node of a move

	# Shortest path:
	path = [] # path
	targetNode = (len(grid)-1, len(grid[0])-1) # target node
	while targetNode in moveFrom:
		path.insert(0, targetNode)
		targetNode = moveFrom[targetNode]
    
	#print("total cost = ", costsVisited[(x,y)]) # prints total time cost to reach destination
	return path, costsVisited[(x,y)]

# Driver Code
if __name__ == "__main__":
    # cellDistribution = random.randint(0, 9) # find another way to define distributions
    sizeGrid = [11, 15]
    random_grid = [[random.randint(0, 9) for t in range(0, sizeGrid[1])]
        for i in range(0, sizeGrid[0])] # initialise a grid
    #for row in random_grid:
        #print(row)
    shortestPath = shortestPathfinder(random_grid)
    print("shortest path = ", shortestPath[0], "total cost = ", shortestPath[1])


# In[6]:


# Task 1 Question 3 Solution:


# In[7]:


# Statistical analysis
# initialize numbers lists
lst = list(range(2, 20))

# simulating permutations of the list in
# a group of 2
pair_list = list(itertools.combinations(lst,2)) # get combinations of possible rows and columns
matrixSize = []
time = []
time_binary = []
for size in pair_list:
    totalNodes = size[0]*size[1] # calculate matrix size for each combination
    if totalNodes not in matrixSize:
        matrixSize.append(totalNodes)
    else:
        continue
    random_grid = [[random.randint(0, 9) for t in range(0, size[1])]
                        for i in range(0, size[0])] # initialise a grid
    binary_grid = [[random.randint(0, 1) for t in range(0, size[1])]
                        for i in range(0, size[0])] # initialise a grid
    shortestTime = shortestPathfinder(random_grid)[1] # calculate shortest path's time
    shortestTime_binary = shortestPathfinder(binary_grid)[1]
    time.append(shortestTime)
    time_binary.append(shortestTime_binary)

zipped = list(zip(matrixSize, time))
zipped_binary = list(zip(matrixSize, time_binary))
# Using sorted and lambda to sort according to matrix size
listSizeTime = sorted(zipped, key = lambda x: x[0])
listSizeTime_binary = sorted(zipped_binary, key = lambda x: x[0])
lx = [x for x,y in listSizeTime]
ly = [y for x,y in listSizeTime]
lx_binary = [x for x,y in listSizeTime_binary]
ly_binary = [y for x,y in listSizeTime_binary]
# plotting
plt.title("Random Valued Matrix Size vs Shortest Time")
plt.xlabel("Matrix Size")
plt.ylabel("Shortest Time")
plt.plot(lx, ly, color ="green")
#calculate equation for quadratic trendline
z = np.polyfit(lx, ly, 2)
p = np.poly1d(z)
#add trendline to plot
plt.plot(lx, p(lx))
plt.show()
print("standard deviation : ", statistics.pstdev(ly))

plt.title("Binary Valued Matrix Size vs Shortest Time")
plt.xlabel("Matrix Size")
plt.ylabel("Shortest Time")
plt.plot(lx_binary, ly_binary, color ="green")

#calculate equation for quadratic trendline
z = np.polyfit(lx_binary, ly_binary, 2)
p = np.poly1d(z)
#add trendline to plot
plt.plot(lx_binary, p(lx_binary))
plt.show()
print("standard deviation : ", statistics.pstdev(ly_binary))


# In[ ]:




