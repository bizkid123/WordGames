import numpy as np
import math
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from copy import copy
import time

class TrieNode:
	def __init__(self, depth, is_word = False):
		self.depth = depth
		self.children = []
		self.has_children = False
		self.word_id = -1

	def addWord(self, word, ind, word_id):
		if ind == len(word):
			self.word_id = word_id
			return
		if not self.has_children:
			self.has_children = True
			self.children = [TrieNode(self.depth+1) for _ in range(26)]
		# print(word[ind])
		(self.children[word[ind]]).addWord(word, ind+1, word_id)

def buildTrie(filename):
	trie = TrieNode(0)
	used_words = []
	with open(filename, "r") as f:
		words = f.readlines()
		used_words = []
		for word_id, word in enumerate(words):
			word = [ord(c)- ord('a') for c in word.lower().strip()]
			trie.addWord(word, 0, word_id)
			used_words.append(word)
	return trie, used_words

def word_score(word):
	length = len(word)
	if length < 3:
		return 0
	elif length == 3:
		return 100
	elif length == 4:
		return 400
	elif length == 5:
		return 800
	else:
		return length*400 - 1000
 
	# length = len(word)
	# # print(word, length)
	# if length < 3:
	# 	return 0
	# elif length <= 4:
	# 	return 1
	# elif length == 5:
	# 	return 2
	# elif length == 6:
	# 	return 3
	# elif length == 7:
	# 	return 5
	# else:
	# 	return 11

class Board:
	def __init__(self, board, width=4, height=4):
		self.board = board
		self.width = width
		self.height = height
		self.initialize_neis()

	def initialize_neis(self):
		self.neighbors_list = []
		for index in range(self.width * self.height):
			index_w = index%self.width
			index_h = index//self.height
			index_neis = []
			for dx, dy in ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)):
				if (0 <= index_w + dx < self.width) and (0 <= index_h + dy < self.height):
					new_index = (index_w+dx) + (index_h + dy) * self.height
					index_neis.append(new_index)
			self.neighbors_list.append(index_neis)

	def getScore(self, show_words=False):
		usedWords = [False]*len(words)
		visited = [False]*(self.width * self.height)
		for i, c in enumerate(self.board):
			visited[i] = True
			self.recursiveScoreHelper(i, visited, trie.children[c], usedWords)
			visited[i] = False
		score = sum(word_score(word) for found, word in zip(usedWords, words) if found)
		# print(*[(''.join(chr (c+ord('a')) for c in word), word_score(word)) for found, word in sorted(zip(usedWords, words), key=lambda x:-len(x[1])) if found],sep="\n")
		if show_words:
			used_words_list = []	
			for i in range(len(words)):
				if usedWords[i]:
					used_words_list.append(''.join(chr(c+ord('a')) for c in words[i]))
			used_words_list.sort(key=len, reverse=True)
			# print("\n".join(used_words_list))
			print(used_words_list)
		return score

	def recursiveScoreHelper(self, index, visited, trie_level, used_words):
		# score = 0
		# print(index, visited, trie_level.)
		if (trie_level.word_id != -1) and (used_words[trie_level.word_id] == False):
			used_words[trie_level.word_id] = True
			# score += word_score(words[trie_level.word_id])
			# print(words[trie_level.word_id])
		if not trie_level.has_children:
			return 
		for nei in self.neighbors_list[index]:
			if visited[nei] == False:
				visited[nei] = True
				self.recursiveScoreHelper(nei, visited, trie_level.children[self.board[nei]], used_words)
				visited[nei] = False
		# return score

	def combineBoardUniform(self, other):
		if self.width != other.width or self.height != other.height:
			raise Exception("Can't combine, different dimensions")
		newBoard = []
		for i in range(self.width * self.height):
			newBoard.append(random.choice([self.board[i], other.board[i]]))
		return Board(newBoard, self.width, self.height)
	
	def combineBoardCrossover(self, other):
		if self.width != other.width or self.height != other.height:
			raise Exception("Can't combine, different dimensions")
		newBoard = []
		crossoverPoint = random.randint(1,25)
		for i in range(self.width * self.height):
			if i<crossoverPoint:
				newBoard.append(self.board[i])
			else:
				newBoard.append(other.board[i])
		return Board(newBoard, self.width, self.height)

	def random_mutation(self, mu):
		for ind in range(len(self.board)):
			if random.random() < mu:
				self.board[ind] = random.randint(0,25)

	def swapRandomNeighbors(self):
		ind1 = random.randint(0,self.width * self.height-1)
		ind2 = random.choice(self.neighbors_list[ind1])
		self.board[ind1], self.board[ind2] = self.board[ind2], self.board[ind1]

	def changeRandom(self):
		ind1 = random.randint(0,self.width * self.height-1)
		self.board[ind1] = random.randint(0,25)

	def __copy__(self):
		new_copy = Board(self.board.copy(), self.width, self.height)
		return new_copy
	copy = __copy__ 

	def __str__(self):
		s=[]
		for i in range(self.height):
			s+=["".join(chr(c+ord('a')) for c in self.board[i*self.width:(i+1)*self.width])]
		return '\n'.join(s)

def hillClimb(board, logging_enabled=False):
	lastImproveTime = 0
	bestScore = board.getScore()
	
	with open('hill_climb_output2.txt', 'w') as log_file:
		if logging_enabled:
			print(logging_enabled)
			log_file.write(f"{board}\n")
			log_file.write(f"{bestScore}\n\n")
		
		while lastImproveTime < 500:
			lastImproveTime += 1
			prevBoard = board.board.copy()
			if random.random() > 0.5:
				board.swapRandomNeighbors()
			else:
				board.changeRandom()
			newScore = board.getScore()
			if newScore > bestScore:
				if logging_enabled:
					log_file.write(f"{board}\n")
					log_file.write(f"{newScore}\n\n")
				bestScore = newScore
				lastImproveTime = 0
			else: 
				board.board = prevBoard

def randomBoard(width = 4, height = 4):
	return Board([random.randint(0,25)for _ in range(width*height)], width, height)

def geneticAlgorithm(pop_size = 300, num_iter = 100, width = 4, height = 4):
	boards = [randomBoard(width, height) for _ in range(pop_size)]
	board_copies = [board.copy() for board in boards]
	min_scores = []
	median_scores = []
	max_scores = []
	for iteration in range(num_iter):
		for board in boards:
			hillClimb(board)

		sorted_copies = [x for _, x in sorted(zip(boards, board_copies), key=lambda board: board[0].getScore(), reverse=True)]
		boards.sort(key=lambda board: board.getScore(), reverse=True)

		scores = np.array([board.getScore() for board in boards])
		
		min_scores.append(scores.min())
		median_scores.append(np.median(scores))
		max_scores.append(scores.max())

		# Calculate the weights based on scores using exponential scaling
		exp_scores = np.exp(scores- scores.max())
		weights = exp_scores / exp_scores.sum()

		print("--------"+str(iteration)+"--------")
		print(f"{scores=}")
		# print([round(weight, 4) for weight in weights])
		print(f"{min_scores=}")
		print(f"{median_scores=}")
		print(f"{max_scores=}")
  
		if max_scores[-1] > max(max_scores[:-1]+[0]):
			print(boards[0])

		newBoards = []
		num_parents = 100
		for _ in range(pop_size):
			parent1, parent2 = random.choices(population=boards[:num_parents], weights=[(num_parents-i)**2 for i in range(num_parents)], k=2)
			# parent1, parent2 = random.choices(population=boards, weights=weights, k=2)
			if random.random() > 0.5:
				newBoard = parent1.combineBoardUniform(parent2)
			else:
	   			newBoard = parent1.combineBoardCrossover(parent2)
			newBoard.random_mutation(mu = 4/(width * height))
			newBoards.append(newBoard)
		boards = newBoards
		board_copies = [board.copy() for board in boards]

		# Plot the scores
		# clear_output(wait=True)
		# plt.figure(figsize=(10, 5))
		# plt.plot(min_scores, label='Min Score')
		# plt.plot(median_scores, label='Median Score')
		# plt.plot(max_scores, label='Max Score')
		# plt.xlabel('Iteration')
		# plt.ylabel('Score')
		# plt.legend()
		# plt.title('Genetic Algorithm Scores')
		# plt.show()
		# time.sleep(0.1)  # Pause to update the plot
  
trie, words = buildTrie("wordsDict.txt")
hillClimb(randomBoard(), logging_enabled=True)
# geneticAlgorithm()