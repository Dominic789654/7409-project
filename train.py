from agent.agent import Agent
from functions import *
import pandas as pd
from pandas_datareader import data
import sys
from tqdm import trange
import yfinance as yf
yf.pdr_override()

df = data.get_data_yahoo(tickers='^HSI',start='2012-01-01', end='2022-12-31')
df = list(df["Close"])
if len(sys.argv) != 3:
	print ("Usage: python train.py [window] [episodes]")
	exit()

window_size, episode_count =  int(sys.argv[1]), int(sys.argv[2])

agent = Agent(window_size)
l = len(df) - 1
batch_size = 32

for e in trange(episode_count + 1):
	print ("Episode " + str(e) + "/" + str(episode_count))
	state = getState(df, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(df, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(df[t])
			print ("Buy: " + formatPrice(df[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(df[t] - bought_price, 0)
			total_profit += df[t] - bought_price
			print ("Sell: " + formatPrice(df[t]) + " | Profit: " + formatPrice(df[t] - bought_price))

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
