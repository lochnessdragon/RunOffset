import json
import math
from math import *
import random

def randdecimal(min, max):
	spread = max-min
	random_num = random.random()
	return min + (random_num * spread)

def ph_f(x, y):
	return (1/6) * (math.sin(x) + (1/6) * cos(y))

def elev_f(x, y):
	return (1/70) * (x + y) + (1/100) * sin(x)

def pest_f(x,y):
	return ((1)/(100)) * (x+y) + ((1)/(9)) * cos(y) + ((1)/(15)) * sin(x-((math.pi)/(2)))

if __name__ == "__main__":
	# generate some test data
	pos_data = []
	ph_min = 6.0
	ph_max = 8.0
	pesticide_min = 5.0
	pesticide_max = 8.0
	elev_min = 500
	elev_max = 600
	pos_min = 0
	pos_max = 600

	start_t = 0.0
	t = 0.0
	t_inc = 1/200

	ph_base = 6.0
	ph_spread = 2.0

	elev_base = 500
	elev_spread = 500

	pest_base = 7.5
	pest_spread = 10

	for x in range(0, 500, 10):
		# pos_data.append({'ph': randdecimal(ph_min, ph_max), 'elevation': randdecimal(elev_min, elev_max), 'pesticide': randdecimal(pesticide_min, pesticide_max), 'pos': [randdecimal(pos_min, pos_max), randdecimal(pos_min, pos_max)]})
		for y in range(0, 500, 10):
			position = {'pos': [x, y], 'ph': ph_f(x, y), 'elevation': elev_f(x,y), 'pesticide': pest_f(x,y)}
			pos_data.append(position)



	with open("data.json", 'w') as file:
		json.dump(pos_data, file)