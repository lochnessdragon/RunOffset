import json
import random

def randdecimal(min, max):
	spread = max-min
	random_num = random.random()
	return min + (random_num * spread)

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

	for i in range(0, 100):
		pos_data.append({'ph': randdecimal(ph_min, ph_max), 'elevation': randdecimal(elev_min, elev_max), 'pesticide': randdecimal(pesticide_min, pesticide_max), 'pos': [randdecimal(pos_min, pos_max), randdecimal(pos_min, pos_max)]})

	with open("data.json", 'w') as file:
		json.dump(pos_data, file)