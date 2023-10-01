import json
from tkinter import *
import customtkinter

def find_min_in_arr(data, name):
	minimum = data[0][name]
	for val in data:
		if val[name] < minimum:
			minimum = val[name]

def find_max_in_arr(data, name):
	maximum = 0 
	for val in data:
		if val[name] > maximum:
			maximum = val[name]

if __name__ == "__main__":
	with open("data.json") as file:
		data = json.load(file)
	
	ph_min = find_min_in_arr(data, 'ph')
	ph_max = find_max_in_arr(data, 'ph')
	pesticide_min = find_min_in_arr(data, 'pesticide')
	pesticide_max = find_max_in_arr(data, 'pesticide')
	elevation_min = find_min_in_arr(data, 'elevation')
	elevation_max = find_max_in_arr(data, 'elevation')
	# pos_x_min = 
	# pos_x_max = 
	# pos_y_min = 
	# pos_y_max = 

	root = customtkinter.CTk()
	root.title("RunOffset")
	root.geometry('600x500')
	root.grid_columnconfigure(0,weight=1)
	icon = PhotoImage(file = "./logo.png")
	root.iconphoto(False, icon)

	content = customtkinter.CTkFrame(root)
	content.grid(column=0, row=0, sticky="nsew")

	display_type_selection_label = customtkinter.CTkLabel(master=content, text="Select map type: ")
	display_type_selection_label.grid(column=0, row=0, padx=2)

	display_type_selections = ["Topographic", "pH", "Pesticide Concentration"]
	display_type = StringVar(root)
	display_type.set("Topographic")
	dropdown = customtkinter.CTkOptionMenu(master=content, variable=display_type, values=display_type_selections)
	dropdown.grid(column=1, row=0, padx=2)

	root.mainloop()
