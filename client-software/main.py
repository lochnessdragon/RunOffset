from tkinter import *
import customtkinter

if __name__ == "__main__":
	root = customtkinter.CTk()
	root.title("RunOffset")
	root.geometry('600x500')
	root.grid_columnconfigure(0,weight=1)

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
