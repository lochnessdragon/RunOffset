# render.py
# render a topo map

import json
import sys
from scipy.spatial import Delaunay
from OpenGL.GL import *
import glfw
import cv2
import numpy
from OpenGL.arrays.vbo import VBO
import ctypes

def extract_array(data, name):
	arr = []
	for val in data:
		arr.append(val[name])

	return arr

def min_idx(arr, idx):
	min = arr[0][idx]
	for val in arr:
		if val[idx] < min:
			min = val[idx]
	return min

def max_idx(arr, idx):
	max = 0
	for val in arr:
		if val[idx] > max:
			max = val[idx]
	return max

def normalize(value, min, max):
	return (value - min) / (max - min)

def isNormalized(arr):
	for val in arr:
		if val < 0 or val > 1:
			return False
	return True

print("Extracting data.")
with open("data.json") as file:
	data = json.load(file)

ph = extract_array(data, 'ph')
pesticide = extract_array(data, 'pesticide')
elevation = extract_array(data, 'elevation')
positions = extract_array(data, 'pos')
ph_min = min(ph)
ph_max = max(ph)
pesticide_min = min(pesticide)
pesticide_max = max(pesticide)
elevation_min = min(elevation)
elevation_max = max(elevation)
pos_min = [min_idx(positions, 0), min_idx(positions, 1)]
pos_max = [max_idx(positions, 0), max_idx(positions, 1)]

print("Min pH:", ph_min, "Max pH:", ph_max, "Min Pesticide:", pesticide_min, "Max Pesticide:", pesticide_max, "Min Elevation:", elevation_min, "Max Elevation:", elevation_max, "Pos Min:", pos_min, "Pos Max:", pos_max)

ph = [normalize(value, ph_min, ph_max) for value in ph]
pesticide = [normalize(value, pesticide_min, pesticide_max) for value in pesticide]
elevation = [normalize(value, elevation_min, elevation_max) for value in elevation]

window_size = [round(pos_max[0] - pos_min[0]), round(pos_max[1] - pos_min[1])]

print("Triangulating.")
triangulation = Delaunay(positions)
# generate indices from delaunay info
indices = triangulation.simplices.flatten()

print("Rendering images: " + str(window_size))
# create opengl image

def checkShaderError(shader):
	status = glGetShaderiv(shader, GL_COMPILE_STATUS)
	if status == GL_FALSE:
		# Note that getting the error log is much simpler in Python than in C/C++
		# and does not require explicit handling of the string buffer
		strInfoLog = glGetShaderInfoLog(shader)
		print("Compilation failure for shader:\n" + str(strInfoLog))

def checkProgramError(shaderProg):
	status = glGetProgramiv(shaderProg, GL_LINK_STATUS)
	if status == GL_FALSE:
		strInfoLog = glGetProgramInfoLog(shaderProg)
		print("Shader link error:\n" + str(strInfoLog))

def createOrthoMat(min_x, max_x, min_y, max_y, min_z, max_z):
	# mat = [2/(max_x - min_x), 0, 0, -(max_x + min_x)/(max_x - min_x),
	# 	   0, 2/(max_y - min_y), 0, -(max_y + min_y)/(max_y - min_y),
	# 	   0, 0, 1/(max_z - min_z), -(max_z + min_z)/(max_z - min_z),
	# 	   0, 0, 0, 1]
	mat = numpy.array([
        [2.0 / (max_x - min_x), 0.0, 0.0, - (max_x + min_x) / (max_x - min_x)],
        [0.0, 2.0 / (max_y - min_y), 0.0, - (max_y + min_y) / (max_y - min_y)],
        [0.0, 0.0, -2.0 / (max_z - min_z), - (max_z + min_z) / (max_z - min_z)],
        [0.0, 0.0, 0.0, 1.0]
    ])
	return mat

# create hidden window
# Initialize glfw
if not glfw.init():
	sys.exit(1)
# Set window hint NOT visible
glfw.default_window_hints()
glfw.window_hint(glfw.VISIBLE, False)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, 1)

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(window_size[0], window_size[1], "hidden window", None, None)
if not window:
	glfw.terminate()
	sys.exit(1)

# Make the window's context current
glfw.make_context_current(window)
glfw.swap_interval(1)
glViewport(0, 0, window_size[0], window_size[1])

# set clear color
glClearColor(0.0, 0.0, 0.0, 0.0)

# make shaders
vert_src = """#version 330 core
layout (location = 0) in vec2 pos;
layout (location = 1) in float pH;
layout (location = 2) in float elevation;
layout (location = 3) in float pesticide;

uniform mat4 projection_matrix;
uniform vec3 color_low;
//uniform vec3 color_mid;
uniform vec3 color_high;

out vec3 frag_color;

void main()
{{
	//float low_t = clamp({0} * 2, 0, 1);
	//float high_t = clamp(({0} * 2) - 1, 0, 1);
	//frag_color = mix(color_low, color_mid, low_t);
	frag_color = mix(color_low, color_high, {0});
	gl_Position = projection_matrix * vec4(pos.x, pos.y, 0.0, 1.0);
}}
"""

frag_src = """#version 330 core
in vec3 frag_color;

void main()
{
	gl_FragColor = vec4(frag_color, 1.0); // color
}
"""

def createShader(shader_type, shader_src):
	shader = glCreateShader(shader_type)
	glShaderSource(shader, shader_src)
	glCompileShader(shader)
	checkShaderError(shader)
	return shader

def createShaderProgram(vert_shader, frag_shader):
	program = glCreateProgram()
	glAttachShader(program, vert_shader)
	glAttachShader(program, frag_shader)
	glLinkProgram(program)
	checkProgramError(program)
	return program

ph_src = vert_src.format("pH")
ph_vert_shader = createShader(OpenGL.GL.GL_VERTEX_SHADER, ph_src)
elevation_vert_shader = createShader(OpenGL.GL.GL_VERTEX_SHADER, vert_src.format("elevation"))
pesticide_vert_shader = createShader(OpenGL.GL.GL_VERTEX_SHADER, vert_src.format("pesticide"))

frag_shader = createShader(OpenGL.GL.GL_FRAGMENT_SHADER, frag_src)

ph_shader = createShaderProgram(ph_vert_shader, frag_shader)
elevation_shader = createShaderProgram(elevation_vert_shader, frag_shader)
pesticide_shader = createShaderProgram(pesticide_vert_shader, frag_shader)

# make vbo
vao = glGenVertexArrays(1)
ebo = glGenBuffers(1)
pos_vbo = glGenBuffers(1)
pH_vbo = glGenBuffers(1)
elevation_vbo = glGenBuffers(1)
pesticide_vbo = glGenBuffers(1)

glBindVertexArray(vao)

collated_positions = []
for pos in positions:
	collated_positions += pos
collated_positions = numpy.array(collated_positions, numpy.float32)

glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, indices, OpenGL.GL.GL_STATIC_DRAW)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, pos_vbo)
glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, collated_positions, OpenGL.GL.GL_STATIC_DRAW)
glVertexAttribPointer(0, 2, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 2*4, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, pH_vbo)
glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, numpy.asarray(ph, numpy.float32), OpenGL.GL.GL_STATIC_DRAW)
glVertexAttribPointer(1, 1, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 4, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, elevation_vbo)
glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, numpy.asarray(elevation, numpy.float32), OpenGL.GL.GL_STATIC_DRAW)
glVertexAttribPointer(2, 1, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 4, ctypes.c_void_p(0))
glEnableVertexAttribArray(2)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, pesticide_vbo)
glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, numpy.asarray(pesticide, numpy.float32), OpenGL.GL.GL_STATIC_DRAW)
glVertexAttribPointer(3, 1, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 4, ctypes.c_void_p(0))
glEnableVertexAttribArray(3)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

# uniforms
proj_mat = createOrthoMat(pos_min[0], pos_max[0], pos_min[1], pos_max[1], -2, 2).flatten()
print("Projection Matrix: " + str(proj_mat))

def writeImage(filename: str):
	image_buffer = glReadPixels(0, 0, window_size[0], window_size[1], OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)
	image = numpy.frombuffer(image_buffer, dtype=numpy.ubyte).reshape(window_size[0], window_size[1], 4)
	image = numpy.flip(image, 0)
	cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def renderImage(vao, indices_len, shader, proj_mat):
	# uniforms
	proj_mat_id = glGetUniformLocation(shader, "projection_matrix")
	low_color_id = glGetUniformLocation(shader, "color_low")
	# mid_color_id = glGetUniformLocation(shader, "color_mid")
	high_color_id = glGetUniformLocation(shader, "color_high")

	# render
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)
	glUseProgram(shader)
	glUniformMatrix4fv(proj_mat_id, 1, OpenGL.GL.GL_TRUE, proj_mat)
	glUniform3f(low_color_id, 0, 1, 0)
	# glUniform3f(mid_color_id, 1, 1, 0)
	glUniform3f(high_color_id, 1, 0, 0)
	glBindVertexArray(vao)
	glDrawElements(OpenGL.GL.GL_TRIANGLES, indices_len, OpenGL.GL.GL_UNSIGNED_INT, None)

def renderImageToFile(vao, indices_len, shader, proj_mat, filename: str):
	renderImage(vao, indices_len, shader, proj_mat)
	writeImage(filename)

# render
# while not glfw.window_should_close(window):
# 	glfw.poll_events()
# 	renderImage(vao, len(indices), ph_shader, proj_mat)
# 	glfw.swap_buffers(window)

renderImageToFile(vao, len(indices), ph_shader, proj_mat, './ph-map.png')
renderImageToFile(vao, len(indices), elevation_shader, proj_mat, './elevation-map.png')
renderImageToFile(vao, len(indices), pesticide_shader, proj_mat, './pesticide-map.png')

glfw.destroy_window(window)
glfw.terminate()