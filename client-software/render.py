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
elevation_min = min(pesticide)
elevation_max = max(pesticide)
pos_min = [min_idx(positions, 0), min_idx(positions, 1)]
pos_max = [max_idx(positions, 0), max_idx(positions, 1)]

print("Min pH:", ph_min, "Max pH:", ph_max, "Min Pesticide:", pesticide_min, "Max Pesticide:", pesticide_max, "Min Elevation:", elevation_min, "Max Elevation:", elevation_max, "Pos Min:", pos_min, "Pos Max:", pos_max)

window_size = [round(pos_max[0] - pos_min[0]), round(pos_max[1] - pos_min[1])]

print("Triangulating.")
triangulation = Delaunay(positions)
# generate indices from delaunay info
indices = triangulation.simplices.flatten()
print(indices)

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
# glfw.window_hint(glfw.VISIBLE, False)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
#glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, 1)

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
glClearColor(0.0, 0.0, 0.0, 1.0)

# make shaders
vert_pH_src = """#version 330 core
layout (location = 0) in vec2 pos;
layout (location = 1) in float pH;
layout (location = 2) in float elevation;
layout (location = 3) in float pesticide;

uniform mat4 projectionMatrix;
uniform vec3 color_low;
uniform vec3 color_high;

out vec3 frag_color;

void main()
{
	frag_color = mix(color_low, color_high, pH);
	gl_Position = projectionMatrix * vec4(pos.x, pos.y, 0.0, 1.0); // projMat*
}
"""

frag_src = """#version 330 core
in vec3 frag_color;

void main()
{
	gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); // color
}
"""

ph_vert_shader = glCreateShader(OpenGL.GL.GL_VERTEX_SHADER)
glShaderSource(ph_vert_shader, vert_pH_src)
glCompileShader(ph_vert_shader)
checkShaderError(ph_vert_shader)

frag_shader = glCreateShader(OpenGL.GL.GL_FRAGMENT_SHADER)
glShaderSource(frag_shader, frag_src)
glCompileShader(frag_shader)
checkShaderError(frag_shader)

ph_shader = glCreateProgram()
glAttachShader(ph_shader, ph_vert_shader)
glAttachShader(ph_shader, frag_shader)
glLinkProgram(ph_shader)
checkProgramError(ph_shader)

# make vbo
vao = glGenVertexArrays(1)
ebo = glGenBuffers(1)
pos_vbo = glGenBuffers(1)
# pH_vbo = glGenBuffers(1)
# elevation_vbo = glGenBuffers(1)
# pesticide_vbo = glGenBuffers(1)

glBindVertexArray(vao)

# collated_positions = numpy.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5], numpy.float32)
collated_positions = []
for pos in positions:
	collated_positions += pos
collated_positions = numpy.array(collated_positions, numpy.float32)

# indices = numpy.array([ 0, 1, 3, 1, 2, 3], numpy.uint32)

glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, indices, OpenGL.GL.GL_STATIC_DRAW)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, pos_vbo)
glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, collated_positions, OpenGL.GL.GL_STATIC_DRAW)
glVertexAttribPointer(0, 2, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 2*4, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, pH_vbo)
# glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, numpy.asarray(ph), OpenGL.GL.GL_STATIC_DRAW)
# glVertexAttribPointer(1, 1, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, numpy.dtype(float).itemsize, 0)
# glEnableVertexAttribArray(1)

# glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, elevation_vbo)
# glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, numpy.asarray(elevation), OpenGL.GL.GL_STATIC_DRAW)
# glVertexAttribPointer(2, 1, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, numpy.dtype(float).itemsize, 0)
# glEnableVertexAttribArray(2)

# glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, pesticide_vbo)
# glBufferData(OpenGL.GL.GL_ARRAY_BUFFER, numpy.asarray(pesticide), OpenGL.GL.GL_STATIC_DRAW)
# glVertexAttribPointer(3, 1, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, numpy.dtype(float).itemsize, 0)
# glEnableVertexAttribArray(3)

glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

# uniforms
projMatLocPH = glGetUniformLocation(ph_shader, "projectionMatrix")
projMat = createOrthoMat(pos_min[0] - 10, pos_max[0] + 10, pos_min[1] - 10, pos_max[1] + 10, -2, 2).flatten()
print("Projection Matrix Location: " + str(projMatLocPH) + " Matrix: " + str(projMat))
# lowColorLocPH = glGetUniformLocation(ph_shader, "color_low")
# highColorLocPH = glGetUniformLocation(ph_shader, "color_high")

# render
while not glfw.window_should_close(window):
	glClearColor(0.0, 0.0, 0.0, 1.0)
	glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)
	glUseProgram(ph_shader)
	glUniformMatrix4fv(projMatLocPH, 1, OpenGL.GL.GL_TRUE, projMat)
	# glUniform3f(lowColorLocPH, 1, 0, 0)
	# glUniform3f(highColorLocPH, 0, 0, 1)
	glBindVertexArray(vao)
	glDrawElements(OpenGL.GL.GL_TRIANGLES, len(indices), OpenGL.GL.GL_UNSIGNED_INT, None)
	glfw.swap_buffers(window)
	glfw.poll_events()
# write map

# image_buffer = glReadPixels(0, 0, window_size[0], window_size[1], OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)
# image = numpy.frombuffer(image_buffer, dtype=numpy.uint8).reshape(window_size[0], window_size[1], 4)

# cv2.imwrite("./ph-map.png", image)

glfw.destroy_window(window)
glfw.terminate()