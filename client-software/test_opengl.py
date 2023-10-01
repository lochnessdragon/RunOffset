# test opengl
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
import glfw
import numpy
import ctypes

window_size = [500, 500]

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

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(window_size[0], window_size[1], "hidden window", None, None)
if not window:
	glfw.terminate()
	sys.exit(1)

# Make the window's context current
glfw.make_context_current(window)
glfw.swap_interval(1)
glViewport(0, 0, window_size[0], window_size[1])
glDisable(OpenGL.GL.GL_CULL_FACE)

# set clear color

# make shaders
vert_pH_src = """#version 330 core
layout (location = 0) in vec2 pos;

void main()
{
	gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
}
"""

frag_src = """#version 330 core
out vec4 color;

void main()
{
	color = vec4(1.0, 1.0, 1.0, 1.0); // color
}
"""

# make vao
positions = numpy.array([0.5, 0.5, 0.0, 0.5, -0.5, 0.0, -0.5, -0.5, 0.0, -0.5, 0.5, 0.0], numpy.float32)
indices = numpy.array([ 0, 1, 3, 1, 2, 3], numpy.uint32)

vao = glGenVertexArrays(1)
ebo = glGenBuffers(1)
pos_vbo = VBO(positions, usage='GL_STATIC_DRAW')
pos_vbo.create_buffers()

glBindVertexArray(vao)

pos_vbo.bind()
pos_vbo.copy_data()
glVertexAttribPointer(0, 2, OpenGL.GL.GL_FLOAT, OpenGL.GL.GL_FALSE, 2*4, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

glBindBuffer(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER, indices, OpenGL.GL.GL_STATIC_DRAW)

pos_vbo.unbind()
glBindVertexArray(0)

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

glDeleteShader(ph_vert_shader)
glDeleteShader(frag_shader)

# render
while not glfw.window_should_close(window):
	glClearColor(0.0, 0.0, 0.0, 1.0)
	glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)
	glUseProgram(ph_shader)
	glBindVertexArray(vao)
	glDrawElements(OpenGL.GL.GL_TRIANGLES, len(indices), OpenGL.GL.GL_UNSIGNED_INT, None)
	glfw.poll_events()
	glfw.swap_buffers(window)

glfw.destroy_window(window)
glfw.terminate()