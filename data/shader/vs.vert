#version 330

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 colour;

uniform mat4 projection;

out vec4 f_colour;

void main() {
  f_colour = colour;

  gl_Position = projection * vec4(position, 0.0, 1.0);
}
