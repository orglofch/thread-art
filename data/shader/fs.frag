#version 330

in vec3 f_colour;

out vec4 out_colour;

void main() {
  out_colour = vec4(f_colour, 1.0);
}
