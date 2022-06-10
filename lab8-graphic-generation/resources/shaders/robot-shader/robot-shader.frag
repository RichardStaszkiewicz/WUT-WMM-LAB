#version 330

uniform vec3 color;
out vec4 f_color;

void main() {
    f_color = vec4(normalize(color), 1.0);
}
