#version 330

uniform mat4 pvm;
in vec3 in_position;

void main() {
    gl_Position = pvm * vec4(in_position, 1.0);
}
