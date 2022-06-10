#version 330

uniform mat4 projection;
uniform mat4 view;

in vec3 in_position;
in vec3 in_normal;

out vec3 position;
out vec3 normal;

void main() {
    position = in_position;
    normal = normalize(in_normal);
    gl_Position = projection * view * vec4(in_position, 1.0);
}
