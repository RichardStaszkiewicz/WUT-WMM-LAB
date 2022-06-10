#version 330

out vec4 f_color;

in vec3 position;
in vec3 normal;

uniform vec3 view_position;
uniform vec3 object_color;
uniform vec3 light_position;
uniform vec3 light_color;

uniform float shininess_param;
uniform float ambient_param;
uniform float specular_param;
uniform float diffuse_param;

void main() {
    // --- Ambient --- //
    vec3 ambient_vec = ambient_param * light_color;

    // --- Diffuse --- //
    vec3 light_direction = normalize(light_position - position);
    vec3 diffuse_vec = diffuse_param * max(dot(normal, light_direction), 0.0) * light_color;

    // --- Specular --- //
    vec3 view_direction = normalize(view_position - position);
    vec3 reflection_direction = reflect(-light_direction, normal);
    vec3 specular_vec = specular_param
        * pow(max(dot(view_direction, reflection_direction), 0.0), shininess_param)
        * light_color;

    // Sum of ambiend diffuse and specular is the result
    vec3 shading = (ambient_vec + diffuse_vec + specular_vec) * object_color;

    f_color = vec4(shading, 1.0);
}
