#version 330

out vec3 N;
out vec3 v;

uniform mat4 u_projMatrix;

layout(location = 0) in vec4 Position;
layout(location = 1) in vec4 Normal;

void main(void)
{
    vec4 v2 = u_projMatrix * Position;
    v = vec3(Position);//vec3(v2 / v2.w);
    N = vec3(Normal);
}