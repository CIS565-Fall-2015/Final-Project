#version 330

in vec3 N;
in vec3 v;

out vec4 fragColor;

void main(void)
{
   vec3 L = normalize(vec3(1, 5, 1));   
   float diffuse = max(dot(N,L), 0.0);
   fragColor = vec4(1.0f); //vec4(vec3(diffuse), 1.0);
}
