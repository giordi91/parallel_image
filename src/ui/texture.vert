#version 430

layout (location=0) in vec3 VertexPosition;
layout (location=1) in vec2 inTextureCoord;


uniform mat4 ortho;



out vec2 outTextureCoord;

void main()
{
    outTextureCoord = inTextureCoord;

    gl_Position = ortho* vec4(VertexPosition,1.0);
}
