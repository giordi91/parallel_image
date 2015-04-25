#version 430

layout (location=0) out vec4 FragColor;

in vec2 outTextureCoord;
uniform sampler2D Tex1;

void main() {
    //vec2 flipped_texcoord = vec2(outTextureCoord.x, 1.0 - outTextureCoord.y);
    FragColor = texture(Tex1, outTextureCoord.st);
}
