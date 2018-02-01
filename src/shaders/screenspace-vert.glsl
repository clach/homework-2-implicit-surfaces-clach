#version 300 es

precision highp float;

in vec4 vs_Pos;
out vec4 fs_Pos;

uniform vec2 u_AspectRatio;

void main() {
	fs_Pos = vs_Pos;
	gl_Position = vs_Pos;
}
