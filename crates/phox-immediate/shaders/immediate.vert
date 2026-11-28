#version 450

// Immediate-mode 2D geometry: positions in logical pixels (origin top left), sRGB colours with straight alpha.

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 viewport;      // logical size of the target
    uint linear_output; // the target encodes sRGB on write, so colours are converted to linear here
    uint padding;
} pc;

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
    gl_Position = vec4(in_position / pc.viewport * 2.0 - 1.0, 0.0, 1.0);
    out_color = pc.linear_output != 0u ? vec4(srgb_to_linear(in_color.rgb), in_color.a) : in_color;
}
