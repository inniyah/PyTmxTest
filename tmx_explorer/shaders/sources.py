"""
Shader source code for OpenGL rendering
"""

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec4 aColor;
layout (location = 3) in float aDepth;
out vec2 TexCoord;
out vec4 Color;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, aDepth, 1.0);
    TexCoord = aTexCoord;
    Color = aColor;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
in vec4 Color;
out vec4 FragColor;
uniform sampler2D texture0;
void main() {
    vec4 texColor = texture(texture0, TexCoord);
    FragColor = texColor * Color;
    if (FragColor.a < 0.01) discard;
}
"""

SIMPLE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor;
out vec4 Color;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    Color = aColor;
}
"""

SIMPLE_FRAGMENT_SHADER = """
#version 330 core
in vec4 Color;
out vec4 FragColor;
void main() { FragColor = Color; }
"""
