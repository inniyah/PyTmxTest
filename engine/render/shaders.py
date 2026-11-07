"""GLSL del motor nuevo.

Profundidad por píxel. En la proyección oblicua, la cercanía a la cámara de un punto del
mundo solo depende de su altura (Y) y de lo al sur que esté (Z), no de su posición este-oeste:

    cercanía = Y · sz + Z · sy        (sz = px por metro de fondo, sy = px por metro de altura)

Y como la posición en pantalla (hacia abajo, en píxeles de mundo) es  s = Z · sz − Y · sy,
la cercanía se puede escribir con la altura del píxel y su posición en pantalla:

    cercanía = Y · (sz + sy² / sz) + s · sy / sz

El shader toma solo la ALTURA del mapa de posición (posición dentro de la celda + origen de
la celda) y la posición en pantalla del propio píxel, que es exacta. Así dos superficies
coplanarias dan exactamente la misma profundidad en cada píxel aunque sus texels se muestreen
con redondeos distintos (tiles en lugares distintos del atlas, zoom fraccionario), y el
desempate es explícito: cada planta suma LEVEL_BIAS píxeles de cercanía, de modo que en
superficies coincidentes (la cara superior de un bloque y el suelo de la planta de encima)
gana siempre la planta superior. Es invisible (0.1 px) y mucho mayor que el ruido de
redondeo, incluso en mapas de kilómetros. Los sprites de personaje usarán la misma
fórmula, con la altura interpolada a lo alto del sprite.
"""

LEVEL_BIAS = 0.1     # píxeles de cercanía por planta (desempate de superficies coincidentes)

TILE_VERTEX = """
#version 330 core
layout(location = 0) in vec2 a_pos;       // píxeles de mundo (y hacia abajo)
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec3 a_origin;    // origen de la celda en metros (X este, Y arriba, Z sur)
layout(location = 3) in float a_alpha;

uniform mat4 u_projection;

out vec2 v_uv;
out vec3 v_origin;
out float v_alpha;
out float v_screen_y;             // posición en pantalla del píxel, en píxeles de mundo
out vec2 v_pos;

void main() {
    v_uv = a_uv;
    v_origin = a_origin;
    v_alpha = a_alpha;
    v_screen_y = a_pos.y;
    v_pos = a_pos;
    gl_Position = u_projection * vec4(a_pos, 0.0, 1.0);
}
"""

TILE_FRAGMENT = """
#version 330 core
in vec2 v_uv;
in vec3 v_origin;
in float v_alpha;
in float v_screen_y;
in vec2 v_pos;

uniform sampler2D u_color;
uniform sampler2D u_position;     // RGBA16: posición dentro de la celda, normalizada
uniform vec3 u_pos_min;           // rangos del mapa de posición (metros)
uniform vec3 u_pos_max;
uniform vec2 u_scale;             // (sz, sy): px por metro de fondo y de altura
uniform vec2 u_closeness_range;   // cercanía mínima y máxima del mundo
uniform float u_level_bias;       // px de cercanía por planta: en empates gana la planta superior
uniform float u_cell_h;
uniform float u_max_y;            // corte de plantas: no se dibuja por encima (metros)
uniform int u_show_depth;
uniform int u_height_tint;        // 1: aclarar con la altura para leer las plantas
uniform float u_cut_y;            // ocultar las celdas cuyo origen esté por encima (metros)...
uniform vec2 u_cut_center;        // ...dentro de este círculo (píxeles de mundo)...
uniform float u_cut_radius;       // ...de este radio; negativo = en todo el mapa

out vec4 frag;

void main() {
    vec4 color = texture(u_color, v_uv);
    float alpha = color.a * v_alpha;
    if (alpha < 0.01) discard;
    if (v_origin.y > u_max_y) discard;
    if (v_origin.y > u_cut_y && (u_cut_radius < 0.0 || distance(v_pos, u_cut_center) < u_cut_radius))
        discard;
    vec3 local = u_pos_min + texture(u_position, v_uv).rgb * (u_pos_max - u_pos_min);
    float height = v_origin.y + local.y;
    float sz = u_scale.x, sy = u_scale.y;
    float closeness = height * (sz + sy * sy / sz) + v_screen_y * sy / sz
                      + u_level_bias * floor(v_origin.y / u_cell_h + 0.5);
    float depth = clamp(1.0 - (closeness - u_closeness_range.x)
                        / (u_closeness_range.y - u_closeness_range.x), 0.0, 1.0);
    gl_FragDepth = depth;
    vec3 rgb = color.rgb;
    if (u_height_tint == 1) rgb *= clamp(0.78 + 0.11 * height / u_cell_h, 0.55, 1.35);
    frag = (u_show_depth == 1) ? vec4(vec3(1.0 - depth), 1.0) : vec4(rgb, alpha);
}
"""
