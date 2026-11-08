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
MAX_LIGHTS = 1024    # luces puntuales en la vista (en una textura; las más cercanas, si hay más)
MAX_BACK = 32           # columnas con "pared del fondo" para el corte de fachada
MAX_CHUNK_LIGHTS = 32   # luces que recorre un píxel de un chunk: las que alcanzan ese chunk (si son
                        # más, ese chunk recorre todas las de la vista: nunca se pierde una luz)

# Iluminación (común a tiles y sprites), en espacio lineal: ambiente de cielo y suelo según la
# normal, un sol direccional y luces puntuales con caída suave hasta su radio.
LIGHTING_GLSL = """
uniform int u_lighting;
uniform vec3 u_sky;
uniform vec3 u_ground;
uniform vec3 u_sun_dir;           // hacia el sol, normalizado
uniform vec3 u_sun_color;
uniform sampler2D u_lights;       // fila i: (0, i) = xyz en metros y radio; (1, i) = color lineal
uniform int u_num_lights;         // cuántas luces recorrer...
uniform int u_use_list;           // ...1: las de u_light_list (las del chunk); 0: las primeras
uniform int u_light_list[@MAX_CHUNK_LIGHTS@];
uniform float u_lamps;            // factor de las luces puntuales (de día se notan menos)

vec3 light_at(vec3 world, vec3 n, float wrap) {
    vec3 light = mix(u_ground, u_sky, n.y * 0.5 + 0.5);
    light += u_sun_color * max((dot(n, u_sun_dir) + wrap) / (1.0 + wrap), 0.0);
    for (int k = 0; k < u_num_lights; k++) {
        int i = (u_use_list == 1) ? u_light_list[k] : k;
        vec4 p = texelFetch(u_lights, ivec2(0, i), 0);
        vec3 d = p.xyz - world;
        float dist = length(d);
        float r = p.w;
        if (dist < r) {
            float q = 1.0 - (dist * dist) / (r * r);
            float lambert = max((dot(n, d / max(dist, 1e-4)) + wrap) / (1.0 + wrap), 0.0);
            light += u_lamps * texelFetch(u_lights, ivec2(1, i), 0).rgb * q * q * lambert;
        }
    }
    return light;
}

vec3 to_linear(vec3 c) { return pow(c, vec3(2.2)); }
// Hombro suave: hasta 0.8 sin cambios; por encima se acerca a 1 sin recortar en seco
vec3 shoulder(vec3 c) {
    vec3 over = max(c - 0.8, 0.0);
    return min(c, vec3(0.8)) + 0.2 * (1.0 - exp(-over / 0.2));
}
vec3 to_srgb(vec3 c) { return pow(shoulder(max(c, 0.0)), vec3(1.0 / 2.2)); }
""".replace("@MAX_CHUNK_LIGHTS@", str(MAX_CHUNK_LIGHTS))

TILE_VERTEX = """
#version 330 core
layout(location = 0) in vec2 a_pos;       // píxeles de mundo (y hacia abajo)
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec3 a_origin;    // origen de la celda en metros (X este, Y arriba, Z sur)
layout(location = 3) in float a_alpha;
layout(location = 4) in float a_flip;
layout(location = 5) in float a_material;

uniform mat4 u_projection;

out vec2 v_uv;
out vec3 v_origin;
out float v_alpha;
out float v_flip;
flat out int v_material;
out float v_screen_y;             // posición en pantalla del píxel, en píxeles de mundo
out vec2 v_pos;

void main() {
    v_uv = a_uv;
    v_origin = a_origin;
    v_alpha = a_alpha;
    v_screen_y = a_pos.y;
    v_pos = a_pos;
    v_flip = a_flip;
    v_material = int(a_material + 0.5);
    gl_Position = u_projection * vec4(a_pos, 0.0, 1.0);
}
"""

# Corte de fachada (casa de muñecas), con una máscara de celdas que calcula la CPU: para cada
# celda (x, y) y cuatro plantas desde u_mask_level (canales R, G, B, A): 0 se dibuja, 0.5 se corta
# lo que esté a más de u_cut_keep sobre u_cut_feet (zócalo), 1 no se dibuja. Solo en interiores. Se mira la celda en
# la que está cada píxel, desplazado: en la planta del suelo cortado (u_mask_floor), 25 cm al sur
# (las paredes del borde sur caen en la fila de delante, las del norte se quedan); en las de
# encima, 25 cm al norte (la pared del fondo cae fuera y se conserva entera hacia arriba).
FACADE_GLSL = """
uniform int u_facade;
uniform sampler2D u_cut_mask;
uniform ivec2 u_mask_origin;       // celda (x, y) de la esquina de la máscara
uniform ivec2 u_mask_size;
uniform int u_mask_level;          // planta del canal R
uniform int u_mask_floor;          // planta del suelo que se conserva (zócalo)
uniform float u_cut_feet;          // altura de ese suelo (m)
uniform float u_cut_keep;          // zócalo que se conserva (m)
uniform float u_facade_cell_w;
const float FACADE_SHIFT = 0.25;
float facade_mask(vec2 xz, int level) {
    if (level < u_mask_floor) return 0.0;
    xz.y += (level == u_mask_floor) ? FACADE_SHIFT : -FACADE_SHIFT;
    ivec2 c = ivec2(floor(xz / u_facade_cell_w)) - u_mask_origin;
    int k = level - u_mask_level;
    if (k < 0 || k > 3 || c.x < 0 || c.y < 0 || c.x >= u_mask_size.x || c.y >= u_mask_size.y) return 0.0;
    vec4 m = texelFetch(u_cut_mask, c, 0);
    return k == 0 ? m.r : (k == 1 ? m.g : (k == 2 ? m.b : m.a));
}
"""


TILE_FRAGMENT = """
#version 330 core
in float v_flip;
flat in int v_material;
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
uniform sampler2D u_albedo;       // color sin luz (si el atlas lo tiene)
uniform sampler2D u_normal;
uniform float u_lod;              // nivel reducido del color, albedo y normales según el zoom
uniform int u_has_albedo;
uniform int u_has_normal;
uniform int u_material_tint;      // 1: teñir el suelo según su material (depuración)
uniform vec3 u_material_colors[16];
uniform float u_cell_w;
""" + LIGHTING_GLSL + FACADE_GLSL + """
out vec4 frag;

void main() {
    // Posición del atlas completo: profundidad y silueta exactas de cada píxel. El color, del
    // nivel reducido (de lejos, una media limpia en lugar de un texel suelto)
    // El texel del atlas completo se calcula aquí, con enteros, y el del nivel reducido es
    // (texel >> nivel): así lo contiene siempre. Con dos muestreos independientes de la GPU, en
    // una frontera exacta entre texels cada uno podía redondear hacia un lado (trazos oscuros).
    ivec2 texel = ivec2(floor(v_uv * vec2(textureSize(u_position, 0))));
    int lod = int(u_lod);
    ivec2 texel_lod = texel >> lod;
    vec4 position = texelFetch(u_position, texel, 0);
    if (position.a < 0.5) discard;
    vec4 color = texelFetch(u_color, texel_lod, lod);
    float alpha = v_alpha;
    if (v_origin.y > u_max_y) discard;
    if (v_origin.y > u_cut_y && (u_cut_radius < 0.0 || distance(v_pos, u_cut_center) < u_cut_radius))
        discard;
    vec3 local = u_pos_min + position.rgb * (u_pos_max - u_pos_min);
    if (u_facade == 1) {
        vec3 world = v_origin + vec3(v_flip > 0.5 ? u_cell_w - local.x : local.x, local.y, local.z);
        float m = facade_mask(world.xz, int(floor(v_origin.y / u_cell_h + 0.5)));
        if (m > 0.75 || (m > 0.4 && world.y > u_cut_feet + u_cut_keep)) discard;
    }
    float height = v_origin.y + local.y;
    float sz = u_scale.x, sy = u_scale.y;
    float closeness = height * (sz + sy * sy / sz) + v_screen_y * sy / sz
                      + u_level_bias * floor(v_origin.y / u_cell_h + 0.5);
    float depth = clamp(1.0 - (closeness - u_closeness_range.x)
                        / (u_closeness_range.y - u_closeness_range.x), 0.0, 1.0);
    gl_FragDepth = depth;
    vec3 rgb = color.rgb;
    if (u_lighting == 1) {
        vec3 base = (u_has_albedo == 1) ? texelFetch(u_albedo, texel_lod, lod).rgb : color.rgb;
        vec3 n = vec3(0.0, 1.0, 0.0);
        if (u_has_normal == 1) {
            n = texelFetch(u_normal, texel_lod, lod).rgb * 2.0 - 1.0;
            if (v_flip > 0.5) n.x = -n.x;          // tile volteado: la normal también
            n = normalize(n);
        }
        if (v_flip > 0.5) local.x = u_cell_w - local.x;
        if (u_material_tint == 1 && v_material > 0 && n.y > 0.5)
            base = mix(base, u_material_colors[v_material], 0.75);
        rgb = to_srgb(to_linear(base) * light_at(v_origin + local, n, 0.0));
    }
    if (u_lighting == 0 && u_material_tint == 1 && v_material > 0)
        rgb = mix(rgb, u_material_colors[v_material], 0.75);
    if (u_height_tint == 1) rgb *= clamp(0.78 + 0.11 * height / u_cell_h, 0.55, 1.35);
    frag = (u_show_depth == 1) ? vec4(vec3(1.0 - depth), 1.0) : vec4(rgb, alpha);
}
"""

