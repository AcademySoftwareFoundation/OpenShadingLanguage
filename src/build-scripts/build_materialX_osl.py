'''
Generate compilable .osl files from .mx templates

Adam Martinez
'''

import os
import sys
import re

MX_SOURCE = '../shaders/MaterialX'
OSL_DEST = '../../dist/linux64/shaders/MaterialX'

MX_SOURCE_EXT = 'mx'

OSLC_CMD = 'oslc '

SHADER_TYPES = {
    'FLOAT': 'float',
    'COLOR': 'color',
    'COLOR2': 'color2',
    'COLOR4': 'color4',
    'VECTOR': 'vector',
    'VECTOR2': 'vector2',
    'VECTOR4': 'vector4'
}

ALL_TYPES = ['FLOAT', 'COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4']
EXTRA_TYPES = ['MATRIX44', 'MATRIX33', 'STRING', 'FILENAME', 'BOOL', 'INT', 'SURFACESHADER']

BUILD_DICT = {
    'mx_absval': ALL_TYPES,
    'mx_add': ALL_TYPES + ['SURFACESHADER'],
    'mx_add_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_ambientocclusion': ['FLOAT'],
    'mx_bitangent': ['VECTOR'],
    'mx_blur': ALL_TYPES,
    'mx_burn': ['FLOAT', 'COLOR', 'COLOR2', 'COLOR4'],
    'mx_cellnoise2d': ['FLOAT'],
    'mx_cellnoise3d': ['FLOAT'],
    'mx_clamp': ALL_TYPES,
    'mx_clamp_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_compare': ALL_TYPES,
    'mx_constant': ALL_TYPES + ['MATRIX44', 'MATRIX33', 'STRING', 'FILENAME', 'BOOL', 'INT'],
    'mx_crossproduct': ['VECTOR'],
    'mx_dodge': ['FLOAT', 'COLOR', 'COLOR2', 'COLOR4'],
    'mx_dotproduct': ['VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_dot': ALL_TYPES + ['MATRIX44', 'MATRIX33', 'STRING', 'FILENAME', 'BOOL', 'INT'],
    'mx_floor': ALL_TYPES,
    'mx_frame': ['FLOAT'],
    'mx_geomattrvalue': ALL_TYPES + ['BOOL', 'STRING', 'INT'],
    'mx_geomcolor': ['FLOAT', 'COLOR', 'COLOR2', 'COLOR4'],
    'mx_heighttonormal': ['VECTOR'],
    'mx_hueshift': ['COLOR', 'COLOR4'],
    'mx_image': ALL_TYPES,
    'mx_max': ALL_TYPES,
    'mx_min': ALL_TYPES,
    'mx_overlay': ['FLOAT', 'COLOR', 'COLOR2', 'COLOR4'],
    'mx_screen': ['FLOAT', 'COLOR', 'COLOR2', 'COLOR4'],
    'mx_inside': ['FLOAT','COLOR', 'COLOR2', 'COLOR4'],
    'mx_outside': ['FLOAT','COLOR', 'COLOR2', 'COLOR4'],
    'mx_disjointover': ['COLOR2', 'COLOR4'],
    'mx_in': ['COLOR2', 'COLOR4'],
    'mx_mask': ['COLOR2', 'COLOR4'],
    'mx_matte': ['COLOR2', 'COLOR4'],
    'mx_out': ['COLOR2', 'COLOR4'],
    'mx_over': ['COLOR2', 'COLOR4'],
    'mx_mix': ALL_TYPES + ['SURFACESHADER'],
    'mx_fractal3d': ALL_TYPES,
    'mx_fractal3d_fa':['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_contrast': ALL_TYPES,
    'mx_contrast_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_smoothstep': ALL_TYPES,
    'mx_smoothstep_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_divide': ALL_TYPES,
    'mx_divide_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_exponent': ALL_TYPES,
    'mx_exponent_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_invert': ALL_TYPES,
    'mx_invert_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_luminance': ['COLOR', 'COLOR4'],
    'mx_magnitude': ['VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_max': ALL_TYPES,
    'mx_max_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_min': ALL_TYPES,
    'mx_min_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_modulo': ALL_TYPES,
    'mx_modulo_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_multiply': ALL_TYPES,
    'mx_multiply_float':['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_noise2d': ALL_TYPES,
    'mx_noise2d_fa':['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_noise3d': ALL_TYPES,
    'mx_noise3d_fa':['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_normal': ['VECTOR'],
    'mx_normalize': ['VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_pack': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_pack_cf': ['COLOR4'],
    'mx_pack_cc': ['COLOR4'],
    'mx_pack_vf': ['COLOR4'],
    'mx_pack_vv': ['COLOR4'],
    'mx_position': ['VECTOR'],
    'mx_premult': ['COLOR', 'COLOR2', 'COLOR4'],
    'mx_ramp4': ALL_TYPES,
    'mx_ramplr': ALL_TYPES,
    'mx_ramptb': ALL_TYPES,
    'mx_remap': ALL_TYPES,
    'mx_remap_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_rotate2d': ['VECTOR2'],
    'mx_saturate': ['COLOR', 'COLOR4'],
    'mx_scale': ['VECTOR', 'VECTOR2'],
    'mx_splitlr': ALL_TYPES,
    'mx_splittb': ALL_TYPES,
    'mx_subtract': ALL_TYPES,
    'mx_subtract_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_swizzle_float': ['COLOR', 'COLOR2', 'COLOR4', 'VECTOR', 'VECTOR2', 'VECTOR4'],
    'mx_swizzle_color': ALL_TYPES,
    'mx_swizzle_color2': ALL_TYPES,
    'mx_swizzle_color4': ALL_TYPES,
    'mx_swizzle_vector': ALL_TYPES,
    'mx_swizzle_vector2': ALL_TYPES,
    'mx_swizzle_vector4': ALL_TYPES,
    'mx_switch': ALL_TYPES,
    'mx_tangent': ['VECTOR'],
    'mx_texcoord': ['VECTOR', 'VECTOR2'],
    'mx_time': ['FLOAT'],
    'mx_triplanarprojection': ALL_TYPES,
    'mx_unpremult': ['COLOR', 'COLOR2', 'COLOR4'],
    'mx_mult_surfaceshader': ['COLOR', 'FLOAT']
}

for shader, shader_types in BUILD_DICT.items():

    mx_filename = '%s.%s' % (shader, MX_SOURCE_EXT)
    mx_filepath = os.path.join(MX_SOURCE, mx_filename)
    try:
        mx_file = open(mx_filepath, 'r')
    except:
        print('ERROR: %s not found' % mx_filename)
        continue
    mx_code = mx_file.read()
    mx_file.close()
    for var_type in shader_types:
        osl_shadername = '%s_%s' % (shader, SHADER_TYPES[var_type])
        osl_filename = '%s.osl' % (osl_shadername)
        osl_filepath = os.path.join(OSL_DEST, osl_filename)
        print('Building %s' % osl_filename)
        osl_code = mx_code.replace('SHADER_NAME(%s)' % shader, osl_shadername)
        osl_code = osl_code.replace('#include \"mx_types.h\"', '#define %s 1\n#include \"mx_types.h\"' % var_type)
        osl_code = re.sub(r'\bTYPE\b', SHADER_TYPES[var_type], osl_code)
        osl_file = open(osl_filepath, 'w')
        osl_file.write(osl_code)
        osl_file.close()
