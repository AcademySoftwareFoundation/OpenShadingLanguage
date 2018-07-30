#!/usr/bin/env python
'''
Generate compilable .osl files from .mx templates

Adam Martinez

TODO:  Some functionalization in place, can it be expanded?
       Add ability to specify shader and shader type to compile to osl
       Is there a more compact representation of the BUILD_DICT we can employ?
'''

import os
import sys
import re
import argparse
from subprocess import call

# SHADER_TYPES map preprocessor flags to osl type declarations
SHADER_TYPES = {
    'float': 'float',
    'color': 'color',
    'color2': 'color2',
    'color4': 'color4',
    'vector': 'vector',
    'vector2': 'vector2',
    'vector4': 'vector4',
    'surfaceshader': 'closure color',
    'matrix44': 'matrix',
    'matrix33': 'matrix',
    'string': 'string',
    'filename': 'string',
    'bool': 'bool',
    'int': 'int',
}

replacements = {
    "float" : (
        ('TYPE_ZERO_POINT_FIVE',   '0.5'),
        ('TYPE_ZERO',              '0'),
        ('TYPE_ONE',               '1'),
        ('TYPE_DEFAULT_IN',        '0'),
        ('TYPE_DEFAULT_OUT',       '0'),
        ('TYPE_DEFAULT_CHANNELS',  '"r"'),
        ('TYPE_STR',               '"Float"'),
        ('TYPE_SUFFIX',            'float'),
        ('TYPE',                   'float'),
    ),
   "color" : (
        ('TYPE_ZERO_POINT_FIVE',   '0.5'),
        ('TYPE_ZERO',              '0'),
        ('TYPE_ONE',               '1'),
        ('TYPE_DEFAULT_IN',        '0'),
        ('TYPE_DEFAULT_OUT',       '0'),
        ('TYPE_DEFAULT_CHANNELS',  '"rgb"'),
        ('TYPE_STR',               '"Color"'),
        ('TYPE_SUFFIX',            'color'),
        ('TYPE',                   'color'),
    ),
    "vector" : (
        ('TYPE_ZERO_POINT_FIVE',   '0.5'),
        ('TYPE_ZERO',              '0'),
        ('TYPE_ONE',               '1'),
        ('TYPE_DEFAULT_IN',        '0'),
        ('TYPE_DEFAULT_OUT',       '0'),
        ('TYPE_DEFAULT_CHANNELS',  '"xyz"'),
        ('TYPE_STR',               '"Vector"'),
        ('TYPE_SUFFIX',            'vector'),
        ('TYPE',                   'vector'),
    ),
    "color2" : (
        ('TYPE_ZERO_POINT_FIVE',   '{0.5,0.5}'),
        ('TYPE_ZERO',              '{0,0}'),
        ('TYPE_ONE',               '{1,1}'),
        ('TYPE_DEFAULT_IN',        '{0,0}'),
        ('TYPE_DEFAULT_OUT',       '{0,0}'),
        ('TYPE_DEFAULT_CHANNELS',  '"rg"'),
        ('TYPE_STR',               '"Color2"'),
        ('TYPE_SUFFIX',            'color2'),
        ('TYPE',                   'color2'),
    ),
    "vector2" : (
        ('TYPE_ZERO_POINT_FIVE',   '{0.5,0.5}'),
        ('TYPE_ZERO',              '{0,0}'),
        ('TYPE_ONE',               '{1,1}'),
        ('TYPE_DEFAULT_IN',        '{0,0}'),
        ('TYPE_DEFAULT_OUT',       '{0,0}'),
        ('TYPE_DEFAULT_CHANNELS',  '"xy"'),
        ('TYPE_STR',               '"Vector2"'),
        ('TYPE_SUFFIX',            'vector2'),
        ('TYPE',                   'vector2'),
    ),
    "color4" : (
        ('TYPE_ZERO_POINT_FIVE',   '{color(0.5,0.5,0.5), 0.5}'),
        ('TYPE_ZERO',              '{color(0,0,0), 0}'),
        ('TYPE_ONE',               '{color(1,1,1), 1}'),
        ('TYPE_DEFAULT_IN',        '{color(0,0,0), 0}'),
        ('TYPE_DEFAULT_OUT',       '{color(0,0,0), 0}'),
        ('TYPE_DEFAULT_CHANNELS',  '"rgba"'),
        ('TYPE_STR',               '"Color4"'),
        ('TYPE_SUFFIX',            'color4'),
        ('TYPE',                   'color4'),
    ),
    "vector4" : (
        ('TYPE_ZERO_POINT_FIVE',   '{0.5,0.5,0.5,0.5}'),
        ('TYPE_ZERO',              '{0,0,0,0}'),
        ('TYPE_ONE',               '{1,1,1,1}'),
        ('TYPE_DEFAULT_IN',        '{0,0,0,0}'),
        ('TYPE_DEFAULT_OUT',       '{0,0,0,0}'),
        ('TYPE_DEFAULT_CHANNELS',  '"xyzw"'),
        ('TYPE_STR',               '"Vector4"'),
        ('TYPE_SUFFIX',            'vector4'),
        ('TYPE',                   'vector4'),
    ),
    "matrix44" : (
        ('TYPE_ZERO_POINT_FIVE',   'matrix(0.5,0,0,0, 0,0.5,0,0, 0,0,0.5,0, 0,0,0,0.5)'),
        ('TYPE_ZERO',              'matrix(0)'),
        ('TYPE_ONE',               'matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)'),
        ('TYPE_DEFAULT_IN',        'matrix(0)'),
        ('TYPE_DEFAULT_OUT',       'matrix(0)'),
        ('TYPE_DEFAULT_CHANNELS',  '"xyzw"'),
        ('TYPE_STR',               '"Matrix44"'),
        ('TYPE_SUFFIX',            'matrix44'),
        ('TYPE',                   'matrix'),
    ),
    "matrix33" : (
        ('TYPE_ZERO_POINT_FIVE',   'matrix(0.5,0,0,0, 0,0.5,0,0, 0,0,0.5,0, 0,0,0,0)'),
        ('TYPE_ZERO',              'matrix(0)'),
        ('TYPE_ONE',               'matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,0)'),
        ('TYPE_DEFAULT_IN',        'matrix(0)'),
        ('TYPE_DEFAULT_OUT',       'matrix(0)'),
        ('TYPE_DEFAULT_CHANNELS',  '"xyzw"'),
        ('TYPE_STR',               '"Matrix33"'),
        ('TYPE_SUFFIX',            'matrix33'),
        ('TYPE',                   'matrix'),
    ),
    "int" : (
        ('TYPE_ZERO_POINT_FIVE',   '1'),
        ('TYPE_ZERO',              '0'),
        ('TYPE_ONE',               '1'),
        ('TYPE_DEFAULT_IN',        '0'),
        ('TYPE_DEFAULT_OUT',       '0'),
        ('TYPE_DEFAULT_CHANNELS',  '"x"'),
        ('TYPE_STR',               '"int"'),
        ('TYPE_SUFFIX',            'int'),
        ('TYPE',                   'int'),
    ),
    "bool" : (
        ('TYPE_ZERO_POINT_FIVE',   '1'),
        ('TYPE_ZERO',              '0'),
        ('TYPE_ONE',               '1'),
        ('TYPE_DEFAULT_IN',        '0'),
        ('TYPE_DEFAULT_OUT',       '0'),
        ('TYPE_DEFAULT_CHANNELS',  '"x"'),
        ('TYPE_STR',               '"bool"'),
        ('TYPE_SUFFIX',            'bool'),
        ('TYPE',                   'int'),
    ),
    "string" : (
        ('TYPE_ZERO_POINT_FIVE',   '"zero point five"'),
        ('TYPE_ZERO',              '"zero"'),
        ('TYPE_ONE',               '"one"'),
        ('TYPE_DEFAULT_IN',        '"default"'),
        ('TYPE_DEFAULT_OUT',       '"default"'),
        ('TYPE_DEFAULT_CHANNELS',  '"a"'),
        ('TYPE_STR',               '"string"'),
        ('TYPE_SUFFIX',            'string'),
        ('TYPE',                   'string'),
    ),
  "filename" :  (
        ('TYPE_ZERO_POINT_FIVE',   '"zero point five"'),
        ('TYPE_ZERO',              '"zero"'),
        ('TYPE_ONE',               '"one"'),
        ('TYPE_DEFAULT_IN',        '"default"'),
        ('TYPE_DEFAULT_OUT',       '"default"'),
        ('TYPE_DEFAULT_CHANNELS',  '"a"'),
        ('TYPE_STR',               '"filename"'),
        ('TYPE_SUFFIX',            'filename'),
        ('TYPE',                   'string'),
    ),
    "surfaceshader" : (
        ('TYPE_ZERO_POINT_FIVE',   '0'),
        ('TYPE_ZERO',              '0'),
        ('TYPE_ONE',               '1'),
        ('TYPE_DEFAULT_IN',        '0'),
        ('TYPE_DEFAULT_OUT',       '0'),
        ('TYPE_DEFAULT_CHANNELS',  '"a"'),
        ('TYPE_STR',               '"surfaceshader"'),
        ('TYPE_SUFFIX',            'surfaceshader'),
        ('TYPE',                   'closure color'),
    )
}

ALL_COLVEC = ['color', 'color2', 'color4', 'vector', 'vector2', 'vector4']
ALL_TYPES = ['float', 'color', 'color2', 'color4', 'vector', 'vector2', 'vector4']
EXTRA_TYPES = ['matrix44', 'matrix33', 'string', 'filename', 'bool', 'int', 'surfaceshader']

BUILD_DICT = {
    'mx_absval': ALL_TYPES,
    'mx_acos' : ALL_TYPES,
    'mx_add': ALL_TYPES + ['surfaceshader'],
    'mx_add_float': ALL_COLVEC,
    'mx_asin' : ALL_TYPES,
    'mx_atan2' : ALL_TYPES,
    'mx_atan2_float' : ALL_TYPES,
    'mx_ambientocclusion': ['float'],
    'mx_bitangent': ['vector'],
    'mx_blur': ALL_TYPES,
    'mx_burn': ['float', 'color', 'color2', 'color4'],
    'mx_ceil' : ALL_TYPES,
    'mx_cellnoise2d': ['float'],
    'mx_cellnoise3d': ['float'],
    'mx_clamp': ALL_TYPES,
    'mx_clamp_float': ALL_COLVEC,
    'mx_compare': ALL_TYPES,
    'mx_constant': ALL_TYPES + ['matrix44', 'matrix33', 'string', 'filename', 'bool', 'int'],
    'mx_cos': ALL_TYPES,
    'mx_crossproduct': ['vector'],
    'mx_determinant' : ['matrix44', 'matrix33'],
    'mx_dodge': ['float', 'color', 'color2', 'color4'],
    'mx_dotproduct': ['vector', 'vector2', 'vector4'],
    'mx_dot': ALL_TYPES + ['matrix44', 'matrix33', 'string', 'filename', 'bool', 'int', 'surfaceshader'],
    'mx_exp' : ALL_TYPES + ['int'],
    'mx_extract' : ALL_COLVEC,
    'mx_floor': ALL_TYPES,
    'mx_frame': ['float'],
    'mx_geomattrvalue': ALL_TYPES + ['bool', 'string', 'int'],
    'mx_geomcolor': ['float', 'color', 'color2', 'color4'],
    'mx_heighttonormal': ['vector'],
    'mx_hsvadjust' : ['color', 'color4'],
    'mx_hsvtorgb' : ['color', 'color4'],
    'mx_hueshift': ['color', 'color4'],   # DEPRECATED in MX 1.36
    'mx_image': ALL_TYPES,
    'mx_ln' : ALL_TYPES + ['int'],
    'mx_max': ALL_TYPES,
    'mx_min': ALL_TYPES,
    'mx_overlay': ['float', 'color', 'color2', 'color4'],
    'mx_screen': ['float', 'color', 'color2', 'color4'],
    'mx_inside': ['float','color', 'color2', 'color4'],
    'mx_outside': ['float','color', 'color2', 'color4'],
    'mx_disjointover': ['color2', 'color4'],
    'mx_in': ['color2', 'color4'],
    'mx_mask': ['color2', 'color4'],
    'mx_matte': ['color2', 'color4'],
    'mx_out': ['color2', 'color4'],
    'mx_over': ['color2', 'color4'],
    'mx_mix': ALL_TYPES + ['surfaceshader'],
    'mx_fractal3d': ALL_TYPES,
    'mx_fractal3d_fa':ALL_COLVEC,
    'mx_contrast': ALL_TYPES,
    'mx_contrast_float': ALL_COLVEC,
    'mx_smoothstep': ALL_TYPES,
    'mx_smoothstep_float': ALL_COLVEC,
    'mx_divide': ALL_TYPES + ['matrix44', 'matrix33'],
    'mx_divide_float': ALL_COLVEC,
    'mx_power': ALL_TYPES,
    'mx_power_float': ALL_COLVEC,
    'mx_invert': ALL_TYPES,
    'mx_invert_float': ALL_COLVEC,
    'mx_luminance': ['color', 'color4'],
    'mx_magnitude': ['vector', 'vector2', 'vector4'],
    'mx_max': ALL_TYPES,
    'mx_max_float': ALL_COLVEC,
    'mx_min': ALL_TYPES,
    'mx_min_float': ALL_COLVEC,
    'mx_modulo': ALL_TYPES,
    'mx_modulo_float': ALL_COLVEC,
    'mx_multiply': ALL_TYPES + ['matrix44', 'matrix33'],
    'mx_multiply_float':ALL_COLVEC,
    'mx_noise2d': ALL_TYPES,
    'mx_noise2d_fa':ALL_COLVEC,
    'mx_noise3d': ALL_TYPES,
    'mx_noise3d_fa':ALL_COLVEC,
    'mx_normal': ['vector'],
    'mx_normalize': ['vector', 'vector2', 'vector4'],
    'mx_combine': ALL_COLVEC,
    'mx_combine_cf': ['color4'],
    'mx_combine_cc': ['color4'],
    'mx_combine_vf': ['vector4'],
    'mx_combine_vv': ['vector4'],
    'mx_position': ['vector'],
    'mx_premult': ['color', 'color2', 'color4'],
    'mx_ramp4': ALL_TYPES,
    'mx_ramplr': ALL_TYPES,
    'mx_ramptb': ALL_TYPES,
    'mx_remap': ALL_TYPES,
    'mx_remap_float': ALL_COLVEC,
    'mx_rgbtohsv' : ['color', 'color4'],
    'mx_rotate': ['vector', 'vector2'],   # Deprecated in MX 1.36
    'mx_rotate2d': ['vector2'],
    'mx_saturate': ['color', 'color4'],
    'mx_scale': ['vector', 'vector2'],
    'mx_separate': ALL_COLVEC,
    'mx_sign': ALL_TYPES,
    'mx_sin': ALL_TYPES,
    'mx_splitlr': ALL_TYPES,
    'mx_splittb': ALL_TYPES,
    'mx_sqrt' : ALL_TYPES,
    'mx_subtract': ALL_TYPES,
    'mx_subtract_float': ALL_COLVEC,
    'mx_swizzle_float': ALL_COLVEC,
    'mx_swizzle_color': ALL_TYPES,
    'mx_swizzle_color2': ALL_TYPES,
    'mx_swizzle_color4': ALL_TYPES,
    'mx_swizzle_vector': ALL_TYPES,
    'mx_swizzle_vector2': ALL_TYPES,
    'mx_swizzle_vector4': ALL_TYPES,
    'mx_switch': ALL_TYPES,
    'mx_tan': ALL_TYPES,
    'mx_tangent': ['vector'],
    'mx_texcoord': ['vector', 'vector2'],
    'mx_tiledimage': ALL_TYPES,
    'mx_time': ['float'],
    'mx_transformpoint' : ['vector', 'vector4'],
    'mx_transformvector' : ['vector', 'vector4'],
    'mx_transformnormal' : ['vector', 'vector4'],
    'mx_transpose' : ['matrix44', 'matrix33'],
    'mx_triplanarprojection': ALL_TYPES,
    'mx_unpremult': ['color', 'color2', 'color4'],
    'mx_mult_surfaceshader': ['color', 'float'],
    'mx_matrix_invert'  : ['matrix44', 'matrix33'],
}

# open_mx_file:  open a file on disk and return its contents
def open_mx_file(shader, options):
    mx_filename = '%s.%s' % (shader, options['mx_ext'])
    mx_filepath = os.path.join(options['source'], mx_filename)
    try:
        mx_file = open(mx_filepath, 'r')
    except:
        print('ERROR: %s not found' % mx_filename)
        return None
    mx_code = mx_file.read()
    mx_file.close()
    return mx_code

# write_mx_file: write the osl_code text to a file
def write_osl_file(osl_shadername, osl_code, options):
    osl_filename = '%s.osl' % (osl_shadername)
    osl_filepath = os.path.join(options['dest'], osl_filename)
    try:
        osl_file = open(osl_filepath, 'w')
        osl_file.write(osl_code)
        osl_file.close()
        return osl_filepath
    except:
        print('ERROR: Could not open %s for writing' % mx_filename)
        return None

# mx_to_osl: open an mx file and for each type in the BUILD_DICT, generate a corresponding .osl file
def mx_to_osl(shader, build_types, options):
    mx_code = open_mx_file(shader, options)
    build_count = 0
    if mx_code is not None:
        for var_type in build_types:
            if var_type in SHADER_TYPES:
                if options['types']:
                    if not var_type in options['types']:
                        if options['v']: print('OSL Generation for type %s skipped.'%var_type)
                        continue
                substitutions = replacements[var_type]
                osl_code = mx_code
                osl_shadername = '%s_%s' % (shader, var_type)
                if options['v']:
                    print('Building %s' % osl_shadername)
                #osl_code = mx_code.replace('SHADER_NAME(%s)' % shader, osl_shadername)
                #osl_code = osl_code.replace('#include \"mx_types.h\"', '#define %s 1\n#include \"mx_types.h\"' % var_type)
                #osl_code = re.sub(r'\bTYPE\b', SHADER_TYPES[var_type], osl_code)
                for s in substitutions :
                    osl_code = osl_code.replace(s[0], s[1])

                osl_filepath = write_osl_file(osl_shadername, osl_code, options)
                build_count += 1
                # build oso bytecode if compile flag is on
                if options['compile']:
                    oso_filename = '%s.oso'%(osl_shadername)
                    osl_filepath = '%s.osl' % (osl_shadername)
                    if options['v']:
                        print('Executing: '+ options['oslc_exec']+' '+osl_filepath)
                    call([options['oslc_exec'], '-I..', osl_filepath], cwd=options['dest'])
            else:
                print('Type %s not found in supported types.'%var_type)
                continue;
    return build_count

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--v', default=0, help='Verbosity, 0|1.  Default: 0')
    parser.add_argument('-mx','--mx', default='.', help='MaterialX source directory.  Default: .')
    parser.add_argument('-oslc_path', '--oslc_path', default='', help='Path to oslc executable.  Default: environment default')
    parser.add_argument('-compile', '--compile', default=0, help='Compile generated osl files in place. 0|1.  Default: 0')
    parser.add_argument('-s', '--shader', default='', help='Specify a comma separated list of mx shaders to convert, e.g. mx_add,mx_absval.  Default: all')
    parser.add_argument('-t', '--types', default='', help='Comma separated list of types to convert, e.g. float,color.  Default: all')
    parser.add_argument('-o', '--out', default='.', help='Destination folder.  Default: current')

    args = parser.parse_args()

    # create a dictionary of options
    oslc_exec = 'oslc'
    types = None

    if args.oslc_path != '':
        oslc_exec = str(os.path.abspath(os.path.join(args.oslc_path, 'oslc')))

    if args.types != '':
        types = args.types.split(',')
        types = [t.lower() for t in types]

    options = {
        'v':int(args.v),
        'source': args.mx,
        'dest':  args.out,
        'mx_ext': 'mx',
        'oslc_path': args.oslc_path,
        'oslc_exec': oslc_exec,
        'compile': args.compile,
        'types': types
    }

    # sanity check paths
    if not os.path.exists(options['dest']):
        print('ERROR: Destination path %s does not exist'%options['dest'])
        return

    if not os.path.exists(options['source']):
        print('ERROR: Source path %s does not exist'%options['source'])
        return

    # If the shader flag was specified, we're only going to build the
    # osl for the named mx file.  If the types flag was specified as well,
    # only generate osl for those types
    if args.shader:
        shaders = args.shader.split(',')
        shaders = [s.split('.')[0] for s in shaders]
        shader_list = { s: BUILD_DICT[s] for s in shaders}
    else:
        shader_list = BUILD_DICT

    # Loop over each shader
    i = 0
    for shader, shader_types in shader_list.items():
        i += mx_to_osl(shader, shader_types, options)
    if options['v']:
        print('Generated ' + str(i) + ' OSL files in ' + options['dest'])

if __name__ == '__main__':
    main()
