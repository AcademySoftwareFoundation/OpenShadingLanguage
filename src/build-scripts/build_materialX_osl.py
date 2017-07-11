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
    'FLOAT': 'float',
    'COLOR': 'color',
    'COLOR2': 'color2',
    'COLOR4': 'color4',
    'VECTOR': 'vector',
    'VECTOR2': 'vector2',
    'VECTOR4': 'vector4',
    'SURFACESHADER': 'closure color',
    'MATRIX44': 'matrix',
    'MATRIX33': 'matrix',
    'STRING': 'string',
    'FILENAME': 'string',
    'BOOL': 'bool',
    'INT': 'int',
}

# TYPE_STRING used for type suffix in osl filenames. Could use var_type.lower(). 
TYPE_STRING = {
    'FLOAT': 'float',
    'COLOR': 'color',
    'COLOR2': 'color2',
    'COLOR4': 'color4',
    'VECTOR': 'vector',
    'VECTOR2': 'vector2',
    'VECTOR4': 'vector4',
    'SURFACESHADER': 'surfaceshader',
    'MATRIX44': 'matrix44',
    'MATRIX33': 'matrix33',
    'STRING': 'string',
    'FILENAME': 'filename',
    'BOOL': 'bool',
    'INT': 'int',
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
    'mx_dot': ALL_TYPES + ['MATRIX44', 'MATRIX33', 'STRING', 'FILENAME', 'BOOL', 'INT', 'SURFACESHADER'],
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
    'mx_pack_vf': ['VECTOR4'],
    'mx_pack_vv': ['VECTOR4'],
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
                osl_shadername = '%s_%s' % (shader, TYPE_STRING[var_type])            
                if options['v']:
                    print('Building %s' % osl_shadername)
                osl_code = mx_code.replace('SHADER_NAME(%s)' % shader, osl_shadername)
                osl_code = osl_code.replace('#include \"mx_types.h\"', '#define %s 1\n#include \"mx_types.h\"' % var_type)
                osl_code = re.sub(r'\bTYPE\b', SHADER_TYPES[var_type], osl_code)
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
    parser.add_argument('-arch','--arch', default='linux64', help='Build architecture flag.  Default: linux64')
    parser.add_argument('-v','--v', default=0, help='Verbosity, 0|1.  Default: 0')
    parser.add_argument('-mx','--mx', default='../shaders/MaterialX', help='MaterialX source directory.  Default: ../shaders/MaterialX')
    parser.add_argument('-oslc_path', '--oslc_path', default='', help='Path to oslc executable.  Default: environment default')
    parser.add_argument('-compile', '--compile', default=0, help='Compile generated osl files in place. 0|1.  Default: 0')
    parser.add_argument('-s', '--shader', default='', help='Specify a comma separated list of mx shaders to convert without the file extension, e.g. mx_add,mx_absval.  Default: none')
    parser.add_argument('-t', '--types', default='', help='Comma separated list of types to convert, e.g. FLOAT,COLOR.  Default: all')

    args = parser.parse_args()

    # create a dictionary of options
    oslc_exec = 'oslc'
    types = None

    if args.oslc_path != '':
        oslc_exec = str(os.path.abspath(os.path.join(args.oslc_path, 'oslc')))

    if args.types != '':
        types = args.types.split(',')

    options_dict = {
        'v':int(args.v),
        'source': args.mx,
        'dest': '../../build/%s/src/shaders/MaterialX'%args.arch,
        'arch': args.arch,
        'mx_ext': 'mx',
        'oslc_path': args.oslc_path,
        'oslc_exec': oslc_exec,
        'compile': args.compile,
        'types': types
    }

    # sanity check paths
    if not os.path.exists(options_dict['dest']):
        print('ERROR: Destination path %s does not exist'%options_dict['dest'])
        return

    if not os.path.exists(options_dict['source']):
        print('ERROR: Source path %s does not exist'%options_dict['source'])
        return

    # If the shader flag was specified, we're only going to build the 
    # osl for the named mx file.  If the types flag was specified as well, 
    # only generate osl for those types
    if args.shader:
        shaders = args.shader.split(',')
        shader_list = {s: BUILD_DICT[s] for s in shaders}
    else:
        shader_list = BUILD_DICT

    # Loop over each shader
    i = 0    
    for shader, shader_types in shader_list.items():
        i += mx_to_osl(shader, shader_types, options_dict)
    print('Generated ' + str(i) + ' OSL files in ' + options_dict['dest'])

if __name__ == '__main__':
    main()
