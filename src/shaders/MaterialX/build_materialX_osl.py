#!/usr/bin/env python
'''
Generate compilable .osl files from .mx templates

Adam Martinez

'''

from __future__ import print_function, absolute_import
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

# Define macro replacement lists for each data type
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



# open_mx_file:  open a file on disk and return its contents
def open_mx_file(mx_filename):
    try:
        mx_file = open(mx_filename, 'r')
    except:
        print('ERROR: %s not found' % mx_filename)
        return None
    mx_code = mx_file.read()
    mx_file.close()
    return mx_code


# write_mx_file: write the osl_code text to a file
def write_osl_file(osl_filename, osl_code):
    try:
        osl_file = open(osl_filename, 'w')
        osl_file.write(osl_code)
        osl_file.close()
        return osl_filename
    except:
        print('ERROR: Could not open %s for writing' % mx_filename)
        return None


# mx_to_osl: open an mx file and for each type in build_types, generate a corresponding .osl file
def mx_to_osl (shadername, mx_filename, osl_filename, build_types, othertypes, verbose):
    # print ('mx_to_osl shader=', shadername, ', build_types=', build_types,
    #        ', othertypes=', othertypes)
    mx_code = open_mx_file(mx_filename)
    if mx_code is None:
        return 0
    build_count = 0
    for var_type in build_types:
        if not var_type in SHADER_TYPES:
            print('Type %s not found in supported types.'%var_type)
            continue
        osl_code = mx_code
        osl_shadername = '%s_%s' % (shadername, var_type)
        othertype = othertypes[0]
        if othertype is not None and othertype != 'none' :
            if othertype == 'same' :
                othertype = var_type
            osl_shadername = '%s_%s' % (osl_shadername, othertype)
            for s in replacements[othertype] :
                osl_code = osl_code.replace('OTHER'+s[0], s[1])
        for s in replacements[var_type] :
            osl_code = osl_code.replace(s[0], s[1])
        if verbose:
            print('Building %s' % osl_filename)
        osl_filepath = write_osl_file(osl_filename, osl_code)
        build_count += 1
    return build_count


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose', default=0, help='Verbosity, 0|1.  Default: 0')
    parser.add_argument('-s','--shader', required=True,
                        help='Shader name.')
    parser.add_argument('-m','--mxfile', required=True,
                        help='MaterialX template file, with full path.')
    parser.add_argument('-o','--osl', required=True,
                        help='OSL specialized output file, with full path.')
    parser.add_argument('-t', '--types', required=True,
                        help='Comma separated list of types to convert, e.g. "float,color".')
    parser.add_argument('--othertypes', '--othertypes', default='',
                        help='Comma separated list of secondary types to convert.  Default: none')

    args = parser.parse_args()

    # create a dictionary of options

    if args.types != '':
        types = args.types.split(',')
        types = [t.lower() for t in types]

    if args.othertypes != '':
        othertypes = args.othertypes.split(',')
        othertypes = [t.lower() for t in othertypes]
    else :
        othertypes = None

    i = mx_to_osl (args.shader, args.mxfile, args.osl, types, othertypes, args.verbose)

    if args.verbose:
        print('Generated ' + str(i) + ' OSL files in ' + options['dest'])

if __name__ == '__main__':
    main()
