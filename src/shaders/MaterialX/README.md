README for MaterialX Shader Nodes
=================================


### [MaterialX](http://www.materialx.org/)

This is a reference implementation of version 1.35 of the MaterialX
specification, drafted by Lucasfilm Ltd., in collaboration with Autodesk and
The Foundry.  This collection is intended to demonstrate the intended logic
of the MaterialX specification shading nodes as well as a practical
application of current features of the OSL library such as constructors and
operator overrides.

This directory contains .osl files that can be compiled directly with
`oslc`, as well as the source .mx template files, which are used to generate
the .osl (one for each of several specialized types).

Each .mx file contains a type-independent template of a shader, using
special strings such as TYPE, TYPE_ZERO, TYPE_DEFAULT_IN, etc. During the
build process, CMake calls a python script to make the proper substitutions
to generate a .osl file. This .osl file is then compiled into bytcode by
`oslc`.  In all, 475 separate nodes will be built.

Mx files can be compiled to bytecode by hand using the python script
`build_materialX_osl.py` in this directory.  By default this script will
generate .osl for all of the .mx files.

To generate all of the valid osl files from a single .mx file:
```
    build_materialX_osl.py -s <SHADER>
```
To generate a single flavor of osl shader from a single .mx file:
```
    build_materialX_osl.py -s <SHADER> -t <TYPES>
```

Where TYPE is a comma separated list from float, color2, color, color4,
vector2, vector, vector4. Some shaders support these extended types:
matrix44, matrix33, string, filename, int, bool, surfaceshader.

`build_materialx_osl.py` Optional arguments:
```
    -h                  Show this help message and exit
    -v V                Verbosity 0 or 1. Default: 0
    -mx MX              MaterialX source directory. Default: current working directory
    -oslc_path PATH     Path to oslc executable. Default: environment default
    -compile COMPILE    Compile generated osl files in place. 0 or 1. Default: 0
    -s SHADER           Specify a comma separated list of mx shaders to convert e.g. "mx_add.mx,mx_absval.mx". Default: convert all
    -t TYPES            Comma separated list of types to convert, e.g. "float,color". Default: all
    -o PATH             Output path.  Default: current working directory
```

The MaterialX specification is authored and maintained by Doug Smythe and
Jonathan Stone at Lucasfilm Ltd.

This reference implementation was authored by Adam Martinez and Derek Haase.
