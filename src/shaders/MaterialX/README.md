README for MaterialX Shader Nodes
=================================


### [MaterialX Specification](http://www.materialx.org/)


This is a reference implementation of version 1.35 of the MaterialX specification, drafted by Lucasfilm Ltd., in collaboration with Autodesk and The Foundry.  This collection is intended to demonstrate the intended logic of the MaterialX specification shading nodes as well as a practical application of current features of the OSL library such as constructors and operator overrides.

Each .mx file contains a type-independent declaration of a shader.  During the build process, CMake will pass pre-processor arguments to oslc to generate the type-specific .oso shaders. In all, 475 separate nodes will be built.

Mx files can be compiled to bytecode by hand using oslc:  
```    
    oslc -D<TYPE> <FILE>.mx
```  
Where TYPE is one of FLOAT, COLOR2, COLOR, COLOR4, VECTOR2, VECTOR, VECTOR4.  
Some shaders support these extended types: MATRIX44, MATRIX33, STRING, FILENAME, BOOL, INT, SURFACESHADER.  

Alternatively, the python script in src/build-scripts/build_materialX_osl.py can be used to generate .osl that will compile with oslc without any preprocessor definitions. 

Optional arguments:  
``` 
    -h                  Show this help message and exit  
    -arch ARCH          Build architecture flag. Default: linux64  
    -v V                Verbosity 0 or 1. Default: 0  
    -mx MX              MaterialX source directory. Default: ../shaders/MaterialX  
    -oslc_path PATH     Path to oslc executable. Default: environment default  
    -compile COMPILE    Compile generated osl files in place. 0 or 1. Default: 0  
    -s SHADER           Specify a comma separated list of mx shaders to convert without the file extension, e.g. mx_add,mx_absval. Default: none  
    -t TYPES            Comma separated list of types to convert, e.g. FLOAT,COLOR. Default: all  
```
The MaterialX specification is authored and maintained by Doug Smythe and Jonathan Stone at Lucasfilm Ltd.

This reference implementation was authored by Adam Martinez and Derek Haase.
