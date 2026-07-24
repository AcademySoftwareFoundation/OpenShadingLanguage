---
numbering:
  heading_1: true
  heading_2: true
  heading_3: true
---

<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: CC-BY-4.0
-->


(chap-oslquery)=
# OSLQuery: Interrogating Compiled Shaders

## Introduction and Tutorial

`OSLQuery` is a C++ class that lets an application interrogate a compiled
shader for information about its parameters.

The shader may be an already-compiled shader file on disk (a `.oso` file),
or the `.oso` equivalent in a string buffer, or the binary representation
used by the OSL `ShaderSystem` runtime (as a pointer to a `ShaderGroup`).
For example:

```cpp
OSLQuery oslquery ("polished_oak");
```

It's then easy to retrieve a specific parameter:

```cpp
int nparams = oslquery.nparams();  // number of params

const OSLQuery::Parameter *p;
p = oslquery.getparam (i);            // by index (0..nparams-1)
p = oslquery.getparam ("woodcolor");  // by name
```

The `Parameter` structure holds all the information you need about that
parameter:

```cpp
std::cout << "Parameter " << p->name
          << " is type " << p->type << "\n";
```

You can find out if the parameter is a closure, an output parameter, etc.
Default values are stored in the vector fields `idefault`, `fdefault`, and
`sdefault` depending on whether the type is based on `int`, `float`, or
`string`, respectively.


## OSLQuery API Reference

```{doxygenclass} OSL::OSLQuery
:members:
:undoc-members:
```


## Example: `oslinfo`

Below is the full source of `oslinfo`, a command-line utility that, for any
compiled shader, prints its parameters (name, type, default values, and
metadata). It serves as a complete example of using `OSLQuery`.

```{literalinclude} ../oslinfo/oslinfo.cpp
:language: cpp
:start-at: #include <cstring>
```
