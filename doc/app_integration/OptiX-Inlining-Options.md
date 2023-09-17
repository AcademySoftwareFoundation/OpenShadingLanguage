<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright Contributors to the Open Shading Language Project. -->

Inlining Options for OptiX and CUDA
===================================

When compiling shaders for OptiX and CUDA (and in general), there is a tradeoff
between compile speed and shade-time performance. The LLVM optimizer generally
does a good job of balancing these concerns, but there might be cases where a
renderer can give additional hints to the optimizer to tip the balance one way
or the other.

Aggressive inlining can increase the run-time performance, but can negatively
impact the compile speed. Inlining can be very helpful with small functions
where the function call overhead tends to dwarf the useful instructions, so it
is important to inline such functions when possible.

Choosing to __not__ inline certain functions (e.g., very large noise functions)
allows them to be excluded from the module prior to running the LLVM optimizer
and JIT engine, which can greatly improve the compile time. This is particularly
beneficial when a function is not likely to be inlined anyway; removing such
large functions from the module prior to optimization can speed up compilation
considerably without affecting the generated PTX.

ShadingSystem Attributes
------------------------

There are a number of `ShadingSystem` attributes to help control the inlining
behavior. The default settings should work well in most circumstances, but they
can be adjusted to favor compile speed over shade-time performance, or vice
versa.

* `optix_no_inline_thresh`: Don't inline functions greater-than or equal-to the
threshold. This allows them to be excluded from the module prior to
optimization, which reduces the size of module and can greatly speed up the
optimization and JIT stages.

* `optix_force_inline_thresh`: Force inline functions less-than or equal-to the
threshold. This tends to be most helpful with relatively low values, < 30.

* `optix_no_inline`: Don't inline any functions. Offers the best compile times
at the expense of shade-time performance. This option is not recommended, but is
included for benchmarking and tuning purposes.
    
* `optix_no_inline_layer_funcs`: Don't inline the shader layer functions. This
can moderately improve compile times at the expense of shade-time performance.
    
* `optix_merge_layer_funcs`: Allow layer functions that are only called once to
be merged into their caller, even if `optix_no_inline_layer_funcs` is set. This
can help restore some of the shade-time performance lost by enabling
`optix_no_inline_layer_funcs`.
    
* `optix_no_inline_rend_lib`: Don't inline any functions defined in the
renderer-supplied `rend_lib` module. As an alternative, the renderer can simply
not supply the LLVM bitcode for the `rend_lib` module to the `ShadingSystem`.

Inline/Noinline Function Registration
-------------------------------------

In addition to the `ShadingSystem` attributes, individual functions can be
registered with the `ShadingSystem` as `inline` or `noinline`. Functions can
be unregistered to restore the default inlining behavior. This registration
takes precedence over the `ShadingSystem` inlining attributes, which allows
very fine-grained control when needed.

```C++
// Register
shadingsys->register_inline_function(ustring("osl_abs_ff"));
shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdfdf"));

// Unregister
shadingsys->unregister_inline_function(ustring("osl_abs_ff"));
shadingsys->unregister_noinline_function(ustring("osl_gabornoise_dfdfdf"));
```

It might be best to prefer the `ShadingSystem` attributes to control the inlining
behavior, and to strategically register functions when it is known to be
beneficial through benchmarking and profiling.

Tuning and Analysis
-------------------

We have added a Python script (`src/build-scripts/analyze-ptx.py`) to help
identify functions that might be good candidates for inlining/noinling. This
script will generate a summary of the functions in the input PTX file, with a
list of all functions and their sizes in CSV format. It will also generate a
graphical reprensentation of the callgraph in DOT and PDF format.

An example tuning workflow might include the following steps:

1. Run `analyze-ptx.py` on the "shadeops" and "rend_lib" PTX files to generate
   a list of the functions contained in those modules.
   
   ```bash
   $ analyze_ptx.py shadeop_cuda.ptx
   $ analyze_ptx.py rend_lib_myrender.ptx
   ```

2. Run `analyze-ptx.py` on the generated PTX for a representative shader:

    ```bash
    $ analyze_ptx.py myshader.ptx
    ```
    
3. View the summary file (`myshader-summary.txt`) and the callgraph
   (`myshader-callgraph.gv`) to deterimine which library functions were _not_
   inlined. They will appear as boxes with a dashed outline in the callgraph.
   
   In particular, be on the lookout for trivial functions (e.g., `osl_floor_ff`)
   which have not been inlined. If such functions appear, that might be a sign
   that the inline thresholds need to be adjusted, or that it might be
   beneficial to register specific functions.
