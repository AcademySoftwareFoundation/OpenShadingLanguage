Release 1.10.8 -- ???, 2019 (compared to 1.10.7)
--------------------------------------------------


Release 1.10.7 -- Oct 1, 2019 (compared to 1.10.6)
--------------------------------------------------
* Adjust for deprecated material in more recent Qt releases. #1043
* Fixes for MinGW compiler. #1047
* Texture "missingalpha" optional parameter generated incorrect code and
  crashed. #1044
* Fix incorrect optimizations surrounding 'exit()' calls in the middle
  of certain shader code blocks. #1051
* LLVM 9 / clang 9 compatibility. #1058
* Fixes to Travis CI system to keep up with OIIO master recently upgrading
  its minimum required CMake. #1065

Release 1.10.6 -- Jul 4, 2019 (compared to 1.10.5)
--------------------------------------------------
* Build: Fixes to FindOpenEXR.cmake. #1022
* ShadingSystem: when building a shader group from serialized commands,
  respect the global lockgeom default. #1032

Release 1.10.5 -- May 1, 2019 (compared to 1.10.4)
--------------------------------------------------
* Extend linearstep() and smooth_linearstep() to work with color, point,
  vector, and normal types (previously restricted to float). #994
* Improve oslc type error detection for variable declarations with init
  expressions. Note that this may flag some warnings and errors that went
  undetected before, involving initialization assignments of incompatible
  types. #991, #993
* Add a build-time option GLIBCXX_USE_CXX11_ABI to force the "new/old string
  ABI" to something other than the default for your version of gcc. #995

Release 1.10.4 -- Apr 1, 2019 (compared to 1.10.3)
--------------------------------------------------
* LPEs: forbid LPE repetitions inside groups. #972
* Build process: build script finding of LLVM is now more robust to certain
  library configurations of llvm, particularly where everything is bundled
  in just one libLLVM without a separate libLLVMMCJIT. #976
* oslc: Improve warnings about ill-advised use of the comma operator. #978
* oslc: Fix an assertion/crash when passing initialization-lists as
  parameters to a function, where the function argument expected was as
  array. #983
* oslc: Fix an assertion/crash for certain type constructors of structs,
  where the struct name was not declared. (This is an incorrect shader,
  but of course should have issued an error, not crashed.) #988
* Note: The experimental OptiX path is not expected to work in this branch!
  Development has continued in the 'master' branch. If you are interested in
  testing the OptiX support while it's under development, please do so with
  the master branch, because fixes and improvements to the OptiX support
  are not being backported to the 1.10 branch.
* Tested and verified that everything builds and works correctly with
  Clang 8 and LLVM 8.0.

Release 1.10.3 -- Feb 1, 2019 (compared to 1.10.2)
--------------------------------------------------
* oslc: Writing to function parameters not marked as `output` was only
  recently properly recognized as an error (it was documented as illegal
  all along). Now we demote to a warning, treating it as a full error was
  too disruptive. #944 (1.10.3)
* testshade: Check that no leftover errors are in the TextureSystem or
  ImageCache at the end of the test (that would indicate that someplace in
  OSL we are failing to check texture errors). #939 (1.10.3)
* Improve oso output efficiency. #938 (1.10.3)
* oslc: Fix bug related to counting the proper number of values used for
  array default value initialization. #948 (1.10.3)
* oslc: Slight modification to the formatting of floating point values in
  the oso file to ensure full precision preservation for float values.
  #949 (1.10.3)
* oslc: Fix crash when encountering empty-expression `for (;;)`. #951 (1.10.3)
* oslc: Fix bug in code generation of certain `while` loops that directly
  tested a variable as a zero/nonzero test, such as:
  ```
      i = 3;
      while (i)
          --i;
  ```
  whereas the following had worked (they should have been identical):
  ```
      i = 3;
      while (i != 0)
          --i;
  ```
  #947 (1.10.3/1.11.0)
* Fix bug in implementation of `splineinverse()` when computing with
  Bezier interpolation. #954 (1.10.3)
* Fix bug in implementation of `transformc` when relyin on OpenColorIO for
  color transformation math, in cases were derivatives were needed (this
  is a rare case probably nobody ran into). #960 (1.10.3)
* Improve thread-safety of the OSLCompiler itself, in case an app wants
  to be compiling several shaders to oso concurrently by different threads.
  #953 (1.10.3)

Release 1.10 -- Dec 1, 2018 (compared to 1.9)
--------------------------------------------------
Dependency and standards changes:
* **LLVM 4.0 / 5.0 / 6.0 / 7.0**: Support has been removed for LLVM 3.x,
  added for 6.0 and 7.0.
* **OpenImageIO 1.8/2.0+**: This release of OSL should build properly against
  OIIO 1.8 or 2.0. Support has been dropped for OIIO 1.7.

New back-end targets:
* **OptiX** Work in progress: Experimental back end for NVIDIA OptiX GPU ray
  tracing toolkit. #861, #877, #902
    - Build with `USE_OPTIX=1`
    - Requires OptiX 5.1+, Cuda 8.0+, OpenImageIO 1.8.10+, LLVM >= 5.0 with
      PTX target enabled.
    - New utility **testoptix** is an example of a simple OptiX renderer
      that uses OSL for shaders.
    - Work is in progress to support the majority of OSL, but right now it
      is restricted to a subset. All the basic math, most of the
      standard library, noise functions, strings (aside from if you create
      entirely new strings in the middle of a shader), and closures work.
      The biggest thing not working yet is textures, but those are coming
      soon.

New tools:
* **osltoy** : GUI tool for interactive shader editing and pattern
  visualization (somewhat in the style of [Shadertoy](http://shadertoy.com).
  #827, #914, #918, #926 (1.10.0)
* **docdeep** : This Python script (in src/build-scripts/docdeep.py) is an
  experimental tool to scrape comments from code and turn them into
  beautiful Markdeep documentation. (A little like a VERY poor man's
  Doxygen, but markdeep!) Experimental, caveat emptor. #842 (1.10.0)

OSL Language and oslc compiler:
* In OSL source code, we now encourage the use of the generic "shader" type
  for all shaders, and it is no longer necessary or encouraged to mark
  the OSL shader with a specific type, "surface", "displacement", etc.
  From a source code and/or `oslc` perspective, all shaders are the same
  generic type. A renderer may, however, have different "uses" or "contexts"
  and may additional runtime perform error checking to ensure that the
  shader group you have supplied for a particular "use" does not do things
  or access globals that are not allowed for that use. #899
* C++11 style Initializer lists. (#838) This lets you have constructs like

        // pass a {} list as a triple, matrix, or struct
        void func (point p);
        func ({x, y, z});

        // Assign {} to a struct if the types match up
        struct vec2 { float x, y; };
        vec2 v = {a,b};

        // Compact 'return' notation, it knows to construct the return type
        vec2 func (float a, float b)
        {
            return {a, b};
        }

* osl now warns when it detects duplicate declarations of functions with
  the exact same argument list, in the same scope. #746
* osl now correctly reports the error when you write to a user-function
  parameter that was not declared as `output` (function params are by
  default read-only, but a shortcoming in oslc has prevented that error
  from being issued). #878 (1.10.0)
* Fix oslc crash with invalid field selection syntax. #835 (1.10.0/1.9.6)
* oslc fix to properly handle command line arguments if the shader file is
  not the last argument on the command line. #841 (1.10.0/1.9.7)
* oslc: when using boost.wave for preprocessing, fix whitespace insertion
  #840 and windows path separators #849. #841 (1.10.0/1.9.7)
* oslc: Fix bug/undefined behavior when trying to format/printf a struct.
  #849 #841 (1.10.0/1.9.7)
* New rules for how polymorphic function variants are chosen: Matches are
  now ranked in an objective way that no longer depends on declaration
  order. Type coercions are preferred in the following order of descending
  score: exact match, int -> float, float -> triple, spatial triple ->
  spatial triple, any triple -> triple. If there is a tie for passed
  arguments, return types will break the tie. If there is still a tie or
  truly ambiguous case, a warning will be printed explaining the choices and
  which was chosen. #844 (1.10.0)
* It is now a warning to define the same function twice in the same scope.
  #853 (1.10.0)
* A shader input parameter marked with metadata `[[ int allowconnect = 0 ]]`
  will disallow runtime connections via `ConnectShaders()`, resulting in an
  error. #857 (1.10.0)
* oslc command-line argument `-Werror` will treat all warnings as hard
  errors (failed compilation). #862 (1.10.0)
* `#pragma osl nowarn` will suppress any warnings arising from code on the
  immediately following line of that source file. #864 (1.10.0)
* oslc error reporting is improved, many multi-line syntactic constructs
  will report errors in a more intuitive, easy-to-understand line number.
  #867 (1.10.0)
* Faster parsing of very large constant initializer lists for arrays in OSL.
  We found an inadvertent O(n^2) behavior when parsing initializer lists.
  It used to be that a constant table in the form an array of 64k floats
  took over 10s to compile, now it is 50x faster. #901
* Resolution of ambiguous return type functions (such as noise) has been
  improved when their results are used as arguments to type constructors.
  #931 (1.10.1)

OSL Standard library:
* There has been a change in the appearance to Cell noise and Gabor noise.
  This is to fix a bug that made an incorrect pattern for certain negative
  exact-integer values for cellnoise, and in lots of places for Gabor noise.
  The only way to fix it was to potentially change the appearance for some
  shaders. Sorry. If this is a real problem, let us know, perhaps we can
  make a build-time switch that will let you use the old buggy noise? But
  since this is a "2.0" release, we figured it was as good a time as ever to
  let it change to the correct results. #912 (1.10.0)

Shaders:
* Contributed shader library changes:
    * mandelbrot.osl: computes Mandelbrot and Julia images. #827 (1.10.0)
* MaterialX support:
    * Improved support for MaterialX 1.36: add sin, cos, tan, atan2, ceil,
      sqrt, exp, determinent, ln, transpose, sign, rotate, transforms,
      rgb/hsv convert, extract, separate, tiledimage. Rename exponent ->
      power, pack -> combine, hsvadjust -> hueshift. Add some color2/4
      mutual conversion operators. Fixes to ramp4, clean up texture mapping
      nodes, fixes to triplanarprojection weighting. Extend add/sub/mul/div
      to include matrices. #903, #904, #905, #907, #909 (1.9.10/1.10.0)

API changes, new options, new ShadingSystem features (for renderer writers):
* ShadingSystem API:
    * It is now permitted to ConnectShaders a single component of a
      color/point/vector/normal to a float and vice versa. #801 (1.10.0)
    * An older version of ShadingSystem::execute, which had been marked
      as deprecated since OSL 1.6, has been fully removed. #832 (1.10.0)
    * `ShadingSystem::Shader()` now has all three parameters required (none
      are optional), though the "use" parameter no longer has any meaning.
      (It will be deprecated and removed in a future release.) #899
    * `ShadingSystem::optimize_group()` now takes an optional pointer to a
      `ShadingContext`, which it will use if needed (if passed NULL, one
      will be internally allocated, used, and freed, as before). #936
* ShadingSystem attributes:
    * New `"allow_shader_replacement"` (int) attribute, when nonzero, allows
      shaders to be specified more than once, replacing their former
      definitions. The default, 0, considers that an error, as it always
      has. #816 (1.10.0).
    * New developer option `"llvm_output_bitcode"` dumps the bitcode for each
      group, even if other debug options aren't turned on, and also any
      dumped bitcode will save as text as well as binary. #831 (1.10.0)
    * New attribute `"error_repeats"`, if set to non-zero, turns off the
      suppression of multiple identical errors and warnings. Setting it
      (even to its existing value) also clears the "already seen" lists.
      #880, #883 (1.10.0/1.9.9/1.8.14)
* Shader group attributes:
    * New attributes that can be queried with `getattribute()`:
      `"globals_read"` and `"globals_write"` retrieve an integer bitfield
      that can reveal which "globals" may be read or written by the shader
      group. The meaning of the bits is given by the enum class `SGBits`
      in `oslexec.h`. #899
* RendererServices API:
    * Older versions of RendererServices texture functions, the old ones
      with no errormessage parameter, which were documented as deprecated
      since 1.8, are now marked OSL_DEPRECATED. #832 (1.10.0)
* OSLCompiler API:
    * Improved error reporting when compiling from memory buffer. The
      `OSLCompiler::compile_buffer()` method now takes an optional filename
      parameter that will make error messages attribute the right "file"
      (e.g., `Error: foo.osl:5...` rather than `<buffer>:5...`). #937 (1.10.2)
* Miscellaneous:
    * liboslnoise: Properly hide/export symbols. #849 (1.10.0/1.9.7)
    * The behavior of the "searchpath:shader" attribute, used for finding
      `.oso` files when shaders is requested, has been modified. If no
      search path is specified, the current working directory (".") will
      be searched. But if there is a search path attribute specified, only
      those directories will be searched, with "." only searched if it is
      explicitly included in the search path value. #925 (1.10.0)

Bug fixes and other improvements (internals):
* The context's `texture_thread_info` is now properly passed to the
  RenderServices callbacks instead of passing NULL. (1.10.0)
* Symbols are enbled in the JIT, allowing Intel Vtune profiler to correctly
  report which JITed OSL code is being executed. #830 (1.10.0)
* ASTNode and OSLCompilerImpl -- add info() and message() methods to
  complement the existing error and warning. #854 (1.10.0)
* Fix incorrect array length on unbounded arrays specified by relaxed
  parameter type checking. #900 (1.9.10/1.10.0)
* oslc bug fix: the `regex_search()`/`regex_match()` functions did not properly
  mark their `results` output parameter as write-only. This was never
  noticed by anyone, but could have resulted in incorrect optimizations.
  #922 (1.10.0)
* When reading `.oso` files, the parser is now more robust for certain ways
  that the oso file might be corrupted (it's more forgiving, fewer possible
  ways for it to abort or assert). #923 (1.10.0)
* Bug fixes related to incorrect reliance on OIIO's `ustring::operator int()`.
  It's being removed from OIIO, so wean ourselves off it. #929 (1.10.0)
* Certain texture3d lookups with derivatives could crash. #932 (1.10.1)
* Fix oslc assertion crash when a struct parameter was initialized with
  a function call. #934 (1.10.1)

Build & test system improvements:
* Appveyor CI testing for Windows. #849,852,855 (1.10.0/1.9.7)
* Our new policy is to disable `STOP_ON_WARNING` for release branches, to
  minimize build breaks for users when a new compiler warning is hit. We
  still enable it in development/master branches as well as any CI build
  in any branch. #850 (1.10.0/1.9.7)
* Testsuite is now Python 2/3 agnostic. #873 (1.10.0)
* Build the version into the shared library .so names. #876
  (1.8.13/1.9.8/1.10.0)
* Update to fix with OpenImageIO 1.9. #882,#889
* Flex/bison fixes on Windows. #891
* Fix Windows build break with iso646.h macros. #892
* Fix warnings on gcc 6. #896
* Fix errors building with MSVC. #898
* Fixes to build with clang 7, and to use LLVM 7. #910, #921 (1.10.0)
* Fix warnings on gcc 8. #921 (1.10.0)
* Build system: the variables containing hints for where to find IlmBase
  and OpenEXR have been changed to `ILMBASE_ROOT_DIR` and `OPENEXR_ROOT_DIR`
  (no longer `ILMBASE_HOME`/`OPENEXR_HOME`). Similarly, `OPENIMAGEIO_ROOT_DIR`
  is the hint for custom OIIO location (no longer OPENIMAGEIOHOME). #928
* Eliminated some in-progress MaterialX tests, they weren't in good order,
  we will do it differently if we want to add MX tests in the future. #928
* Build options `OSL_BUILD_SHADERS` and `OSL_BUILD_MATERIALX` (both default
  to ON) can be used to disable building of all shaders or MX shaders,
  respectively. #935 (1.10.1)

Documentation:
* `osltoy` documentations in `doc/osltoy.md.html` (1.10.0).



Release 1.9.13 -- 1 Dec 2018 (compared to 1.9.12)
------------------------------------------------
* Fix crash with texture3d lookups with derivatives. #932
* Fix oslc crash when a struct parameter is initialized with a function call
  that returns a structure. #934

Release 1.9.12 -- 1 Nov 2018 (compared to 1.9.11)
------------------------------------------------
* Fix oslc read/write error for `regex_search`/`regex_match` #922
* Make oso reading more robust to certain broken inputs. #923
* Internals: make safe for some changes coming to ustring API in OIIO
  master. #929
* Several docs fixes.

Release 1.9.11 -- 1 Oct 2018 (compared to 1.9.10)
------------------------------------------------
* Full support for using LLVM 6.0 and 7.0. #913, #919
* Support for building with gcc 8. #921
* Fix testrender bug with undefined order of operations (only was a problem
  with gcc5 and clang7). #916

Release 1.9.10 -- 1 Sep 2018 (compared to 1.9.9)
------------------------------------------------
* Fix Windows compile of the flex/bison compiler components. #891
* Fix for compatibility with OIIO 1.9.
* Fix incorrect array length on unbounded arrays specified by relaxed
  parameter type checking. #900
* Speed up oslc parsing of long constant initializer lists. #901
* Add more functions to color2.h, color4.h, vector2.h, vector4.h: ceil,
  sqrt, exp, log2, log, sign, sin, cos, tan, asin, acos, atan2. #903, #904
* Improved support for MaterialX 1.36: add sin, cos, tan, atan2, ceil, sqrt,
  exp, determinent, ln, transpose, sign, rotate, transforms, rgb/hsv convert,
  extract, separate, tiledimage. Rename exponent -> power, pack -> combine,
  hsvadjust -> hueshift. Add some color2/4 mutual conversion operators.
  Fixes to ramp4, clean up texture mapping nodes, fixes to triplanarprojection
  weighting. Extend add/sub/muldiv to include matrices. #903, #904, #905

Release 1.9.9 -- 1 May 2018 (compared to 1.9.8)
-----------------------------------------------
* New SS attribute `"error_repeats"`, if set to non-zero, turns off the
  suppression of multiple identical errors and warnings. Setting it (even to
  its existing value) also clears the "already seen" lists. #880
  (1.8.14/1.9.9)
* Update to fix with some changes in OpenImageIO 1.9. #882

Release 1.9.8 -- 1 Apr 2018 (compared to 1.9.7)
-----------------------------------------------
* Build the version into the shared library .so names. #876 (1.8.13/1.9.8)

Release 1.9.7 -- 1 Feb 2018 (compared to 1.9.6)
-----------------------------------------------
* oslc fix to properly handle command line arguments if the shader file is
  not the last argument on the command line. #841
* oslc: when using boost.wave for preprocessing, fix whitespace insertion
  #840 and windows path separators #849.
* oslc: Fix bug/undefined behavior when trying to format/printf a struct.
  #849
* liboslnoise: Fix symbol export/hiding. #849
* Misc build issue cleanup on Windows. #849
* For release branches, we no longer have builds consider every compiler
  warning to be an error (except in master or for CI builds).

Release 1.9.6 -- 1 Jan 2018 (compared to 1.9.5)
-----------------------------------------------
* Fix oslc crash with invalid field selection syntax. #835
* Certain texture calls were inadvertently not passing in thread data,
  forcing the texture system to look it up again redundantly. #829


Release 1.9 -- 4 December 2017 (compared to 1.8)
--------------------------------------------------

Dependency and standards changes:
* **C++11 required**: OSL 1.9 requires a minimum standard of C++11. It
  should also build against C++14 and C++17.
* **LLVM 3.5 / 3.9 / 4.0 / 5.0**: Support has been added for LLVM 3.9, 4.0,
  and 5.0. Support has been removed for for LLVM 3.4.
* **OpenImageIO 1.7+**: This release of OSL should build properly against
  OIIO 1.7 or newer. You may find that 1.6 is still ok, but we are not doing
  any work to ensure that.
* **CMake >= 3.2.2**
* **Boost >= 1.55**
* **OpenEXR/IlmBase >= 2.0** (recommended: 2.2)

Language features:
* New preprocessor symbols: `OSL_VERSION_MAJOR`, `OSL_VERSION_MINOR`,
  `OSL_VERSION_PATCH`, and `OSL_VERSION` (e.g. 10900 for 1.9.0) reveal the
  OSL release at shader compile time. #747 (1.9.0)
* Structure constructors: If you have a struct `S` comprising fields with
  types T1, T2, ..., you may now have an expression `S(T1 v2, T2 v2,...)`
  that constructs and returns an `S` with those field values, much in the
  same way that you can say `color(a,b,c)` to construct a color out of
  components a, b, c.  #751 (1.9.0)
* User-defined operator overloading: If you make a new (struct) type, it
  is possible to define overloaded operators, like this:

      struct vec2 { float x; float y; };

      vec2 __operator__add__ (vec2 a, vec2 b) { return vec2(a.x+b.x, ay+b.y); }

      vec2 a, b, c;
      a = b + c;   // chooses __operator__add__()

  This can be done with any of the operators, see the OSL Language Spec PDF
  for details. #753 (1.9.0)

Standard library additions/changes:
* `getattribute ("osl:version", intvar)` at runtime can reveal the OSL
  version on which the shader is being executed. #747 (1.9.0)
* `pointcloud_search()/pointcloud_get()` have more flexibility in what type
  of data it may retrieve: you can now retrieve arrays, if that is what is
  stored per-point in the point cloud (for example, a `float[4]`).
  #752 (1.9.0)
* `smoothstep()` has been extended to `T smoothstep(T edge0, T edge1, T x)`
  for T of any the `triple` types (previously, `smoothstep` only came in
  the `float` variety). #765 (1.9.0/1.8.10)
* `mix()` has been extenended to support
      `color closure mix (color closure A, color closure B, color x)`
  #766 (1.9.0/1.8.10)
* `hashnoise()` is like cellnoise (1D, 2D, 3D, or 4D input, 1D or 4D output
  on [0,1]), but is discontinuous everywhere (versus cellnoise, which is
  constant within each unit cube and discontinuous at at integer coordinates).
  #775 (1.9.0/1.8.10)
* `int hash (...)` has been extended to take arguments that are int, float,
  2 floats, point, or point+float. #775 (1.9.0/1.8.10)
* `transformc()` can now perform any color transformations understood by
  OpenColorIO (assuming OCIO support was enabled when OSL was build, and that
  a valid OCIO configuration is found at runtime). #796 (1.9.1) Also,
  `transformc()` now fully supports derivatives in all cases. #798 (1.9.1)

Contributed shader library changes:
* New headers: color2.h, color4.h, vector2.h, vector4.h. Technically these
  are not part of the OSL specification and standard library, but are
  "contributed" code that you may find handy. They implement 2- and 4-
  component colors (RA and RGBA) and 2D and 4D vectors. #777 (1.9.1)
* A full complement of MaterialX shaders is now included in the OSL
  distribution. #777 (1.9.1)

API changes, new options, new ShadingSystem features (for renderer writers):
* ShadingSystem API changes:
    * New `set_raytypes()` call sets the known raytypes (on and off) for
      a shader group for subsequent optimization. This lets you combine ray
      specialization with lazy compilation. #733 (1.9.0)
    * `Parameter()` is now less strict about type checking when setting
      parameter instance values. In particular, it's now ok to pass a
      `float` value to a "triple" (color, point, etc.) parameter, and to
      pass one kind of triple when a different kind of triple was the
      declared parameter type. In this respect, the rules now more closely
      resample what we always allowed for `ConnectShaders`. #750 (1.9.0)
    * More optional `Parameter()` type checking relaxation: if the
      ShadingSystem attribute `"relaxed_param_typecheck"` is nonzero, an
      array of floats may be passed for float aggregates (e.g. color) or
      arrays of float aggregates, as long as the total number of floats
      matches. For example, with this attribute turned on, a `float[3]`
      may be passed as a parameter that expected a `vector`. Also, an `int`
      may be passed to a `float` parameter, and an `int[1]` may be passed
      to an `int` parameter. #794,#797 (1.9.1)
    * A new, optional, slightly relaxed policy for type checking what is
      passed via `Parameter()`
    * `Shader()` will now accept the name of the shader as if it were the
      filename, with trailing `.oso`, and it will be automatically stripped
      off. #741 (1.9.0)
    * `convert_value()` now allows conversions between `float[3]` and triple
      values. #754 (1.9.0)
* ShadingSystem attribute additions/changes:
    * `"relaxed_param_typecheck"` (default=0) enables more relaxed type
      checking of `Parameter()` calls: arrays of float may be passed to
      parameters expecting a float-aggregate or array thereof, and an `int`
      may be passed to a parameter expecting a `float`, and an `int[1]` may
      be passed to an `int` parameter. #794,#797 (1.9.1)
* Fixed `ClosureComponent` to work with SSE alignment requirements. #810
  (1.9.3)

Performance improvements:
* Shader JIT time is improved by about 10% as a result of pre-declaring
  certain function addresses instead of relying on LLVM to use dlsym() calls
  to find them within the executable. #732 (1.9.0)
* The runtime cost of range checking array accesses has been reduced by
  about 50%. #739 (1.9.0)
* Runtime optimization: Constant folding of `%` operation. #787 (1.9.1)

Bug fixes and other improvements (internals):
* Avoid division by 0 when computing derivatives in pointcloud_search.
  #710 (1.9.0/1.8.7)
* Avoid subtle use-after-free memory error in dictionary_find().
  #718 (1.9.0/1.8.6)
* Fix minor inconsistency in the behavior of `normalize()` when the input
  has derivatives versus when it does not. #720 (1.9.0/1.8.7)
* Fix an optimization bug where calls to `trace()` could accidentally get
  elided if the results of the function call were unused in the shader.
  This is incorrect! Because `trace()` has side effects upon subsequent
  calls to `getmessage("trace",...)`. #722 (1.9.0/1.8.7)
* Runtime optimizer is sped up by avoiding some string operations related
  to searching for render outputs when none are registered. (1.9.0)
* Searching for stdosl.h now works uniformly whether it's oslc itself, or
  apps that use OSLCompiler, and in all cases are better about guessing
  where the header is located even when `$OSLHOME` environment variable is
  not set. #737 (1.9.0)
* Internals: Fix the handling of alignment for closure structs. #740 (1.9.0)
* oslc: fix internal memory leak of ASTNode's. #743 (1.9.0)
* testshade improvements:
    * New option `--texoptions` lets you directly set extra TextureSystem
      options for tests. #744 (1.9.0)
    * Fix that allows you to set a parameters that is an array-of-strings.
      #745 (1.9.0)
    * Rename `--scalest/--offsetst` to `--scaleuv/--offsetuv` to properly
      reflect that they alter u and v (there is no s, t). #757 (1.9.0)
    * `--print` prints the value of all saved outputs. #757 (1.9.0)
    * Automatically convert to sRGB when saving outputs to JPEG, PNG, or GIF
      images, to make them more "web ready." #757 (1.9.0)
    * `--runstats` is more careful about not including the time to write
      output images in the main shader run time statistic. #757 (1.9.0)
    * Rename `-od` option to `-d` to match oiiotool and maketx. #757 (1.9.0)
* testrender: Automatically convert to sRGB when saving outputs to JPEG,
  PNG, or GIF images, to make them more "web ready." #757 (1.9.0)
* Slight efficiency improvement when you call texture functions with the
  optional `"subimage"` parameter and pass the empty string (which means
  the first subimage, equivalent to not passing `"subimage"` at all).
  #749 (1.9.0)
* oslc bug fixes where in some circumstances polymorphic overloaded
  functions or operators could not be properly distinguished if their
  parameters were differing `struct` types. #755 (1.9.0)
* Fix minor numerical precision problems with `inversespline()`. #772 (1.9.0)
* `testshade` gives better error messages for the common mistake of using
  `-param` after the shader is set up. #773 (1.9.0)
* Fix bug with transitive assignment for arrays, like `a[0] = a[1] = 0;`
  #774 (1.9.0)
* The standard OSL library function fprintf() was not properly implemented.
  #780 (1.9.1)
* Fix subtle bugs related to our ignorance of "locales" -- we now are very
  careful when parsing `.osl` source (and other places) to be always use
  the `'.'` (dot) character as decimal separator in floating point number,
  even when running on a computer system configured to use a foreign locale
  where the comma is traditionally used as the decimal separator. #795 (1.9.1)
* Fix param analysis bug for texture or pointcloud functions with optional
  token/value parameters where the token name wasn't a string literal -- it
  could fail to recognize that certain parameters would be written to by the
  call. #812 (1.9.3)
* ShadingSystem statistics are now printed if any shaders were
  declared/loaded, even if no shaders were executed. #815 (1.9.3)
* Minor OSLQuery implementation improvements: add move/copy constructors
  for OSLQuery::Parameter, make the ShadingSystem side of OSLQuery correctly
  report default parameter values. #821 (1.9.4)

Build & test system improvements:
* C++11 is the new language baseline. #704, #707
* Many uses of Boost have been switched to use C++11 features, including
  prior uses of boost::shared_ptr, bind, ref, unordered_map, unordered_set
  now using the std:: equivalents; BOOST_FOREACH -> C++11 "range for"; Boost
  string algorithms replaced by OIIO::Strutil functions; Boost filesystem
  replaced by OIIO::Filesystem functions; Boost scoped_ptr replaced by
  std::unique_ptr; boost::random replaced by std::random;
  boost::intrusive_ptr replaced by OIIO::intrusive_ptr; boost thread
  replaced by std::thread and various OIIO::thread utilities. #707 (1.9.0)
* CMake 3.2.2 or higher is now required. The CMake build scripts have been
  refactored substantially and cleaned up with this requirement in mind.
  #705 (1.9.0)
* Boost 1.55 or higher is now required (per VFXPlatform guidelines).
* Big refactor of the CMake scripts for how we find LLVM, now also broken
  out into a separate FindLLVM.cmake. #711 (1.9.0)
* Remove direct references to tinyformat in lieu of C++11 variadic
  templates and use of OIIO::Strutil::format. #713 (1.9.0)
* Use std::regex rather than boost::regex any time the former is available.
  (Note: gcc 4.8 will automatically still fall back to boost, since correct
  implementation of std::regex did not happen until gcc 4.9.) #714 (1.9.0)
* When available (and with the right compiler version combinations), OSL
  will rely on Clang library internals to "preprocess" oso input, rather
  than Boost Wave. This solves problems particularly on OSX and FreeBSD
  where clang/C++11-compiled OSL was having trouble using Boost Wave if
  Boost was not compiled in C++11 mode (which is difficult to ensure if
  you don't control the machine or build boost yourself). #715 (1.8.5/1.9.0)
  #719 #721 (1.9.0/1.8.7)
* Tweaks to FindOpenImageIO.cmake. (1.9.0)
* Fixed linkage problems where some of our unit test programs were unwisely
  linking against both liboslcomp and liboslexec (not necessary, and caused
  problems for certain LLVM components that appeared statically in both).
  #727 (1.9.0/1.8.7)
* Added an easy way to invoke clang-tidy on all the files. #728
* All internal references to our public headers have been changed to the
  form #include <OSL/foo.h>, and not "OSL/foo.h" or "foo.h". #728
* The namespace has been changed somewhat, is now only one level deep and
  contains the version, eliminating version clashes within the same
  executable. You still refer to namespace "OSL", it's an alias for the
  real versioned (and potentially customized) one. #732 (1.9.0)
* Symbol visibility is now properly restricted for certain "C" linkage
  functions needed for availability by the LLVM-generated code. And overall,
  the HIDE_SYMBOLS build mode is now on by default. #732 (1.9.0)
* More robust finding of external PugiXML headers. (1.9.0)
* Fix ilmbase linker warning with LINKSTATIC on Windows. #768 (1.9.0)
* Fix osl_range_check not found error when USE_LLVM_BITCODE=OFF. #767 (1.9.0)
* Windows fixes where BUILDSTATIC incorrectly set /MT flag. #769 (1.9.0)
* Some preliminary work to make OSL safe to compile with C++17. (1.9.1)
* C++11 modernization: use range-for loops in many places. #785 (1.9.1)
* Make OSL build with clang 5.0 and against LLVM 5.0. #787 (1.9.1)
* Removed support for building against LLVM 3.4. #792 (1.9.1)
* Use GNUInstallDirs to simplify build scripts and more properly conform to
  widely established standards for layout of installation directory files.
  #788 (1.9.1)
* Improved proper rebuilding of the LLVM bitcode for llvm_ops.cpp when only
  certain headers change. #802 (1.9.1)
* Fix gcc7 warnings about signed vs unsigned compares. #807 (1.9.2)
* Simplify the build logic for finding PugiXML and prefer a system install
  when found, rather than looking to OIIO to supply it. #809 (1.9.2)
* MSVS 2015 x64 compilation fixes. #820 (1.9.4)
* Fix debug compile against OIIO 1.7. #822 (1.9.4)

Developer goodies:
* The `dual.h` implementation has been completely overhauled. The primary
  implementation of dual arithmetic is now the template `Dual<T,PARTIALS>`
  where `PARTIALS` lets it specialize on the number of partial derivatives.
  This lets you use the `Dual` templates for automatic differentiation of
  of ordinary 1D functions (e.g., `Dual<float,1>`) or 3D volumetric
  computations (e.g., `Dual<float,3>`). Most of OSL internals use automatic
  differentiation on 2 dimenstions (`Dual<float,2>` a.k.a. `Dual2<float>`),
  but this change makes `dual.h` a lot more useful outside of OSL.
  #803 (1.9.1)

Documentation:
* Fixed unclear explanation about structures with nested arrays. (1.9.0)
* Full testshade docs in `doc/testshade.md.html` (1.9.0)



Release 1.8.15 -- 1 Aug 2018 (compared to 1.8.14)
--------------------------------------------------
* Fixes for compatibility with OIIO 1.9.

Release 1.8.14 -- 1 May 2018 (compared to 1.8.13)
--------------------------------------------------
* New SS attribute "error_repeats", if set to non-zero, turns off the
  suppression of multiple identical errors and warnings. Setting it (even to
  its existing value) also clears the "already seen" lists. #880
* Update to fix with some changes in OpenImageIO 1.9. #882

Release 1.8.13 -- 1 Apr 2018 (compared to 1.8.12)
--------------------------------------------------
* Build the version into the shared library .so names. #876

Release 1.8.12 -- 1 Nov 2017 (compared to 1.8.11)
--------------------------------------------------
* Improve type checking error messages with structs. #761

Release 1.8.11 -- 3 Oct 2017 (compared to 1.8.10)
--------------------------------------------------
* Builds properly against LLVM 5.0, and if compiled by clang 5.0.
* Changes to test/CI with recent OIIO release.

Release 1.8.10 -- 1 Jul 2017 (compared to 1.8.9)
--------------------------------------------------
* Missing faceforward() implementation was added to stdosl.h. #759
* `testshade` new option `--shader <shadername> <layername>` is a
  convenience that takes the place of a separate `--layer` and `--shader`
  arguments, and makes the command line look more similar to the serialized
  (text) description of a shader group. #763
* README.md, CHANGES.md, and INSTALL.md are now installed in the "doc"
  directory of a fully-built dist area, rather than at its top level.
* `smoothstep()` now has variants where the arguments and return type may
  be color, vector, etc. (previously it only worked for `float`. #765
* `mix()` now has a variant that combines closures. #766
* testshade comprehensive documentation in `doc/testshade.md.html`, just
  view with your browser.
* Fixed a numerical precision problem with `inversespline()`. #772
* Fixed a bug in transitive assignment of indexed arrays, like
  `f[0] = f[1] = f[2]`. This previously hit an assertion. #774
* New standard OSL function `hash()` makes a repeatable integer hash of
  a float, 2 floats, a triple, or triple + float. #775
* New `hashnoise()` is similar to cellnoise(), but has a different
  repeatable 0.0-1.0 range value for every input position (whereas cellnoise
  is constant between integer coordinates). #775

Release 1.8.9 -- 1 Jun 2017 (compared to 1.8.8)
--------------------------------------------------
* Minor speedup when passing blank subimage name:
  texture (..., "subimage", "", ...)
  #749
* Fix namespace clash after recent OIIO (master branch) PugiXML version
  upgrade.
* Several testshade changes: renamed options --scalest/--offsetst to
  --scaleuv/--offsetuv to more closely say what they really do;
  rename -od to -d to match how oiiotool and maketx call the same
  option; automatically convert to sRGB when instructed to save an
  output image as JPEG, GIF, or PNG; -runstats is more careful with
  timing to not include image output in shader run time; new --print
  option prints outputs to console (don't use with large images!). #757

Release 1.8.8 -- 1 May 2017 (compared to 1.8.7)
--------------------------------------------------
* New ShadingSystem::set_raytypes() can be used to configure default ray
  types for subsequent lazy optimization of the group upon first execution.
  #733
* Hide symbol visibility for extern "C" linkage osl_* functions that LLVM
  needs. #732
* New oslc-time macros OSL_VERSION, OSL_VERSION_MAJOR, OSL_VERSION_MINOR,
  OSL_VERSION_PATCH lets you easily test OSL version while compiling your
  shader. You can also find out at runtime with getattribute("osl:version").
  #747

Release 1.8.7 -- 1 Apr 2017 (compared to 1.8.6)
--------------------------------------------------
* Fix possible division-by-zero when computing derivatives in
  pointcloud_search. #710
* When using clang components as the C preprocessor for .osl files, better
  reporting of any C preprocessor errors. #719
* Fix minor inconsistency in the behavior of `normalize()` when the input
  has derivatives versus when it does not. #720
* Fixes to using clang for the C preprocessing -- we discovered cases where
  it's unreliable for older clang versions, so it now only works when using
  LLVM >= 3.9 (for older LLVM versions, we fall back to boost wave for our
  preprocessing needs). #721
* Fix an optimization bug where calls to `trace()` could accidentally get
  elided if the results of the function call were unused in the shader.
  This is incorrect! Because `trace()` has side effects upon subsequent
  calls to `getmessage("trace",...)`. #722
* Fixed linkage problems where some of our unit test programs were unwisely
  linking against both liboslcomp and liboslexec (not necessary, and caused
  problems for certain LLVM components that appeared statically in both).
  #727

Release 1.8 [1.8.6] -- 1 Mar 2017 (compared to 1.7)
---------------------------------------------------
Dependency and standards changes:
* OSL now builds properly against LLVM 3.9 and 4.0 (in addition to 3.5 and
  3.4). Please note that for 3.9 or 4.0 (which use MCJIT), the JIT overhead
  time is about twice as much as when using LLVM 3.4/3.5 ("old" JIT), but
  the resulting code runs a bit faster. You may see fast, low-quality,
  interactive renders slow down, but long complex renders speed up. We are
  working to address the JIT overhead. In the mean time, if you are
  especially sensitive to interactive compile speed, just keep using LLVM
  3.5 (or 3.4) while we work to address the problem.  **OSL 1.8 will be the
  last branch that supports LLVM 3.4 and 3.5.**
* C++ standard: OSL 1.8 should build againt C++03, C++11, or C++14.
  **OSL 1.8 is the last release that will build against C++03. Future
  major releases will require C++11 as a minimum.**
* OpenImageIO: This release of OSL should build properly against OIIO
  1.6 or newer (although 1.7 is the current release and therefore
  recommended).

**New library: `liboslnoise`** and header `oslnoise.h` expose OSL's noise
  functions to C++ apps. Currently, only signed and unsigned perlin, and
  cellnoise, are exposed, but other varieties and options may be added if
  desired. #669 (1.8.2)

Language features:
* Function return values are now allowed to be structs. (#636)
* oslc now will warn when assigning the result of a "comma operator",
  which almost always is a programming error where the shader writer
  omitted the type name when attempting to construct a vector or matrix.
  (#635) (1.8.0/1.7.3)
* Nested structs (structs containing other structs) are now functional
  as shader parameters, with initialization working as expected with
  nested `{ {...}, {...} }`.  #640 (1.8.0)
* oslc (and liboslcomp) now supports UTF-8 filenames, including on
  Windows. #643 (1.8.0)
* osl now accepts hexidecimal integer constants (such as `0x01ff`).
  #653 (1.8.1)

Standard library additions/changes:
* `pow(triple,triple)` is now part of the standard library. (1.8.2)
* New `getattribute()` tokens: `"shader:shadername"`, `"shader:layername"`,
  and `"shader:groupname"` lets the shader retrieve the shader, layer,
  and group names. This is intended primarily for helpful error messages
  generated by the shader. #673 (1.8.2)
* The texture(), texture3d(), and environment() functions now accept
  an optional "errormessage" parameter (followed by a string variable
  reference). A failure in the texture lookup will result in an error
  message being placed in this variable, but will not have any error
  message passed to the renderer (leaving error handling and reporting
  entirely up to the shader). #686 (1.8.2)
* `select(a,b,cond)` is similar to `mix`, but without interpolation --
  if cond is 0, returns a, else return b (per comonent). #696 (1.8.3)

API changes, new options, new ShadingSystem features (for renderer writers):
* ShadingSystem API changes:
   * Long-deprecated ShadingAttribState type has been removed (now called
     ShaderGroup).
   * ShadingSystem::create() and destroy() are no longer necessary and have
     been removed.
   * A new Shadingsystem::optimize_group() method takes bit fields
     describing which ray types are known to be true or false, allowing
     specialized versions of the shader to be optimized for certain ray
     types (such as for shadow rays only, or displacement shades only).
     #668 (1.8.2)
* ShadingSystem attribute additions/changes:
    * Attribute "userdata_isconnected", if set to a nonzero integer, causes
      a shader call to isconnected(param) to return 1 if the parameter is
      potentially connected to userdata (that is, if lockgeom=0 for that
      parameter). This if for particular renderer needs, and the default
      remains for isconnected() to return 1 only if there is an actual
      connection. #629 (1.8.0/1.7.3)
    * Attribute "no_noise" will replace all noise calls with an inexpensive
      function that returns 0 (for signed noise) or 0.5 (for unsigned noise).
      This can be helpful in understanding how much render time is due to
      noise calls. #642 (1.8.0)
    * Attribute "no_pointcloud" will short-circuit all pointcloud lookups.
      This can be helpful in understanding how much render time is due to
      pointcloud queries. #642 (1.8.0)
    * When the attribute "profile" is set, the statistics report will
      include a count of the total number of noise calls. #642 (1.8.0)
    * ShadingSystem and shader group attribute "exec_repeat" (int; default is
      1) can artifically manipulate how many times the shader is executed.
      This can sometimes be handy for tricky benchmarks. #659 (1.8.2)
    * Attribute "force_derivs" can force derivatives to be computed
      everywhere. (Mostly good for debugging or benchmarking derivatives,
      not intended for users.) #667 (1.8.2)
    * Attribute "llvm_debug_ops" adds a printf before running each op.
      This is mostly for debugging OSL itself, not for users. (1.8.3)

Performance improvements:
* New runtime optimization: better at understanding the initial values of
  output parameters and what optimizations can be done based on that. #657
  (1.8.1)
* Better runtime optimization of certain matrix constructors. #676 (1.8.2)
* Improved alias analyis for parameters with simple initialization ops.
  #679 (1.8.2)
* Runtime optimizer now constant-folds bitwise `&`, `|`, `^`, `~`
  operators. #682 (1.8.2)
* Improved runtime constant folding of triple(space,const,const,const)
  when the space name is known to be equivalent to "common". #685 (1.8.2)

Bug fixes and other improvements:
* oslc bug: getmatrix() failed to note that the matrix argument was
  written. #618 (1.8.0/1.7.2)
* Fix oslc incorrectly emitting init ops for constructor color(float).
  #637 (1.8.0/1.7.3)
* Fix subtle bug in runtime optimizer that could seg fault. #651 (1.8.0)
* Fixed incorrect instance merging when unspecified-length arrays
  differed. #656 (1.8.1/1.7.4)
* Fix oslc crash/assertion when indexing into a closure type. #663
* A version of spline() where the knots were colors without derivs
  and the abscissa was a float with derivs, computed its result
  derivatives incorrectly. #666
* Fix incorrect dropping of certain optional texture() arguments. #671
  (1.8.2/1.7.5)
* Fix: matrix parameters initialized with matrix(space,1) did not
  initialize properly. #675 (1.8.2/1.7.5)
* Bug fix: matrix parameters initialized with matrix(space,1) did not
  initialize properly. #675 (1.8.2/1.7.5)
* oslc: better error reporting for use of non-function symbol as a
  function. #684 (1.8.2/1.7.5)

Build & test system improvements and developer goodies:
* Default build is now C++11! Currently, the project will still build as
  C++03, but to do so you must explicitly 'make USE_CPP11=0'. #615 (1.8.0)
* Build: better handling of multiple USE_SIMD options. #617 (1.8.0/1.7.2)
* CMake FindOpenEXR.cmake totally rewritten, much simpler and better.
  #621 (1.8.0/1.7.2)
* 'make CODECOV=1' on a gcc/clang system will run the testsuite and
   generate a code coverage report in build/$PLATFORM/cov/index.html
   #624 (1.8.0)
* Remove support for LLVM 3.2 or 3.3. Now the minimum supported version
  is LLVM 3.4. #627 (1.8.0)
* Changed the CMake variable OSL_BUILD_CPP11 to simply USE_CPP11 (matching
  the name used for the Makefile wrapper). #634 (1.8.0)
* Clear up inconsistencies with USE_LIBCPLUSPLUS vs OSL_BUILD_USECPLUSPLUS
  #639 (1.8.0)
* Windows build improvements: Fixed build breaks #643 (1.8.0); fix problems
  with lexer #646 (1.8.0/1.7.3).
* Improvements in finding OpenEXR installations. #645 (1.8.0)
* Travis tests now include gcc 6 tests. #647 (1.8.0)
* Fixes for build breaks and warnings for gcc6. #647 (1.8.0/1.7.3)
* Fix build breaks for certain older versions of OIIO.
* Failed to properly propagate the C++ standard flags (e.g. -std=c++11)
  when turning llvm_ops.cpp into LLVM bitcode. #674 (1.8.2/1.7.5)
* Suppress compile warnings when using clang 3.9. #683 (1.8.2/1.7.5)
* Improve search for Partio. #689 (1.8.3)
* More robust finding of LLVM components. (1.8.3)
* OSL now builds properly with LLVM 3.9 and 4.0. #693 (1.8.3)
* When available (and with the right compiler version combinations), OSL
  will rely on Clang library internals to "preprocess" oso input, rather
  than Boost Wave. This solves problems particularly on OSX and FreeBSD
  where clang/C++11-compiled OSL was having trouble using Boost Wave if
  Boost was not compiled in C++11 mode (which is difficult to ensure if
  you don't control the machine or build boost yourself). #715 (1.8.5)

Documentation:
* Various typos fixed.
* Explain rules for connections.
* Explain that getattribute retrieves userdata.
* Document that `#once` works in shader source code.
* The CHANGES and INSTALL files have been changed from plain text to Markdown.



Release 1.7.5 -- 1 Nov 2016 (compared to 1.7.4)
--------------------------------------------------
* Fix oslc crash/assertion when indexing into a closure type. #663
* A version of spline() where the knots were colors without derivs
  and the abscissa was a float with derivs, computed its result
  derivatives incorrectly. #666
* Incorrect dropping of certain optional texture arguments. #671
* Bug fix: matrix parameters initialized with matrix(space,1) did not
  initialize properly. #675
* Suppress warnings when compiling with clang 3.9. #683
* oslc: better error reporting for use of non-function symbol as a
  function. #684
* Filed to properly propagate the C++ standard flags (e.g. -std=c++11)
  when turning llvm_ops.cpp into LLVM bitcode. #674

Release 1.7.4 -- 1 Aug 2016 (compared to 1.7.3)
--------------------------------------------------
* Bug fix: incorrect instance merging when unspecified-length arrays
  differed. (#656)
* Make oslc understand hex integer constants (like 0x01fc). #653

Release 1.7.3 -- 1 Jul 2016 (compared to 1.7.2)
--------------------------------------------------
* ShadingSystem attribute "userdata_isconnected", if set to a nonzero
  integer, causes a shader call to isconnected(param) to return 1 if the
  parameter is potentially connected to userdata (that is, if lockgeom=0
  for that parameter). This if for particular renderer needs, and the
  default remains for isconnected() to return 1 only if there is an
  actual connection. #629 (1.8.0/1.7.3)
* oslc now will warn when assigning the result of a "comma operator",
  which almost always is a programming error where the shader writer
  omitted the type name when attempting to construct a vector or matrix.
  #635 (1.8.0/1.7.3)
* Fix oslc incorrectly emitting init ops for constructor color(float).
  #637 (1.8.0/1.7.3)
* Bug fix: don't test or set the "layer ran bits" on the last layer of the
  group. #626 (1.8.0/1.7.3)
* Windows fixes: build breaks #643; fixes for lexer #646.
* Fixes for clean compile with gcc 6.x. #647

Release 1.7.2 -- 1 Mar 2016 (compared to 1.7.1)
--------------------------------------------------
* Build: better handling of multiple USE_SIMD options. #617
* oslc bug: getmatrix() failed to note that the matrix argument was
  written. #618
* CMake FindOpenEXR.cmake totally rewritten, much simpler and better. #621


Release 1.7 [1.7.1] -- 29 Jan 2016 (compared to 1.6)
--------------------------------------------------
Language, standard libary, and compiler changes (for shader writers):
* Language features:
   * Support for closure arrays in shader inputs. #544 (1.7.0)
   * Disallow quote strings from spanning lines (just like you can't in
     C).  This was never intended, but was accidentally allowed by the
     oslc lexer.  #613 (1.7.1)
* Standard library additions/changes:
   * linearstep() and smooth_linearstep() functions. #566 (1.7.0)
   * isconstant(expr) returns 1 if the expression can be turned into
     a constant at runtime (with full knowledge of the shader group's
     parameter values and connections). This is helpful mostly for advanced
     shader writers to verify their assuptions about which expressions can
     end up being fully constant-folded by the runtime optimizer. #578 (1.7.1)
   * New behavior of isconnected(parmaname): return 1 if the parameter
     is connected to an upstream layer, 2 if connected to a downstream layer,
     3 if both, 0 of neither. #589 (1.7.1)
   * int hash(string) returns an integer hash of a string. #605 (1.7.1)
   * int getchar(string s,int i) returns the integer character code of the
     i-th character in string s. #589 (1.7.1)
* Removed from stdosl.h all closure function declarations that were not
  documented in the spec and not implemented in testrender. (1.7.0)
* osutil.h changes:
   * New draw_string() rasterizes a string! #606 (1.7.1)

API changes, new options, new ShadingSystem features (for renderer writers):
* shade_image() utility runs a shader to generate every pixel of an ImageBuf.
  (1.7.0)
* ShaingSystem::optimize_group() is a new method that lets you force a
  shader group to optimize/JIT (without executing it). (1.7.0)
* Remove deprecated ShadingSystem::state() method. #551 (1.7.0)
* Changed logic of lazy shader layer evaluation: "unconnected" layers
  are now lazy. That means that shaders that appear to have no connected
  outputs, and don't set globals or set renderer outputs, are no longer
  run unconditionally. #560 (1.7.0)
* ShadingSystem attribute additions/changes:
   * ShadingSystem attribute "lockgeom" default changed to 1. (1.7.0)
   * SS::getattribute new attributes "num_attributes_needed",
     "attributes_needed", "attribute_scopes", and
     "unknown_attributes_needed" can be used to query for a shader group
     what OSL getattribute queries it might make. #537 (1.7.0)
   * "llvm_debug_layers" controls debug printfs for layer enter/exit
     (strictly for our debugging of the system; not intended for
     users). (1.7.0)
   * "opt_passes" controls the number of optimization passes over each
     instance (default: 10). This is mainly for developer debugging, you
     should not need to change or expost this. #576 (1.7.1)
   * "connection_error" controls whether a bad connection is considered
     a true error (or is just ignored). #583 (1.7.1)
* Shader group attribute additions/changes:
   * Group attribute "entry_layers" can explicitly provide a list of
     which layers within the group (specified by layer names) are "entry
     points", and then new ShadingSystem methods execute_init(),
     execute_layer(), and execute_cleanup() can be used to call them
     individually in any order.  The old execute(), which executes
     starting with the "root" and lazily evaluating any upstream nodes
     needed, still works and is still the usual way to run shaders for
     most purposes. #553 (1.7.0)
* RendererServices:
   * Remove has_userdata() method, which was unused. #603 (1.7.1)
* osl.imageio.so -- an OIIO image reader plugin that runs OSL shaders to
  generate pixels in an image (or texture). #531 (1.7.0)

Performance improvements:
* Much faster (10x or more, for large shaders) parsing of oso files (and
  .osl source as well). #511 (1.7.0)
* Optimize away unused optional arguments to noise functions. #548 (1.7.0)
* Improved runtime optimization via better tracking of variable aliasing
  across certain basic block boundaries. #585,588 (1.7.1)

Bug fixes and other improvements:
* oslinfo now has a "--runstats" flag to report how much time it takes
  to OSLQuery the shaders. (Mostly used for benchmarking the speed of
  OSLQuery internals.) #510 (1.7.0)
* Improved "debug_uninit" mode eliminates certain false alarms, and when
  uninitialized values are found, the error messages is more clear about
  which shader group and layer were involved (in addition to the previous
  reporting of the OSL source file ane line number). #512 (1.7.0)
* Bug fix serializing shader groups with "unsized arrays." #539 (1.7.0)
* Make sure OSL works fine with LLVM 3.5. #542 (1.7.0)
* Bug fix with unconditional running of earlier layers. (1.7.0)
* Improved error message for range errors and uninitialized variable errors
  (when using "range_checking" and "debug_uninit", respectively). #540 (1.7.0)
* Fixed oslc bug where shader parameters that were structs and whose
  values were initialized by copying another struct whole, generated
  incorrect initialization code. (1.7.0)
* Better tracking of layers that are the unused duplicates after an
  instance merge. This allows us to be sure those layers are skipped,
  and avoid situations where they look like they should run
  unconditionally.  #506 (1.7.0)
* Record oslc options in a comment at the start of the oso file. #575 (1.7.1)
* Better error messages when an closure is used that is unknown to the
  renderer. #576 (1.7.1)
* Fix potential SIGFPE from divide-by-zero in shaders. #596 (1.7.1)
* Improve optimizer logic behind lifetime analysis inside loops, fixing some
  subtle optimizer bugs. #598 (1.7.1)

Build & test system improvements and developer goodies:
* Allow testshade to test OSL snippets without whole full shaders. #564 (1.7.0)
* Much less verbose builds, when not using VERBOSE=1. (1.7.0)
* New build-time options:
  * USE_NINJA=1 will use Ninja rather than Make (faster!). #546 (1.7.0)
  * USE_fPIC=1 will build with -fPIC. (1.7.0)
  * OSL_BUILD_TESTS=0 will cause it to not build unit tests, testshade,
    or testrender. (1.7.0)
  * OSL_NO_DEFAULT_TEXTURESYSTEM=1 will cause the ShadingSystem to not
    create a raw OIIO::TextureSystem, and instead hit an assertion if
    a TextureSystem is not passed to the ShadingSystem constructor.
    Use this when building OSL to embed in renderers that supply their
    own TextureSystem class rather than using OIIO's. (1.7.0)
* Fixes when building for VS2012. #543 (1.7.0)
* Add --path to testshade and testrender to allow setting search path
  to find oso files. #549 (1.7.0)
* Testsuite: the presence of a file called "OPTIMIZEONLY" in a test
  directory will cause that test to run only in optimized mode. (1.7.0)
* Better detection of clang version. #559 (1.7.0)
* testshade --offsetst and --scalest let you make custom st ranges for
  the image being sampled (not restricted to 0-1 defaults). #565 (1.7.0)
* C++11 compatibility. #573 (1.7.1)
* Ensure proper build on Apple Clang 6.1. #580 (1.7.1)
* CMake build option OSL_BUILD_PLUGINS, if set to OFF, will disable building
  of any OSL plugins (such as the osl.imageio.so). #587 (1.7.1)
* CMake build option OSL_BUILD_TESTS, if set to OFF, will disable building
  of standalone test programs. #587 (1.7.1)
* We now use Travis-CI to do continuous integration and testing of every
  checkin and PR for a variety of platform / compiler / option combinations.
  #592,595,600 (1.7.1)
* Get rid of the "make doxygen" targer, which did nothing. (1.7.1)
* Build scripts automatically use ccache if available and in an obvious
  location. This can be forcefully disabled by setting USE_CCACHE=0.
  #595,597 (1.7.1)
* Use deprecation attribute for the methods that are considered deprecated.
  #602,607,608 (1.7.1)
* Testsuite organization overhaul now avoids copying test source to build.
  #612 (1.7.1)
* New macro OSL_CPLUSPLUS_VERSION (set to 03, 11, 14, etc) rather than
  using OSL_USING_CPP11. #581 (1.7.1)

Documentation:



Release 1.6 -- 26 May 2015 (compared to 1.5)
-----------------------------------------------
Language, standard libary, and compiler changes (for shader writers):
* It is now supported to have shader parameters that are arrays of
  unspecified length (like 'float foo[] = {...}'). Their length will be
  fixed when they are assigned an instance value of fixed length, or when
  connected to another layer's output parameter of fixed length. If there
  is no instance value nor a connection, the length will be fixed to the
  number of elements given in the default initializer. #481 #497 (1.6.4)
* The "comma operator" now works in the way you would expect in other
  C-like languages. #451 (1.6.2)
* The "and", "or", and "not" keywords are synonyms for the &&, ||, and !
  operators, respectively. (If this seems strange, note that this is
  true of C++ as well.) #446 (1.6.2)
* oslc will now silently ignore trailing commas in function or shader
  parameter lists and metadata lists. #447 (1.6.2)
* The "gabor" varieties of noise were found to have a serious mathematical
  bug, which after fixing results in a change to the pattern. We are hoping
  that they were rarely used, but beware a change in appearance. (1.6.1)
* We have clarified that shader metadata may be arrays (still not structs
  or closures) and fixed the implementation to support this (in addition
  to several other ways that metadata was brittle). (1.6.2)
* Spec clarification: emission() has units of radiance, so a surface
  directly seen by the camera will directly reproduce the closure weight
  in the final pixel, regardless of it being a surface or volume. Thus,
  you don't need to weight it by PI. #427 (1.6.2)
* oslc bug fixed in variable declaration + assignment of nested structs.
  #453 (1.6.2)

API changes, new options, new ShadingSystem features (for renderer writers):
* New ShadingSystem attribute int "profile", if set to 1, will include
  in the runtime statistics output a measure of how much total time was
  spent executing shaders, as well as the identities and times of the 5
  shader groups that spent the most time executing.  Note that this
  option is 0 (off) by default, since turning it on can add
  significantly to total runtime for some scenes.  #418 (1.6.2)
* New per-ShaderGroup attributes:
   * You can retrieve information about all closures that an optimized group
     might create using new queries "num_closures_needed", "closures_needed",
     "unknown_closures_needed". This could be useful for renderers by, for
     example, noticing that a material can't possibly call the "transparent"
     closure, and thus concluding that it must be opaque everywhere and thus
     shadow rays need not execute shaders to determine opacity. (#400) (1.6.0)
   * You can retrieve information about all "globals" (P, N, etc.) that an
     optimized group needs using new queries "num_globals_needed" and
     "globals_needed". This could be useful for renderers for which
     computing those globals is very expensive; examining which are truly
     needed and only computing those. (#401) (1.6.0)
* It is now possible to specify per-group "renderer_outputs" in the form
  "layername.paramname", thus designating a parameter of a specific layer
  as a render output that should be preserved, but excluding identically
  named output variables in other layers. #430 (1.6.2)
* New ShadingSystem runtime option int "lazy_userdata", when set to
  nonzero, causes userdata to be interpolated lazily (on demand) rather
  than at the beginning of a shader that needs it. For some renderer
  implementations, this may be a performance improvement. #441 (1.6.2)
* New OSLCompiler::compile_buffer() method compiles OSL source in a
  string into oso stored in a string. When combined with ShadingSystem::
  LoadMemoryCompiledShader, this allows an application to go straight from
  osl source code to executing shader without writing disk file or launching
  a shell to invoke oslc. #444 (1.6.2)
* OSLQuery: in the OSLQuery::Parameter structure, the 'validdefault' field
  now reliably is true for a valid default value, false if no default value
  could be statically determined (for example, if the param default is
  the result of code that must run at runtime). (1.6.2)
* Extend ShadingSystem::get_symbol() to retrieve the symbol from a
  particular named layer (rather than always getting it from the last
  layer in which the name is found). #458 (1.6.2)
* New SS option "opt_merge_instances_with_userdata", which if set to 0 will
  prevent instance merging for instances that access userdata. If you don't
  know why you'd want this turned off, you don't want it turned off!
  #459 (1.6.2)
* New RendererServices::supports() function allows for future expansion
  of querying (by the ShadingSystem) of features supported by a
  particular renderer. #470 (1.6.2)
* New ShadingSystem::execute() variety that takes a ShadingContext* rather
  than a ShadingContext&. If you pass NULL for the context pointer, the
  SS will automatically allocate (and eventually release) a context for
  you. #471 (1.6.2)
* New RendererServices API methods added that supply a handle-based
  version of texture(), texture3d(), environment(), and get_texture_info(),
  in addition to the existing name-based versions. Renderer implementations
  will need to add these, if they overload the name-based ones. #472 (1.6.2)
* RendererServices API methods for texture have been revised again, now
  each variety takes a filename (will always be passed) and also a
  TextureHandle* and TexturePerthread* (which may be NULL). Renderers
  may use the name only if that's easiest, but also may use the handle when
  supplied. #478 (1.6.3)
* New ShadingSystem methods: find_symbol() and symbol_address() tease apart
  the functionality of the (now deprecated) get_symbol(). The fairly
  expensive name lookup is in find_symbol(), and it can be done once per
  shader group (and reused for the lifetime of the group), while the
  relatively inexpensive symbol_address() can be cheaply done after each
  shader execution. If your renderer is currently using get_symbol() to
  retrieve the shader output values after each shade, we strongly recommend
  changing to the find_symbol() [once up front] and symbol_address() [once
  per execution] combination for improved performance. #495 (1.6.4)
  
Performance improvements:
* Many noise() varieties have been sped up significantly on
  architectures with SIMD vector instructions (such as SSE). Higher
  speedups when returning vectors rather than scalars; higher speedups
  when supplying derivatives as well, higher speedups the higher the
  dimensionality of the domain. We're seeing 2x-3x improvement for
  noise(point), depending on the specific variety, 3x-4x speedup for 4D
  noise varieties. #415 #422 #423 (1.6.1, 1.6.2)
* Change many transcendental functions (sin, cos, exp, etc) to use
  approximations that are must faster than the 100% correct versions in
  libm. The new ones may differ from the old in the last decimal place,
  which is probably undesirable if programming ICBM guidance systems,
  but seems fine for rendering when the result is many times faster to
  evaluate those functions. You can turn it back to the fully precise and
  slow way by building with OSL_FAST_MATH=0. #413 (1.6.0)
* New runtime optimization of mix(a,a,x) -> a (even if none of them are
  constant). #438 (1.6.2)
* Fix performance regression: avoid incrementing profile-related stats on
  every execution if profiling is disabled. (1.6.2)
* When the runtime optimizer can figure out which textures are being
  called, it will retrieve the handle once and generate code for
  handle-based RendererServices callbacks, rather than name-based
  ones. This speeds up texture-heavy shading by 5-10% by eliminating the
  need for the name-to-handle lookups for every texture call. #472 (1.6.2)
* New runtime optimization: constant folding of sincos(). #486 (1.6.4)
* New runtime optimization: constant folding of matrix construction from
  16 floats. #486 (1.6.4)
* Improved optimizations -- caught some additional places where assignments
  could be found to be useless and eliminated. #403 (1.6.4)
* Improved speed of runtime optimization for large shaders. #499 (1.6.5)
* Improved threading performance by not locking during pointcloud_search
  and pointcloud. (1.6.5)
* Dramatic speedup of runtime optimization of large shaders (5x for
  big shaders (1.6.5)
* Speed up loading of large shaders from disk (1.6.5).

Bug fixes and other improvements:
* oslinfo --param lets you ask for information on just one parameter of
  a shader. #397 (1.6.0)
* oslc now properly catches and issues error messages if you define a struct
  with duplicate field names. #398 (1.6.0)
* Some math speedups (especially those involving sqrt). (1.6.0)
* Improved detection of shader groups that optimize away to nothing. (1.6.2)
* Fix of possible (but never reported) bug with instancing merging.
  #431 (1.6.2)
* Fix incorrect oslc code generation when assigining an element of an
  array-of-structs to another element of array-of-structs. #432 (1.6.2)
* Bug fix in code generation of constant folding of getmatrix() calls
  (never reported symptomatic). #440 (1.6.2)
* Fixes to constuction of temporary filename used for group serialization
  archive. (1.6.2)
* Internals fix: logb() incorrectly declaring derivative functions (it
  has no derivs). #437 (1.5.12/1.6.2)
* Catch serialization errors resulting from failed system() calls to
  zip/tar the archive. #436 (1.5.12/1.6.2)
* Bug fix: Delay instance merging from ShaderGroupEnd until optimize
  time, it messed with the serialization. #426 (1.5.12/1.6.2)
* Fix OSLQuery crash when reading .oso file of a compiled shader that had
  local variables but no parameters. #435 (1.5.12/1.6.2)
* Shader-wide metadata is now correctly output by oslc, and correctly
  read by OSLQuery. #455 (1.6.2)
* Only merge instances when optimize level >= 1. #456 (1.6.2)
* Bug fix: runtime optimizer should not consider a layer unused if its
  outputs are renderer outputs. Also fix subtle bug in detecting whether
  an output param is a renderer output. #457 (1.6.2)
* Clear derivatives properly in the implementaiton of gettextureinfo.
  #463 (1.6.2)
* Correct handling of point-like shader parameters defined with default
  values that use the constructors that provide "space" names. #465 (1.6.2)
* Fixed some subtle runtime optimizer bugs. #485 #491 (1.6.4)
* Using "debugnan" mode now also checks for NaNs in the results of
  interpolated userdata. #490 (1.6.4)
* Fixed bug in generating code for closure optional keyword parameters.
  #492 (1.6.4)
* Fixed bug where instances were incorrectly merged if they had a parameter
  that was never used in the shader, but was connected downstream, and each
  of the two instances had a difference instance value for that parameter.
  #493 (1.6.4)
* Fix critical bug that in very particular circumstances could result in
  shader layers not executing when they should. (1.6.5)

Under the hood:
* Change a couple long-deprecated OIIO calls to modern versions.
  #448 (1.5.12/1.6.2)
* Change runtime code generation, putting TextureOptions, NoiseParams,
  TraceOptions in the ShadingContext, which avoided a dynamic alloca
  which was tickling an LLVM code generation bug. #442 (1.5.12/1.6.2)
* Allow memory-aligned closure component classes. (1.6.2)
* Lots of internal refactoring of llvm_ops.cpp in order to better and
  more performantly support MCJIT and newer LLVM releases. #468 (1.6.4)

Build & test system improvements and developer goodies:
* Windows build fix for compiling shaders finding the right oslc.exe.
  #399 (1.6.0/1.5.11)
* Fix thread contention issue in testshade that was making benchmarks
  inaccurate. $407 (1.6.0)
* Build scripts better help messages. (1.6.0)
* Changes to build correctly against LLVM 3.5, though please note that
  using LLVM 3.5 seems to require C++11. If you have C++98 toolchain, then
  please continue using LLVM 3.4. #412 (1.6.0)
* testrender improvements: update the microfacet models to the latest
  improved visible normal sampling technique of d'Eon & Heitz, stratified
  sampling, simplification of Oren-Nayar, and speed improvements.
  #425 (1.6.2)
* Build fixes for recent OIIO changes. (1.5.12/1.6.2)
* testshade -group now accepts the name of a file containing the serialized
  group. #439 (1.6.2)
* Various fixes for Windows and Visual Studio 2010. #461 #462 #464 (1.6.2)
* testshade -param can now specify arrays. (1.6.4)
* Remove the USE_MCJIT build flag. Now we ALWAYS compile with ability to
  use MCJIT any time we are using LLVM that's new enough to have it, and
  a renderer can select MCJIT at runtime by setting the SS attribute "mcjit"
  to nonzero (the default, 0, will use old JIT). When compiling against
  LLVM 3.6 or newer, which has removed MCJIT, OSL will always use MCJIT.
  #487 #488 (1.6.4)
* Fixes to work with LLVM 3.5 and 3.6, though 3.6 especially has not yet
  been very thoroughly tested for OSL. #487 (1.6.4)
* C++11 support: the preprocessor symbol OSL_BUILD_CPP11 is set to 1 if
  OSL was built with C++11 (or higher) support, and the separate
  OSL_USING_CPP11 is set to 1 if C++11 is detected right now. (These can
  be different from the point of view of an app using OSL headers.)
  #508 (1.6.6RC1)

Documentation:
* Short chapter added to spec describing the syntax of our serialization
  protocol.
* Documentation fix: surfacearea() should return float, not int. (1.5.12)
* Clarified that the way to use gettextureinfo() to retrieve matrices from
  textures created from rendered images (or shadow maps) is with the tags
  "worldtocamera" (for the world-to-3D-camera) and "worldtoscreen" (for
  the world-to-2D-projection that runs -1 to +1 in x and y).
* Clarified documentation on gettxtureinfo(file,"exists"). (1.6.2)
* Improved the comments in shaderglobals.h to explain in detail the meaning
  and use of all the fields. #489 (1.6.4)



Release 1.5.12 -- 26 Dec 2014 (compared to 1.5.11)
--------------------------------------------------
* Build fixes for 32 bit machines.
* Documentation fix: surfacearea() should return float, not int.
* Bug fix: Delay instance merging from ShaderGroupEnd until optimize
  time, it messed with the serialization.
* Build fixes for recent OIIO changes.
* Fix OSLQuery crash when reading .oso file of a compiled shader that had
  local variables but no parameters.
* Catch serialization errors resulting from failed system() calls to
  zip/tar the archive.
* Fix logb() incorrectly declaring derivative functions (it has no derivs).
* Change a couple long-deprecated OIIO calls to modern versions.

Release 1.5.11 -- 22 Sep 2014 (compared to 1.5.10)
--------------------------------------------------
* Windows compilations fixes for very MSVC 2013.
* Windows build fix for compiling shaders finding the right oslc.exe.
  #399
* Speedup from telling gcc/clang to ignore the fact that math functions
  can set errno (helps code generation of sqrtf in particular.
* Fix thread contention issue in testshade which could make it hard to
  use testshade as a benchmark with many threads.
* Make it build properly with LLVM 3.5.
* Fix compiler warnings about signed/unsigned comparisons.



Release 1.5 (1.5.10) -- July 30, 2014 (compared to 1.4)
----------------------------------------------
Language, standard libary, and compiler changes (for shader writers):
* New closure function for microfacet BSDFs:
      closure color microfacet (string distribution, normal N,
                              float alpha, float eta, int refract)
      closure color microfacet (string distribution, normal N, vector U,
                          float xalpha, float yalpha, float eta, int refract)
  These replace the growing zoo of microfacet_blah functions (using
  different distribution values such as "beckmann" or "ggx", as well as
  selecting refraction when refract!=0). The old varieties are now considered
  deprecated and will eventually be removed (but will alow back-compatibility
  of compiled oso files).
* Remove documentation about unimplemented hash() function -- it was never
  implemented, and we can't figure out what it's good for. (#359) (1.5.7)
* Documented the specific names that should be used with getattribute() to
  retrieve common camera parameters. We should expect these to be the same
  for all renderers (though individual renderers may also support additional
  attribute queries). #393 (1.5.9)

API changes, new options, new ShadingSystem features (for renderer writers):
* ShadingSystem API changes:
   * ShadingSystem::Parameter() has a new variety that can set lockgeom=0
     for a particular parameter at runtime, even if it was not declared
     this way in the shader's metadata.  This allows an app to effectively
     say "I intend to vary this shader's parameter value, don't optimize as
     if it were a a constant." (1.5.1)
   * New ShadingSystem::ReParameter() lets you change the instance value of
     a shader parameter after the shader group has already been declared
     (but only if the shader has not yet been optimized, OR if the
     parameter was declared with lockgeom=0). This, in combination with the
     new SS::Parameter() variety, lets an app select a parameter for
     variation, compile the shader network, and then continuously modify
     the value and re-render without needing to redeclare and recompile the
     shader group (though at a slightly higher execution cost than if the
     parameter were fully optimized. (1.5.1)
   * ShadingAttribState has been simlified and replaced with ShaderGroup.
     And ShadingAttribStateRef replaced by ShaderGroupRef. (1.5.1)
   * ShaderGroupBegin returns the ShaderGroupRef for the group, there is
     no need to call state() to get the group reference. (1.5.1)
   * New ShadingSystem::getattribute() and attribute() varieties that takes
     a ShaderGroup* as a first argument, allows setting and retrieval of
     attributes pertaining to a specific shader group. (#362, #368) (1.5.8)
     SEE SECTION BELOW ON GROUP-SPECIFIC ATTRIBUTES
   * New ShaderGroupBegin() variant takes a text description of a shader
     group. (3.5.8) (#379) For example:
         ShaderGroupBegin ("groupname", "surface",
                           "param float fin 3.14; " /*concatenate string*/
                           "shader a alayer;"
                           "shader b blayer;"
                            "connect alayer.f_out blayer.f_in;");
   * ShadingSystem methods renderer() and texturesys() retrieve the
     RendererServices and TextureSystem, respectively. (1.5.8)
   * ShadingSystem statistics output now prints the version and major
     runtime options. (1.5.8) (#369)
* New/changed ShadingSystem global attributes:
   * Attributes "archive_groupname" and "archive_filename", when set, will
     cause the named group to be fully archived into a .tar, .tar.gz, or
     .zip file containing the serialized description of the group, and all
     .oso files needed to reconstruct it. This is great for shipping
     debugging or performance benchmark cases to the developers! It's also
     a good way to "extract" a shader network from the midst of a complex
     renderer and then be able to reconstitute it inside a debugging test
     harness like testshade. (1.5.8) (#381) For more detail and explanation,
     please see https://github.com/imageworks/OpenShadingLanguage/pull/381
* New per-ShaderGroup attributes:
   * ShadingSystem::getattribute(group,...) allows queries that retrieve
     all the names and types of userdata the compiled group may need
     ("num_userdata", "userdata_names", "userdata_types"), as well as the
     names of all textures it might access and whether it's possible that
     it will try to acccess textures whose names are not known without
     executing the shader ("num_textures_needed", "textures_needed",
     "unknown_textures_needed"). (#362) (1.5.8)
   * Retrieving the group attribute "pickle" (as a string) gives you the
     serialized description of the shader group in the same format that is
     accepted by ShaderGroupBegin(). (1.5.8) (#381)
   * Group-wise attribute "renderer_outputs" (array of strings) lets you
     specify different renderer outputs for each shader group in addition
     to the ShadingSystem-wide "renderer_outputs" that apply to all groups.
     Remember that naming the renderer outputs prevents those from being
     optimized away if they are not connected to downstream
     shaders. (1.5.8) (#386)
   * New group attribute queries: "groupname", "num_layers", "layer_names"
     allow you to ask for the name of the group, how many layers there are,
     and the names of all the layers.
* ShaderGlobals changes:
   * Added a RendererServices pointer to ShaderGlobals, to eliminate a
     bunch of pointless indirection during shader execution. (1.5.2)
   * Split the definition of ShaderGlobals into its own header file,
     shaderglobals.h. (1.5.1)
* RendererServices changes:
   * RendererServices::get_texture_info method parameters have changed, now
     includes an additional ShaderGlobals* parameter (as did all the other
     texture functions already). (1.5.5)
   * Split RendererServices definition into separate rendererservices.h. (1.5.6)
   * Add RendererServices::texturesys() that returns a pointer to the
     RS's TextureSystem. (1.5.6)
   * Add a ShaderGlobals* parameter to the various get_matrix and
     get_inverse_matrix methods. This makes it easier for renderers that
     might need the ShaderGlobals (or access to the void* renderstate
     contained therein) to look up matrices, especially if the renderer
     supports different per-primitive transformation name bindings.
     #396 (1.5.10)
   * Change the last 4 RenderServices methods that took a void* renderstate
     to instead take a ShaderGlobals *. It's more general, since the SG
     does itself contain the renderstate pointer, and some implementations
     or features might need the other contents of the SG. #396 (1.5.10)
* OSLQuery changes:
   * OSLQuery::Parameter has a new 'structname' field, that for struct
     parameters reveals the name of the structure type. The oslinfo utility
     has also been updated to print this information. (1.5.4)
   * OSLQuery now may be initialized by a ShaderGroup*, which allows an
     app with a live shading system to query information about shaders
     much more efficiently than reloading and parsing the .oso file from
     disk. #387 (1.5.8)
   * OSLQuery API has replaced most of its std::string parameters and
     members with string_view and ustring, to speed it up and eliminate
     lots of small allocations. This may introduce some small source
     incompatibilities with the old OSLQuery, but they should be very
     easy to fix in your code.  #387 (1.5.8)
* Miscellaneous/general:
   * Use OIIO's new string_view for many public API input parameters that
     previously took a std::string& or a const char*. (1.5.7) (#361)
   * De-virtualized OSLCompiler and ShadingSystem classes -- it is now safe to
     construct them in the usual way, the create() and destroy() functions for
     them are optional/deprecated. (#363) (1.5.8)

Performance improvements:
* Dramatically speed up LLVM optimization & JIT of small shaders --
  reducing the per-shader-network compile overhead of small shaders
  from 0.1-0.2s to less than 0.01s per shader group. (1.5.1)
* Runtime constant folding of substr(). (1.5.6)
* Improved code generation for texture calls, removing redundant setting
  of optional texture parameters when they don't change from the default
  values. (1.5.6)
* Remove lock from RendererServices texture calls that would create a big
  thread botteneck (but only for renderers that used the RS base class
  texture calls; those that overrode them would not notice). (1.5.6)
* Constant folding of certain assignments to individual components or
  array elements. (#356) (1.5.7)
* Much more complete constant-folding of format(). (1.5.8) (#366)
* Enhanced simplification of component and array assignment, when
  multiple assignments actually fill out the entire array or aggregate.
  For example: 'color C; C[0] = 3.14; C[1] = 0; C[2] = 42.0;'
  Now it recognizes that all components are assigned and will replace
  the whole thing (and subsequent uses of C) with a constant
  color. (1.5.8) (#367)
* Eliminate redundant retrieval of userdata within a shading
  network. (1.5.8) (#373)

Bug fixes:
* testshade/testrender timer didn't properly break out execution time
  from setup time. (1.4.1/1.5.1)
* Guard matrix*point transformations against possible division by zero.
  (1.4.1/1.5.1)
* Fix subtle bug with splines taking arrays, where the number of knots
  passed was less than the full length of the array, and the knots had
  derivatives -- the derivatives would be looked up from the wrong spot
  in the array (and could read uninitialized portions of the array).
  (1.4.1/1.5.3)
* Fix incorrect function declarations for the no-bitcode case (only
  affected Windows or if a site purposely built without precompiled
  bitcode for llvm_ops.cpp). (1.4.1/1.5.3)
* When using Boost Wave for oslc's preprocessor, fixed it so that
  variadic macros work. (1.5.4)
* Ensure that isconnected() works for struct parameters. (1.5.4)
* Fix incorrect data layout in heap for certain array parameters.
  (1.5.5/1.4.2)
* oslc -D and -U did not work properly when OSL was built against
  Boost Wave as the C preprocessor. (1.5.5/1.4.2)
* oslc bug fix for innitializing array elements inside a parameter that's a
  struct. (1.5.6)
* Implementation of the luminance() function has been improved to
  guarantee that all the components sum to exactly 1.0. (1.5.6)
* OSLQuery: for array parameters whose source code only specify default
  values for partially initializing the array, nonetheless make the
  default vectors be the correct length for the array (with zeroes for
  array elements without explicit initialization). (1.5.6)
* oslc: negating closures resulted in hitting an assertion. (1.5.6)
* ShadingSystem - bug fix when Parameter() sets the same parameter twice
  for the same instance, the second time back to the default
  value. (1.5.8) (#370)
* Fix debug printing crash when group name was not set. (1.5.7)
* Bug fix with runtime optimization instance merging -- differing
  "lockgeom" status of parameters should invalidate the merge. (1.5.8) (#378)
* Runtime optimizer bug fix: output parameter assignment elision was
  overly aggressive when the output parameter in question was a renderer
  output (and therefore should not be elided). (1.5.8) (#378)
* Fix printf("%f",whole_float_array) to work properly. (1.5.8) (#382)
* oslc improperly allowed ONE too many initializers for arrays. (1.5.8) (#383)

Under the hood:
* Internals refactor that separates the runtime optimization from the
  LLVM generation and JIT. (1.5.1)
* Removed legacy built-in closures that weren't actually used. (1.5.1)
* Split all the direct calls to LLVM API into a separate LLVM_Util
  class, which has no other OSL dependencies and may be used from other
  apps to simpilfy use of LLVM to JIT code dynamically. (1.5.1)
* Stop using deprecated TypeDesc #defines. (1.5.4)
* A shader's printf, warning, and error output is now buffered so that
  all the "console" output from a shader invocation will appear
  contiguous in a log, rather than possibly line-by-line interspersed
  with the output of another shader invocation running concurrently in a
  different thread (1.5.5).
* By de-virtualizing OSLCompiler and ShadingSystem classes, we make it
  much easier in the future to change both these classes (at least, adding
  new methods) and changing the Impl internals, without breaking link
  compatibility within release branches. (#363) (1.5.8)
* Eliminate the need for a special dlopen when using OSL from within
  a DSO/DLL plugin of another app. (1.5.8) (#374, #384)

Build & test system improvements and developer goodies:
* Make OSL work with LLVM 3.4 (1.5.4)
* New llvm_util.h contains an LLVM_Util standalone utility class that can
  be used from any app as a simplified wrapper for LLVM code generation
  and JIT. (1.5.1)
* testshade consolidates --iparam, --fparam, --vparam, --sparam into
  just --param that figures out the types (or could be explicitly
  told. (1.5.1)
* More robust detection of clang, for older cmake versions.
  (1.4.1/1.5.1)
* Handle 3-part version numbers for LLVM installations. (1.4.1/1.5.1)
* Fix USE_EXTERNAL_PUGIXML versus USING_OIIO_PUGI issue. (1.4.1/1.5.3)
* Remove all ehader include guards, switch to '#pragma once' (1.5.2)
* Adjust build to account for new library names in OpenEXR 2.1. (1.5.3)
* Make sure we include all OpenEXR and Ilmbase headers with <OpenEXR/foo.h>
  and remove the addition of OpenEXR sub-folder from system search paths.
  (1.5.1)
* Better detection of LLVM version number from the LLVM installation.
  (1.5.1)
* Support for using LLVM's "MCJIT" instead of "old JIT" (controlled by
  using the optional 'make USE_MCJIT=1'). This is mostly experimental,
  and not recommended since performance of old JIT is still much better
  than MCJIT. (1.5.6) Fixes in (1.5.8) (#385)
* Moved the source of all public header files to src/include/OSL, to more
  closely mimic the install file layout. (1.5.6)
* Fixes for compilation warnings with newer gcc. (1.5.7) (#351)
* Fixes for IlmBase 2.1 build breakages. (#354) (1.5.7)
* Make testshade multithreaded, so it's always exercising the thread
  safety of the shading system. (1.5.8) (#365)
* Compatibility with cmake 3.0. (1.5.8) (#377)
* Fix warnings with some gcc versions. (1.5.8)
* Make testrender accept the new shadergroup syntax. (1.5.9)
* Make testrender & testshade support all the standard camera getattribute
  queries. (1.5.9)
* testshade/testrender: have get_attribute fall back on get_userdata,
  and have get_userdata bind "s" and "t" to make it easy to test
  shader that rely on userdata. #395 (1.5.10)
* Fixes for Boost 1.55 + Linux combination, where it needed -lrt on the
  link line. #394 (1.5.10)
* Windows build fix for compiling shaders finding the right oslc.exe.
  #399 (1.5.11)

Documentation:
* Clarified docs about floor(). (1.5.8)
* Clarified that oren_nayar() is the name of the function, not
  orennayar (it was not consistent). (1.5.8) (#375)
* backfacing() was previously left out of the docs. (1.5.8) (#376)
* Documented the specific names that should be used with getattribute() to
  retrieve common camera parameters. We should expect these to be the same
  for all renderers (though individual renderers may also support additional
  attribute queries). #393 (1.5.9)
* Clarify that we intend for getattribute() to be able to retrieve
  "userdata" (aka primitive variables) if no attribute is found with the
  matching name. #395 (1.5.10)




Release 1.4.1 - 19 Dec 2013 (compared to 1.4.0)
-----------------------------------------------
* Guard matrix*point transformations against possible division by zero.
* Fix subtle bug with splines taking arrays, where the number of knots
  passed was less than the full length of the array, and the knots had
  derivatives -- the derivatives would be looked up from the wrong spot
  in the array (and could read uninitialized portions of the array).
* testshade/testrender - Fix timer call that wasn't correctly reporting
  execution time versus total time.
* Fix incorrect function declarations for the no-bitcode case (only
  affected Windows or if a site purposely built without precompiled
  bitcode for llvm_ops.cpp).
* Build: more robust detection of clang, for older cmake versions.
* Build: handle 3-part version numbers for LLVM installations.
* Build: Fix USE_EXTERNAL_PUGIXML versus USING_OIIO_PUGI issue.


Release 1.4.0 - 12 November 2013 (compared to 1.3)
--------------------------------------------------

Language, standard libary, and compiler changes (for shader writers):

* Simplex noise: new noise varieties "simplex" and "usimplex" are 
  faster than Perlin noise (and MUCH faster for the 4-D variety),
  but has a somewhat different appearance.  Judge for yourself, use
  whichever noise suits your purpose.
* oslc now catches mismatches between the format string and the types
  of arguments to format(), printf(), fprintf(), warning(), and error().
* oslc now gives a compile error when trying to use a function as
  if it were an ordinary variable. (1.3.2)
* Improvement to oslc function type checking -- allow more nuanced
  matching of polymorphism based on return types.  First we check for
  identical return types, then we test allowing non-identical spatial
  triples to match, then finally we allow any return type to match.
* New ShadingSystem attribute "debug_uninit", which when set to nonzero
  tries to detect use of uninitialized variables at runtime.  (Off by
  default because it's very expensive, just use if you are debugging.)
* Implemented the OSL aastep() function, which was always described in
  the docs but never implemented.
* The texture lookup functions all take new optional parameters,
  "missingtexture", (color), "missingalpha", (float), which when supplied
  will substitute that color and alpha for missing or broken textures,
  rather than treating it as an error condition.
* New gettextureinfo() queries: "datawindow", "displaywindow" (both take
  an int[4] as destination argument).

ShadingSystem API changes and new options (for renderer writers):

* New ShadingSystem attribute "renderer_outputs" enumerates the renderer
  outputs (AOVs) explicitly, so that elision of unused outputs can be done
  with full understanding of which params will or will not be sued.
  Beware: if you enable "opt_elide_unconnected_outputs", you MUST pass
  "renderer_outputs" or your AOVs may be eliminated!

Performance improvements:

* "Middleman" optimization notices when you have 3 layers A,B,C with
  connections A.Aout -> B.Bin and B.Bout -> C.Cin, an Bout is simply
  copied unconditionally from Bin.  In this case, Aout is connected
  directly to Cin, potentially leading to further optimizations, including
  layer B getting culled entirely.
* Layer-to-layer global propagation: Notice cases where an upstream
  output is guaranteed to be aassigned a global variable (such as N or u)
  and then alias the downstream parameter to that global (eliminating
  the connection and possibly culling the upstream layer entirely).
* Improve constant propagation from layer to layer (it used to do so only
  if the output of the earlier layer was never used in the upstream layer;
  but actually it's ok for it to be read, as long as it's not rewritten).
* Runtime constant-folding of cellnoise() calls.
* Local variables that are arrays initialized with float, int, or string
  constants have been greatly optimized.  If that's the only time anything
  is written into the array, it will be turned into a true static constant
  with no per-shader-invocation initialization or allocation costs.
  Thus, it is no longer expensive to use a large array of constants as a
  table... it may look like a per-shade cost to set up the array, but by
  the time the optimizer gets done with it, it is not.
* Improved runtime constant-folding of max/min (now works for the integer
  case, as well as previously working for the float-based types).
* Runtime constant-folding of cos, sin, acos, asin, normalize, inversesqrt,
  exp, exp2, expm1, log, log2, log10, logb, erf, erfc, radians, degrees.
* Optimization that speeds up certain closure manipulations and can
  cut overhead especially for relatively simple shaders.
* New peephole optimizations for two common patterns: 
  (1) add A B C ; sub D A C => [add A B C;] assign D B
  and the same if the order of add & sub are switched.
  (2) OP A B... ; assign A C => OP C B...  as long as A is not subsequently
  used and OP fully overwrites A and writes to no other variables.
* Constant folding of Dx, Dy, Dz, area, and filterwidth, when passed a
  constant (the deriv will be 0 in that case).
* Substantial multithread speedups by eliminating testing of some shared
  variables.
* Constant folding of add & sub for the mixed float/triple cases.
* Speed up closure construction by combining the closure primitive with
  its (nearly inevitable) multiplication by weight, into a single internal
  operation.
* The OSLQuery library has been sped up by 20x when querying the parameters
  of large shaders.

Bug fixes:

* Fix 'debugnan' tendency to give false positives for certain assignments
  to arrays and multi-component variables (colors, points, etc.).
* The integer versions of min(), max(), and clamp() were not working
  properly. (1.3.1)
* fmod(triple,triple,float) was not working properly in Windows. (1.3.2)
* Avoid incorrect heap addressing when retrieving a symbol's address. (1.3.2)
* Fix for a bug where an early 'return' from a shader could lead to
  output parameters not being properly copied to the inputs of their
  downstream connections. (1.3.2)
* Fix bug where filterwidth() didn't properly mark itself as needing
  derivatives of its argument. (1.3.2)
* Bug fix: Was not properly initializing the default values of output
  params if they were written but never subsequently read.
* Better error handling for missing pointclouds.
* Fixed improper initialization of arrays of closures.
* Fixed constant folding bug for the split() function.
* Fixed constant folding bug for pow(x,2.0).

Under the hood:

* Eliminate use of custom DEBUG preprocessor symbol in favor of the more
  standard NDEBUG (beware: its sense is reversed).
* All references to and vestiges of TBB have been removed.

Build & test system improvements and developer goodies:

* testshade: --options lets you set arbitrary options to the ShadingSystem
  for test runs.
* Added CMake option 'ENABLERTTI' that enables use of RTTI for sites that
  must link against an RTTI-enabled LLVM (the default is to assume LLVM
  is built without RTTI, so we don't use it either). (1.3.1)
* Fix some ambiguity about which variety of shared_ptr we are using -- 
  this could produce problems in apps that include the OSL headers but also
  use different shared_ptr varieties.
* Fix a variety of build issues and warnings under Windows.
* Work around CMake bug where on some platforms, CMake doesn't define the
  'NDEBUG' when making the 'RelWithDebInfo' target (which is sometimes
  used for profiling). (1.3.1)
* Add Make/CMake option 'USE_EXTERNAL_PUGIXML' which if set to nonzero
  will find and use a system-installed PugiXML rather than assuming that
  it's included in libOpenImageIO.  When used, it will also use the
  environment variable PUGIXML_HOME (if set) as a hint for where to find
  it, in case it's not in a usual system library directory. (1.3.1)
* Nicer debug printing (for developers) when an op doesn't have an
  associated source line. (1.3.2)
* Reorder include directories for building LLVM bitcode so that the
  openimageio, ilmbase and boost directories are ahead of any system
  directories given by llvm-config --cxxflags.  This avoids the LLVM
  bitcode getting compiled against different library versions than the
  rest of the code if multiple versions are installed.
* Have debug_groupname and debug_layername apply to LLVM debugging info
  as well.
* Support for LLVM 3.3.
* Fix compiler warnings for clang 3.3 and libc++.
* CMake files now more carefully quote assembled file paths to more
  gracefully handle paths with spaces in them.
* CMakeLists.txt has been moved to the top level (not in src/) to conform
  to the usual CMake conventions.

Documentation:

* Eliminate references to OSL random() function; it was never
  implemented and nobody seemed to miss it or have a compelling use
  case. (1.3.1)




Release 1.3.3 - 11 July 2013 (compared to 1.3.2)
------------------------------------------------
* Fix bug in the implementation of filterwidth() when passed a constant
  (for which no derivative could be taken).
* Changes to support LLVM 3.3.


Release 1.3.2 - 19 Jun 2013 (compared to 1.3.1)
-----------------------------------------------
* fmod(triple,triple,float) was not working properly in Windows.
* Nicer debug printing (for developers) when an op doesn't have an
  associated source line.
* Avoid incorrect heap addressing when retrieving a symbol's address.
* oslc new gives a compile error when trying to use a function as
  if it were an ordinary variable.
* Fix for a bug where an early 'return' from a shader could lead to
  output parameters not being properly copied to the inputs of their
  downstream connections.
* Fix bug where filterwidth() didn't properly mark itself as needing
  derivatives of its argument.


Release 1.3.1 - 15 May 2013 (compared to 1.3.0)
-----------------------------------------------
* The integer versions of min(), max(), and clamp() were not working
  properly.
* Added CMake option 'ENABLERTTI' that enables use of RTTI for sites that
  must link against an RTTI-enabled LLVM (the default is to assume LLVM
  is built without RTTI, so we don't use it either).
* Work around CMake bug where on some platforms, CMake doesn't define the
  'NDEBUG' when making the 'RelWithDebInfo' target (which is sometimes
  used for profiling).
* Docs: Eliminated discussion of random(), which was never implemented
  and we can't think of a good use case.
* Add Make/CMake option 'USE_EXTERNAL_PUGIXML' which if set to nonzero
  will find and use a system-installed PugiXML rather than assuming that
  it's included in libOpenImageIO.  When used, it will also use the
  environment variable PUGIXML_HOME (if set) as a hint for where to find
  it, in case it's not in a usual system library directory.


Release 1.3.0 - 14 Feb 2013 (compared to 1.2)
----------------------------------------------

Language, standard libary, and compiler changes (for shader writers):

* pointcloud_write() allows shaders to write data and save it as a point
  cloud.
* spline now accepts a "constant" interpolation type, which interpolates
  with discrete steps.
* isconnected(var) returns true if var is a shader parameter and is
  connected to an earlier layer. This is helpful in having shader logic
  that can discern between a connected parameter versus a default.
* Whole-array assignment is now supported.  That is, if A and B are arrays
  of the same type and length(A) >= length(B), it is legal to write
  A=B without needing to write a loop to copy each element separately.
* stoi() and stof() convert strings to int or float types, respectively
  (just like the identically-named functions in C++11).  These do properly
  constant-fold during runtime optimization if their inputs can be deduced.
* split() splits a string into tokens and stores them in the elements of an
  array of strings (and does constant-fold!).
* distance(point A, point B, point Q) gives the distance from Q to the
  line segment joining A and B.  This was described by the OSL spec all
  along, but somehow was not properly implemented.

ShadingSystem API changes and new options (for renderer writers):

* Default implementation of all the pointcloud_*() functions in
  RendererServices, if Partio is found at build time.
* ShadingSystem attribute "compile_report" controls whether information
  should be sent to the renderer for each shading group as it compiles.
  (The default is now to be more silent, set this to nonzero if you want
  to see the old per-group status messages.)
* ShadingSystem attribute "countlayerexecs" adds debugging code to count
  the total number of shader layer executions (off by default; it slows
  shader execution down slightly, is only meant for debugging).
* Add a parameter to OSLCompiler::compile to specify the path to the
  stdosl.h file.
* New call: ShadingSystem::LoadMemoryCompiledShader() loads a shader with
  OSO data from a memory buffer, rather than from a file.

Performance improvements:

* Reduce instance symbol memory use, lowering OSL-related memory by 1/2-2/3
  for large scenes.
* Identical shader instances within a shader group are automatically
  merged -- this can have substiantial performance and memory
  improvements for very complex shader groups that may (inadvertently,
  for convenience, or as an artifact of certain lookdev tools) have
  multiple instances with identical input parameters and connections.
  We have seen some shots that render up to 20% faster with this change.
* Better constant-folding for pointcloud_get (if indices are known).
* Speed up implementation of exp() for some Linux platform (works around
  a known glibc issue).
* Speed up of mix() by making it a built-in function (rather than defined
  in stdosl.h) and doing aggressive constant folding on it. Also some special
  cases to try to avoid having mix() unnecessarily trigger execution of
  earlier layers when one or both of its arguments is a connected shader
  param and the mix value turns out to be 0 or 1.
* Runtime optimization now includes constant-folding of getattribute()
  calls, for those attributes which are scene-wide (such as renderer options
  that apply to the whole frame).
* Improvements in the algorithm that tracks which symbols need to carry
  derivatives around results in MUCH faster compile times for extremely
  complex shaders.
* Improve bad performance for shaders that call warning() prolifically, by
  imposing a maximum number of shader warnings that are echoed back to the
  renderer (controlled by ShadingSystem attribute "max_warnings_per_thread",
  which defaults to 100; 0 means that there is no maximum).

Bug fixes and minor improvements:

* Fix incorrect oso output of matrix parameter initialization. (This did
  not affect shader execution, but could result in oslquery/oslinfo not
  reporting default matrix parameter values correctly.)
* Bug fix: oslinfo didn't print int metadata values properly.
* oslc better error checking when calling a variable as if it were a
  function.
* Parsing of oso files was broken for compiled shaders that had string
  metadata whose metadata value contained the '}' character.
* Pointcloud function bug fixes: now robust to being passed empty filenames,
  or calling pointcloud_get when count=0.
* Fix crash with C preprocessor when #include fails or no shader function
  is defined.
* Improve error reporting when parsing oso, especially in improving the
  error messages when an old renderer tries to load a .oso file created by
  a "newer" oslc, and the oso contains new instruction opcodes that the
  older renderer didn't know about.
* oslc now gives a helpful error message, rather than hitting an assertion,
  if you try to index into a non-array or non-component type.
* Broken "XYZ" color space transformations because of confusion between
  "xyz" and "XYZ" names.
* oslc: correctly catch errors with integer literals that overflow.
* oslc: proper error, rather than assertion, for 
  'closure color * closure color'.
* oslc: proper handling of preprocessor errors when stdosl.h is in a path 
  containing spaces.
* Fix bugs in the OSL exit() function.
* Don't error/warn if metadata name masks a global scope name.

Under the hood:

* Simplify code generation of binary ops involving closures.

Build & test system improvements and developer goodies:

* Many, many fixes to enable building and correct running of OSL on
  Windows.  Too many to list individually, let's just say that OSL 1.3
  builds and runs pretty robustly on Windows, older versions did not.
* Remove unused OpenGL and GLEW searching from CMake files.
* Fix a variety of compiler warnings, particularly on newer compilers,
  32 bit platforms, and Windows.
* Find LLVM correctly even when using "svn" LLVM versions.
* Adjust failure threshold on some tests to account for minor platform
  differences.
* New minimum OIIO version: 1.1.
* CMake fixes: Simplify searching for and using IlmBase, and remove
  unneeded search for OpenEXR.
* Find Partio (automatically, or using 'make PARTIO_HOME=...'), and if
  found, use it to provide default RendererServices implementations of
  all the pointcloud-related functions.
* Make/CMake STOP_ON_WARNING flag (set to 0 or 1) controls whether the
  build will stop upon any compiler warning (default) or if it should
  keep going (=0).
* Fixed bad interaction between weave preprocessor, OIIO::Strutil::format,
  and gcc 4.7.2.
* Fixes for static linkage of liboslcomp.
* Addressed compilation errors on g++ 4.7.2.
* Improved logic and error reporting for finding OpenImageIO at build time.
* Add support for linking to static LLVM libraries, and for specifying
  a particular LLVM version to be used.
* New BUILDSTATIC option for CMake/Make allows building of static OSL 
  libraries.
* Fixes to allow OSL to use LLVM 3.2.
* Fixes to allow OSL to be compiled by clang 3.2.
* Fixes to allow building on Windows with MinGW.
* Fix for building from a chroot where the host may be, e.g., 64 bit, but
  the target is 32 bit.
* Minimize cmake output clutter for things that aren't errors, unless
  VERBOSE=1.
* Various fixes to get OSL building and running on FreeBSD.
* Cmake/make option USE_LLVM_BITCODE to be able to force LLVM bitcode
  compilation on or off (defaults to on for Windows, off for Unix/Linux).

Documentation:
* Clarify that documentation is licensed under Creative Commons 3.0 BY.
* Clean up OSL Spec discussion of built-in material closures.
* Overhaul the suggested metadata conventions in the OSL spec (now 
  conforms to the conventions used by Katana).




Release 1.2.1 - 6 Nov, 2012 (compared to 1.2.0)
-----------------------------------------------
* Fix incorrect oso output of matrix parameter initialization. (This did
  not affect shader execution, but could result in oslquery/oslinfo not
  reporting default matrix parameter values correctly.)
* Build: remove unused OpenGL and GLEW searching from CMake files.
* Build: Fix a variety of compiler warnings, particularly on newer compilers,
  32 bit platforms, and Windows.
* Build: Find LLVM correctly even when using "svn" LLVM versions.
* Bug fix: oslinfo didn't print int metadata values properly.


Release 1.2.0 - Aug 30, 2012 (compared to 1.1)
----------------------------------------------
Note: this is more or less the production-hardened version of OSL that
was used to complete Men in Black 3, The Amazing Spider-Man, and Hotel
Transylvania.

New tools/utilities:
* New program "testrender" is a tiny ray-tracing renderer that uses OSL
  for shading.  Features are very minimal (only spheres are permitted at
  this time) and there has been no attention to performance, but it
  demonstrates how the OSL libraries may be integrated into a working
  renderer, what interfaces the renderer needs to supply, and how the
  BSDFs/radiance closures should be evaluated and integrated (including
  with multiple importance sampling).
* shaders/ubersurface.osl is an example of a general purpose surface
  shader.

Language, standard libary, and compiler changes:
* texture()/texture3d() support for subimage/face selection by name as
  well as by numeric index.
* getattribute() performs automatic type conversions that mostly
  correspond to the kind of automatic type casting you would get from
  ordinary assignments in OSL.  For example, getattribute("attrib",mycolor)
  used to fail if "attrib" turned out to be a float rather than a color;
  but now it succeeds and copies the float value to all three channels
  of mycolor.

ShadingSystem API changes and new options:
* Remove unused 'size' parameter from the register_closure API.

Optimization improvements:
* Constant-fold pointcloud_search calls when the position is a constant if
  the search returns few enough results, by doing the query at optimization
  time and putting the results into new constant arrays.
* Matrix parameters initialized by m=matrix(constA,constB,...) now 
  statically initialize, and no longer need to run initialization code
  each time the shader executes.

Bug fixes and minor improvements:
* Fix pointcloud_search to optionally take a 'sort' parameter, as
  originally documented.
* Unit tests weren't properly run as part of the testsuite.
* Track local+temp memory usage of optimized shaders and consider it an 
  error if a shader needs more than a maximum amount at runtime, set with
  the "max_local_mem_KB" attribute.
* Add pointcloud statistics.
* Fix derivative error for sincos() when the inputs have no derivatives but
  the outputs do.
* Bug fix to vector-returning Gabor noise (it could previously generate
  different values for different platforms).
* printf() of closures built from other closures allows for proper
  recursive printing of the closure tree.

Build & test system improvements and developer goodies:
* Simplify the namespace scheme.
* Remove support for certain old dependencies: OIIO versions < 0.10,
  LLVM < 3.0, and Boost < 1.40.
* Lots of little fixes to solve compiler warnings on various compilers.
* Support for newer OSX releases, particularly if /usr/bin/cpp-4.2 is not
  found.
* Better support for Boost::wave (portable C preprocessor replacement).
  Build with 'make USE_BOOST_WAVE=1' to force Wave use instead of system
  cpp.
* You can select a custom LLVM namespace with 'make LLVM_NAMESPACE=...'.
* Symbols that are not part of the OSL public APIs are now hidden from the
  linker in Linux/OSX if the CMake variable HIDE_SYMBOLS is on.
* New Makefile/CMake option LLVM_STATIC can be used to use static LLVM
  libraries rather than the default dynamic libraries.
* Support for LLVM 3.1.
* Support for building with Clang 3.1 (lots of warning fixes).
* Makefile/CMake variable EXTRA_CPP_DEFINITIONS allows you to inject
  additional compiler flags that you may need to customize the build for
  your site or a choice of unusual compiler.
* Add support for 'PROFILE=1' builds that are appropriate for use with
  a profile.



Release 1.1.0 - Mar 14, 2012 (compared to 1.0.0)
------------------------------------------------
Language, standard libary, and compiler changes:
* Allow closures as parameters to closures.
* New constants: M_2PI, M_4PI
* Generic noise: noise("noisetype",coords,...)
* Gabor noise (anisotropic, automatically antialiased) via noise("gabor").
* Fix mod/fmod discrepancy: fmod() now matches C, mod() always returns a
  positive result like in RSL.
* Allow "if (closure): and "if (!closure)" to test if a closure is empty
  or not.
* New optional parameter to trace(): "traceset" allows you to specify a
  named geometry set for tracing.

ShadingSystem API changes and new options:
* New "greedyjit" option will optimize & JIT all shader groups up front,
  concurrently, without locking.
* Add a way to name shader groups.
* attribute("options",...) lets you set a bunch of options at once.
* Options to enable/disable individual optimizations (mostly useful for
  debugging)

Optimization improvements:
* Allow block alias tracking on non-constants when it's safe.
* Track "stale" values to eliminate pointless assignments.
* Eliminate redundant "useparam" ops.
* Assignments to output parameters that are not connected to any
  downstream layers are now eliminated.
* More aggressive elision of ops that only write to symbols that won't
  be subsequently used.
* More careful identification and removal of parameters (input and output)
  that are both unused in the shader and not connected downstream.

Bug fixes and minor improvements:
* Minor blackbody fixes.
* Bug fix: don't mark constants as having their derivatives taken.
* Clamp splineinverse() for out-of-knot-range x input.
* Bug fix: the optimization of "a=b; a=c" was incorrect if c was an
  alias for a (it incorrectly eliminated the first assignment).
* Bug fix: work around LLVM thread safety issues during JIT.
* Bug fix: symbol_data() wasn't returning the right address for non-heap
  parameters.
* Bug fix: optimization errors related to break, continue, and return not
  properly marking the next instruction as a new basic block.
* Bug fix: luminance() with derivatives didn't work.
* Bug fix: in code generation of structure initializers.
* Improved error messages from ConnectShaders.
* Bug fix: type checking bug could case non-exactly-matching polymorphic 
  functions to coerce a return value even when that was not intended.
* Type checking improvements: Make sure point-point is a vector
  expression, and point+vector & point-vector are point expressions.

Build & test system improvements and developer goodies:
* testsuite overhauls:
    - run each test both optimized and not
    - generate all tests in build, not directly in ./testsuite
    - greatly simplify the run.py scripts
* Much more detailed debugging logs of the optimization process.
* Upgrade to clang/llvm 3.0.
* Lots of infrastructure to make debugging the optimizer easier.
  Including new options debug_groupname, debug_layername, only_groupname.
* Improved the build system's LLVM-finding logic.
* Fix warnings from gcc 4.6.



Release 1.0.0 - Oct 12, 2011
----------------------------
* Modified testshade (and the underlying SimpleRender class) to handle
  several standard named coordinate systems such as "camera", "screen",
  "NDC", "raster."
* blackbody() and wavelength_color().
* New ShadingSystem configuration attribute: "colorspace" lets you explain
  to OSL what RGB really means (e.g., "Rec709", "sRGB", "NTSC", etc.).
  The luminance(), blackbody(), wavelength_color, and conversion to/from
  XYZ now takes this into account correctly.
* rotate()  (always in spec, never implemented)



Release 0.6.2 - Sept 29, 2011
-----------------------------
* Statistics overhaul -- added optimization stats, eliminated unused ones.
* Allow a shader parameter to mask a global built-in function with
  only a warning, and improve scope conflict errors by pointing out the
  file and line of the previous definition.
* Altered the RendererServices API to add transform_points() method, which
  allows renderers to support nonlinear transformations (i.e., those that
  are not expressible as a 4x4 matrix).
* Issue a renderer error when unknown coordinate system names are used
  (can be turned of by setting the new ShadingSystem attribute 
  "unknown_coordsys_error" to false).
* New OSL built-in function: splineinverse().


Release 0.6.1 - Sept 20, 2011
-----------------------------
* Be more aggressive in freeing shader instance memory that's no longer
  needed after optimization and LLVM JIT.  This greatly reduces
  OSL-related memory consumption for scenes with large numbers of very
  complicated shading networks.
* Add Dz() which is helpful for apps that use OSL to shade volumes.  At
  present, we only correctly compute Dz(P), all other Dz() queries
  return 0.
* Additional statistics on how many instances and groups compile, and
  how many are empty after all optimizations are performed.
* Make sure all the relevant statistics can be queried via
  ShadingSystem::getattribute.


Release 0.6.0 - Sept 9, 2011
----------------------------
* ShadeExec API overhaul -- an app using it no longer needs
  ShadingSystemImpl internal knowledge.
* Thread-parallel runtime optimization and LLVM JIT of different shader
  groups.
* Optimizations: runtime constant folding of arraylength and regex_search,
  new instruction 'arraycopy' will copy an entire array at once.
* Renamed patterns.h to oslutil.h.
* Do not generate unnecessary code when optional texture parameters are set
  to their default values.
* Restore long-lost ability for layers to run unconditionally (not lazily)
  if they were marked as "non-lazy" (for example, if they write to globals.
* Make the "debugnan" attribute work again -- when turned on, code will
  be inserted after every op to be sure that no NaN or Inf values are
  generated, and also verify that shader inputs (globals) don't have NaN
  or Inf values passed in by the renderer.  A similar facility existed a
  long time ago, but we lost that functionality when we switched from
  the interpreter to LLVM.
* Looks for release versions of LLVM-2.9 (allows using a Macports LLVM 
  installation).


Release 0.5.4 - Jul 21, 2011
----------------------------
* Several fixes related to arrays of structs, and structs containing
  other structs.
* Fixed arrays of closures.
* Removed support for old LLVM 2.7, nobody seemed to be using it any more.
* Changed the definition of dict_find() to return -1 for an invalid
  dictionary, to distinguish it from 0 meaning that the query failed but
  the dictionary itself was valid.
* Make array parameters safe to convert to constants during runtime
  optimization.
* Support derivatives in pointcloud searches.
* Fixed several runtime optimizer bugs.
* Fixed a bug where environment() calls with an optional "alpha" parameter
  that has derivatives was overwriting memory.
* Fixed code generation bug for a*=b and a/=b.
* Automatically initialize all local string variables to NULL, to avoid
  bad pointers for uninitialized strings.
* Bug fix: dict_value() wasn't properly marking its argument as writable.
* Automatic range checking of array and component access.
* Fix uninitialized derivatives for pointcloud().
* Speed up getattribute() calls by caching information about failed
  getattribute queries in the ShadingContext.
* Fix to constant folding of gettextureinfo: a failed lookup should not
  fold, because we want the error to occur in shader execution, not during
  optimization.
* Refactor, clean up, and comment testshade.
* oslc now gives an error on non-void functions that fail to return a value.
* Fixed implementation of area() that could generate an assertion.
* Fix escape sequences in string literals: we were handling it correctly
  for printf-like format strings, but not other string literals.
* break and continue now work properly (just like in C/C++).
* You can now return from anywhere in a user function (multiple times if
  you want), just like C/C++, and are no longer restricted to the only
  return statement being the last statement of the function.
* New include file for shaders: patterns.h.  Now, it only includes a handy
  'wireframe()' function, but will expand for other useful things.
* New function: int getmatrix(from,to,M) is like the matrix(from,to)
  constructor, but returns a success value so a shader can tell if the
  named coordinate systems failed to be found by the renderer.


Release 0.5.3 - Apr 19, 2011
----------------------------
* Fix missing derivatives for sign() function.
* Fix closure color type size (crashes).
* Fix bug with environment() when passed "alpha" pointers with derivatives.
* Improve error messages for getmessage/setmessage to catch the most
  common sources of non-deterministic behavior.
* Bug fix when constant-folding gettextureinfo().
* Fix mismatched prototype for subsurface() closure.
* Texture errors encountered in shader constant folding are now properly
  reported to the renderer.
* Allow functions to have array parameters of unspecified length.
* Fix subtle bug related to lifetime analysis of variables in loops (led
  to incorrect optimizations).


Release 0.5.2 - Mar 14, 2011
----------------------------

* Windows: use boost::wave instead of external cpp; various other Windows
  compilation fixes.
* texture & environment now take an optional "interp" parameter that
  overrides the interpolation/filtering method (valid arguments:
  "smartcubic", "cubic", "linear", "closest").
* Bug fixes to getmessage() and its handling of derivatives, which includes
  a slight RendererServices API change.
