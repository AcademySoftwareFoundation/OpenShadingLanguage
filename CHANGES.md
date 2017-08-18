Release 1.8.11 -- 11 Sep 2017 (compared to 1.8.10)
--------------------------------------------------
* Builds properly against LLVM 5.0.

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
