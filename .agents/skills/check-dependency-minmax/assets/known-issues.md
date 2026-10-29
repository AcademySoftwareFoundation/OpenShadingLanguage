# Known issues from past audits of OSL (update this list)

Problems found by `check-dependency-minmax` audits, with the date found,
their status, and the right fix once known. Mark resolved items instead of
deleting them while the lesson is still useful (for example, a fix that was
tried and had to be reverted).

## Found 2026-10-26

- Stale `desc`s (VP2025/OptiX OIIO from container is 3.1, icx container
  LLVM is 15, latest-releases gcc is 14, bleeding-edge LLVM is 18, macOS
  Apple clang versions, MacOS-26 LLVM 23, MacOS-15-Intel OIIO is `dev-3.0`,
  Windows-2022 OIIO `release` is 3.2). FIXED in desc only, 2026-10-26.
  (Bleeding edge and latest releases have since moved to LLVM 23.)
- `LIBTIFF_VERSION`, `PTEX_VERSION`, `FREETYPE_VERSION` in ci.yml: no OSL
  script reads them (only OIIO's own gh-installdeps.bash). FIXED
  2026-10-26 by removing them (to pin these, pass OIIO cmake
  `-D<Pkg>_BUILD_VERSION` via `OPENIMAGEIO_CMAKE_FLAGS` instead).
- `OSL_HARDENING=3`, meant for the latest-releases gcc job's `setenvs`,
  was pasted below the next job's `- desc:` line, so it became part of the
  clang job's name and was never set. Fixed 2026-10-26 by moving it back
  to the end of the gcc job's `setenvs`.
- `pybind11_ver` pins on bleeding edge / latest releases / clang14 build
  pybind11 but the bindings auto-select nanobind (OIIO >= 3.2 and
  Python >= 3.10), so pybind11 master/latest is never tested. FIXED
  2026-10-27 for bleeding edge with `python_bindings_backend: both`
  (others still build pybind11 unused).
- latest-releases jobs lag: LLVM 18.1.7 (pinned), CMake 3.31.6 (runner),
  fmt 12.1.0, nanobind 3.0.1, Python 3.12 (system). PARTLY FIXED
  2026-10-27: CMake 4.4.4, fmt 12.2.0 (needs `fmt_commit`, OIIO verifies
  tag hashes), LLVM 23 (see below). nanobind 3.0.1 and Python 3.12 OPEN.
- LLVM >= 19 from KyleMayes/install-llvm-action (the llvm.org
  `LLVM-X-Linux-X64.tar.xz` releases) is unusable: static libs are ThinLTO
  bitcode, GNU ld fails "file format not recognized", no libLLVM.so.
  Tried 21 that way 2026-10-27 and reverted. RIGHT FIX: apt.llvm.org via
  `LLVM_APT_VERSION=<major>|main` in `setenvs` (gh-installdeps.bash).
  latest releases and bleeding edge now use LLVM 23 this way.
- Bleeding edge on LLVM main (24) fails to build (API changes) and needs
  `VERSION_MAX` raised in externalpackages.cmake. Deferred 2026-10-27:
  bleeding edge uses 23 until OSL is adapted, then set
  `LLVM_APT_VERSION=main`. OPEN.
- Before 2026-10-27 no x86 job ran batched with LLVM > 18, which hid a
  bug since the LLVM 21 opaque pointer change: `type_wide_void_ptr()` was
  a scalar pointer instead of a vector of pointers, breaking every batched
  test (BatchedShaderGlobals layout validation). FIXED 2026-10-27. Lesson:
  keep batched on a job with the newest LLVM.
- clang 23 adds `-Wunused-but-set-global`, which broke testshade and
  testrender (dead globals). FIXED 2026-10-27 by removing them.
- MacOS-26-ARM has no `LLVMBREWVER`, gets Homebrew default LLVM; old desc
  said llvm21. No CI job uses LLVM 16, 19, 21, 22. OPEN.
- INSTALL minimums no job exercises: gcc 9.3, clang 5, MSVS 2017, icx 2022,
  OptiX 7.0, CUDA 9.0, Flex 2.5.35, Bison 2.7, nanobind 2.8.0, Qt5 5.6.
  Oldest job uses apt Flex/Bison and installs no Qt or partio. OPEN.
