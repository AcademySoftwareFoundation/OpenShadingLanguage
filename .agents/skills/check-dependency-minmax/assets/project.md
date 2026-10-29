# OSL project facts for check-dependency-minmax

Project-specific information used by `SKILL.md`. Keep this up to date;
the OpenImageIO copy of this file has the same sections with different
contents.

## Buildinfo tool

- Tool: `testshade`. `src/build-scripts/ci-test.bash` runs
  `$OSL_ROOT/bin/testshade --buildinfo` (and `testshade --help` before it).
  `testshade` has no `--version`; the version is on the first buildinfo
  line (`OSL 1.16.0.0dev | Linux/x86_64`).
- Local build: `find dist build -maxdepth 4 -name testshade -type f`.
- Besides the compiler and `Dependencies:` lines, OSL's buildinfo has a
  `Back ends:` line (e.g. `x86_64 single, batched b8_AVX2`) giving the
  batched-shading targets built, and a `Running on:` line describing the
  CI machine (not the build). The parser stops at `Running on:`.

## CI job conventions

- `oldest` jobs (analysis B): `Oldest everything ...`. The `Debug gcc9 ...`
  job shares many of its minimum pins but does not claim to be oldest.
- `latest releases` jobs (analysis C): `latest releases gcc... ` and
  `latest releases clang...`.
- `bleeding edge` job (analysis D): `bleeding edge ...`
  (`oiio/ocio/exr/pybind-main` in its desc).
- `VP20xx` jobs use the ASWF VFX Platform containers for that year
  (`aswf/ci-osl:20xx`), so most dependencies are container versions.
- `oiio-main`/`oiio-release`/`oiio-rel`/`oiio-3.x` in a desc refers to the
  `openimageio_ver` input (OIIO is built from that branch or tag).
- Jobs with no buildinfo: `abi check`, `clang-format`, and the Windows
  jobs (all `skip_tests: 1`); the OptiX job has buildinfo only when it runs
  on the ASWF GPU runners.

## Buildinfo data quirks

- Dependency names are CMake package names: `Python3`, `Qt5`/`Qt6`
  (separate keys), `BISON`, `FLEX`, `ZLIB`, `Robinmap`, `CUDAToolkit`.
- `OpenImageIO` versions have four parts (`3.1.18.1`); a main build
  reports the next unreleased version (e.g. `3.3.0.0`).
- `partio` and `Robinmap` are reported without a version (`present`).
- `CUDAToolkit` and `OptiX` appear only in the OptiX job.
- NOT reported by buildinfo: OpenEXR, OpenColorIO, fmt, libpng (these are
  dependencies of OIIO that CI may build from a pinned `*_ver`), NumPy.
  Confirm their versions from the build step of the job log when needed.
- The `C++` field is `C++<OSL_BUILD_CPP>/<__cplusplus>`, so the standard
  is reported correctly (no mislabeling of newer standards).

## Key dependencies for the description check (analysis E)

LLVM, OpenImageIO, Python, OpenEXR and OCIO (`exr3.x`, `ocio`, from the
job log, see above), pybind11/nanobind, and the batched back ends
(`batch-b16avx512` and the like in a desc vs. the `Back ends:` line).

## Where to find latest upstream releases (step 4)

| Dependency | Where to look |
|---|---|
| LLVM | llvm/llvm-project (`^llvmorg-[0-9]+\.[0-9]+\.[0-9]+$`) |
| OpenImageIO | AcademySoftwareFoundation/OpenImageIO (4-part `v3.x.y.z` tags; ignore `-beta`/`-RC`) |
| Imath, OpenEXR, OCIO | AcademySoftwareFoundation/{Imath,openexr,OpenColorIO} |
| pugixml | zeux/pugixml |
| partio | wdas/partio (tags only, no releases) |
| robin-map | Tessil/robin-map |
| fmt | fmtlib/fmt |
| zlib | madler/zlib |
| libpng | pnggroup/libpng (tags only, `^v1\.6\.[0-9]+$`) |
| pybind11 | pybind/pybind11 |
| nanobind | wjakob/nanobind (drop `-dev` tags) |
| Qt | qt/qtbase (`^v6\.[0-9]+\.[0-9]+$`; Qt5 is EOL, use INSTALL) |
| Flex | westes/flex |
| Bison | `https://ftp.gnu.org/gnu/bison/` (file listing) |
| OptiX | NVIDIA/optix-dev (header-only SDK tags) |
| CUDA | NVIDIA website; CI uses the toolkit in the container |
| CMake | Kitware/CMake (`^v[0-9]+\.[0-9]+\.[0-9]+$`) |
| NumPy | numpy/numpy |
| Python | python.org; CI uses runner/container Pythons |
