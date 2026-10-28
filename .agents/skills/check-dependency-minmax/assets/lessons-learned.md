# Lessons learned running the audit on OSL (update this list)

Project-specific tips. Lessons that apply equally to OIIO belong in
`SKILL.md` instead (see its "Project-specific files" section).

- `testshade --buildinfo` (with the `Dependencies:` list) was added to
  `ci-test.bash` in 2026-10; CI runs from before that have no buildinfo
  and cannot be audited with this skill.
- Windows jobs have no buildinfo; get OIIO/pugixml/Python/CMake from the
  `-- Found <pkg> <ver>` and `-- CMake X` lines in their logs.
- OpenEXR/OCIO/fmt/libpng: grep `-- All build dependencies:` and
  `Found <pkg> <ver>` in the log (from the OIIO build).
- Which binding backend was used: buildinfo lists `pybind11` or `nanobind`;
  the log also has `OSL_PYTHON_BINDINGS_BACKEND not set; auto-selected ...`.
