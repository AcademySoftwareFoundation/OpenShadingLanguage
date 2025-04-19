---
name: Build problems
about: I'm having trouble building OSL. Help!
title: "[BUILD]"
labels: ''
assignees: ''

---

**PLEASE DO NOT REPORT BUILD TROUBLES AS GITHUB "ISSUES" UNLESS YOU ARE RALLY SURE IT'S A BUG**

The best way to get help with your build problems is to ask a question on the
[osl-dev developer mail list](https://lists.aswf.io/g/osl-dev). For quick
questions, you could also try the [ASWF Slack](https://slack.aswf.io)
`#openshadinglanguage` channel.

When you email about this, please attach one or both of the following:
1. The full verbose build log, which you can create like this:
   ```make clean ; make VERBOSE=1 > build.log```
2. The `CMakeCache.txt` file from your build directory.

If you are pretty sure that you have identified a **BUG** or other actual
problem with the OSL build scripts and you can describe how to reproduce it,
please file a [bug report issue](https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/issues/new?template=bug_report.md).
