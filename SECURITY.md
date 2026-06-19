# Security Policy

## Supported Versions

This gives guidance about which branches are supported with patches to
security vulnerabilities.

| Version / branch  | Supported                                            |
| --------- | ---------------------------------------------------- |
| main      | :white_check_mark: :construction: ALL fixes immediately, but this is a branch under development with a frequently unstable ABI and occasionally unstable API. |
| 1.15.x    | :white_check_mark: All fixes that can be backported without breaking ABI compatibility. New tagged releases monthly. |
| 1.14.x    | :warning: Only the most critical fixes, only if they can be easily backported. |
| <= 1.13.x | :x: No longer receiving patches of any kind. |


## Reporting a Vulnerability

If you think you've found a potential vulnerability in OSL, please
report it to the maintainers. Include detailed steps to reproduce the issue,
and any other information that could aid an investigation.

The best way to report a vulnerability is to file a GitHub [security
advisory](https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/security/advisories/new).
If that is not possible, it is also fine to email your report to
security@openshadinglanguage.org. Only the project administrators have access
to these reports.

Our policy is to respond to vulnerability reports within 14 days, and to
address critical security vulnerabilities rapidly and post patches as quickly
as possible.


## What do we consider a vulnerability?

We only consider a situation to be a security vulnerability if an untrusted
party can plausibly trigger the flaw through normal product inputs (for
example, a maliciously crafted oso file that might compromise a renderer when
loaded). We do not support requesting a CVE for API-only or caller-controlled
failures with no realistic adversarial path.

The OSL project adopts the same security stance as many other language
compilers: we believe that shaders that will be JITed and executed are by
definition *trusted inputs*, and should not be accepted from untrusted
sources.  A shader that causes damage when it faithfully executes is not a
vulnerability per se.  In rare circumstances, we might consider it a
vulnerability if a maliciously crafted shader can cause the renderer or OSL
library to do something damaging that is different from what the shader text
implies.

Flaws whose root cause lies in a dependency should be reported and fixed
upstream; the upstream project owns the CVE when one is warranted.


## Other security features

### Signed tags

Starting with OSL 1.14.3.0, we cryptographically sign release tags.
To verify a tag, you can use the `git tag -v` command, which will check
the signature against the public key that is included in the repository.
For example,

```bash
git tag -v v1.14.3.0
```

## Outstanding Security Issues

None known


## History of CVE Fixes

None to date
