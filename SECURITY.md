# Security Policy

## Supported Versions

This gives guidance about which branches are supported with patches to
security vulnerabilities.

| Version / branch  | Supported                                            |
| --------- | ---------------------------------------------------- |
| main      | :white_check_mark: :construction: ALL fixes immediately, but this is a branch under development with a frequently unstable ABI and occasionally unstable API. |
| 1.14.x    | :white_check_mark: All fixes that can be backported without breaking ABI compatibility. New tagged releases monthly. |
| 1.13.x    | :warning: Only the most critical fixes, only if they can be easily backported. |
| <= 1.12.x | :x: No longer receiving patches of any kind. |


## Reporting a Vulnerability

If you think you've found a potential vulnerability in OSL, please report it
by emailing the project administrators at
[security@openshadinglanguage.org](security@openshadinglanguage.org). Only the
project administrators have access to these messages. Include detailed steps to
reproduce the issue, and any other information that could aid an
investigation. Our policy is to respond to vulnerability reports within 14
days.

Our policy is to address critical security vulnerabilities rapidly and post
patches as quickly as possible.


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

