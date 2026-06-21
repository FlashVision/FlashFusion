# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in FlashFusion, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Email the security concern to: security@flashvision.dev
3. Include a description of the vulnerability, steps to reproduce, and potential impact.
4. You will receive an acknowledgment within 48 hours.

## Response Timeline

- **Acknowledgment:** Within 48 hours of report
- **Assessment:** Within 7 days
- **Fix/Patch:** Within 30 days for confirmed vulnerabilities

## Scope

This policy applies to the FlashFusion library code. Third-party dependencies
(PyTorch, OpenCV, etc.) should be reported to their respective maintainers.

## Best Practices

When using FlashFusion in production:

- Keep dependencies up to date (`pip install --upgrade flashfusion`)
- Use `torch.load(..., weights_only=True)` when loading untrusted checkpoints
- Validate input sources before passing to prediction pipelines
- Run the library in sandboxed environments when processing untrusted data
