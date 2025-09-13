# Security Policy

## Supported Versions

ContentEngineAI is currently in pre-production (0.x versions). Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

**Note**: During the 0.x development phase, we recommend always using the latest released version for the most recent security fixes.

## Reporting a Vulnerability

We take security seriously and appreciate your help in keeping ContentEngineAI secure.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by:

1. **Email**: Send details to [stkzlv+ContentEngineAI@gmail.com](mailto:stkzlv+ContentEngineAI@gmail.com)
2. **GitHub Security Advisories**: Use GitHub's private vulnerability reporting feature
3. **GitHub Issues**: Only for non-sensitive security discussions

### What to Include

When reporting a security vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and attack scenarios
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Environment**: Your system details (OS, Python version, ContentEngineAI version)
- **Evidence**: Screenshots, logs, or proof-of-concept (if applicable)

### Response Timeline

We aim to respond to security reports according to the following timeline:

- **Initial Response**: Within 48 hours
- **Vulnerability Assessment**: Within 1 week
- **Fix Development**: Within 2-4 weeks (depending on severity)
- **Public Disclosure**: After fix is released and users have time to update

### Security Update Process

1. **Assessment**: We evaluate the report and determine severity
2. **Fix Development**: We develop and test a fix
3. **Coordinated Disclosure**: We work with the reporter on disclosure timing
4. **Release**: We publish a security update
5. **Advisory**: We publish a security advisory with details

## Security Best Practices

### For Users

When using ContentEngineAI, follow these security best practices:

#### API Keys and Secrets
- **Never commit API keys** to version control
- **Use environment variables** for all sensitive configuration
- **Rotate API keys regularly**
- **Use least-privilege access** for API keys

#### File System Security
- **Verify output directory permissions** before processing
- **Sanitize user input** when using custom configuration
- **Monitor disk usage** to prevent resource exhaustion
- **Use secure temporary directories** for processing

#### Network Security
- **Use HTTPS endpoints** for all API calls
- **Validate TLS certificates**
- **Implement rate limiting** for production deployments
- **Monitor network traffic** for unusual patterns

### For Developers

#### Code Security
- **Input validation**: Validate all user inputs and configuration
- **Dependency management**: Keep dependencies updated
- **Secure defaults**: Use secure configuration defaults
- **Error handling**: Don't expose sensitive information in errors

#### Development Environment
- **Use virtual environments** to isolate dependencies
- **Run security scans** regularly with our CI/CD pipeline
- **Keep development tools updated**
- **Use pre-commit hooks** for automated security checks

## Security Tools

ContentEngineAI includes automated security scanning:

### Automated Scans
- **Bandit**: Scans for common security issues in Python code
- **Safety**: Checks for known security vulnerabilities in dependencies
- **GitHub Dependabot**: Monitors for vulnerable dependencies
- **CodeQL**: Advanced semantic code analysis (planned)

### Running Security Scans

```bash
# Run all security checks
make security

# Individual tools
poetry run bandit -r src/
poetry run safety check
poetry run vulture src/ --min-confidence 80
```

## Known Security Considerations

### Current Limitations

1. **Web Scraping**: Browser automation may expose system information
2. **File Operations**: Processing user-provided files requires validation
3. **External APIs**: Dependent on third-party service security
4. **Local Processing**: FFmpeg and media processing use system resources

### Planned Security Enhancements

- [ ] Input sanitization framework
- [ ] Sandboxed processing environment
- [ ] Enhanced logging and monitoring
- [ ] Security testing automation
- [ ] Third-party security audit

## Security Contact Information

- **Security Team**: [stkzlv+ContentEngineAI@gmail.com](mailto:stkzlv+ContentEngineAI@gmail.com)
- **General Contact**: [stkzlv+ContentEngineAI@gmail.com](mailto:stkzlv+ContentEngineAI@gmail.com)
- **GitHub Security**: Use GitHub's private vulnerability reporting

## Acknowledgments

We appreciate the security research community and will acknowledge researchers who help improve ContentEngineAI's security (with their permission).

## Updates to This Policy

This security policy may be updated as the project evolves. Check the [CHANGELOG](CHANGELOG.md) for security-related updates.