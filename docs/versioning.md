# Versioning Strategy

ContentEngineAI follows [Semantic Versioning (SemVer)](https://semver.org/) principles with a clear pre-production strategy.

## Version Format

**`MAJOR.MINOR.PATCH`**

- **MAJOR**: Breaking changes that require user intervention
- **MINOR**: New features, backward-compatible functionality
- **PATCH**: Bug fixes, performance improvements, documentation updates

### Version Sequencing

**Version numbers must be sequential** - skipping versions is not allowed, even when a release contains multiple significant features. Each version increment represents a single release, regardless of the number or magnitude of changes included.

**Examples:**
- ✅ Correct: 0.14.0 → 0.15.0 (sequential)
- ❌ Incorrect: 0.14.0 → 0.17.0 (skips 0.15.0 and 0.16.0)

This ensures version consistency, predictability, and adherence to semantic versioning principles.

## Pre-Production Phase (0.x.y)

**Current Status**: ContentEngineAI is in active development toward a stable 1.0.0 release.

### 0.x.y Strategy

Per SemVer spec, 0.x.y is the development sandbox where the API may change freely:

- **Minor (0.X.0)**: New features, enhancements, non-breaking changes
- **Patch (0.x.Y)**: Bug fixes, performance improvements, documentation
- **1.0.0**: First stable release (when stability criteria below are met)

There is no predetermined version for reaching 1.0.0 - development continues in 0.x.y until the project meets production-readiness criteria.

### Breaking Changes in 0.x

⚠️ **Important**: During the 0.x phase, breaking changes may occur in minor versions.

We will:
- Document all breaking changes in the CHANGELOG
- Provide migration guides for significant changes
- Announce breaking changes in advance when possible
- Maintain backward compatibility when feasible

## Release Process

### Release Schedule

- **Patch releases**: As needed for bug fixes (weekly/bi-weekly)
- **Minor releases**: Monthly for new features
- **Major releases**: When significant breaking changes are necessary

### Release Workflow

1. **Development**: Feature branches with development work
2. **Testing**: Automated CI/CD pipeline validates all changes
3. **Release Preparation**: Version bumped and CHANGELOG updated in feature branch
4. **Pull Request**: Feature branch merged to `main` via Pull Request (includes version changes)
5. **Tagging**: Git tag created from `main` after merge (`v0.1.0`, `v0.2.0`, etc.)
6. **GitHub Release**: Automated release notes generation
7. **Communication**: Community notification of new releases

### Dependency Updates

Automated dependency updates (Dependabot) follow the **batch into patch releases** approach:

- Dependabot PRs target `main` but are **not merged immediately**
- "Grouped security updates" enabled — all security fixes in one PR
- Open PRs serve as visible queue of pending updates
- Updates are batched and included in the next scheduled patch release
- Security-critical updates may trigger an immediate patch release
- CHANGELOG includes dependency updates under "Dependencies" or "Security" section

**Workflow**:
1. Dependabot creates grouped PR with `poetry.lock` updates targeting `main`
2. PR remains open until next patch release cycle
3. At release time, checkout the Dependabot branch and prepare it for release:
   ```bash
   # Rebase onto latest main (Dependabot branches may be stale)
   gh pr checkout <PR-number>
   git rebase main

   # Install updated deps and run full test suite
   poetry install
   poetry run pytest

   # Bump version and update CHANGELOG on the branch
   # Edit pyproject.toml: version = "X.Y.Z"
   # Edit CHANGELOG.md: add [X.Y.Z] section with "Dependencies" subsection

   # Commit, force-push (rebase changed history), merge
   git add pyproject.toml CHANGELOG.md poetry.lock
   git commit -m "Bump version to X.Y.Z"
   git push --force-with-lease
   gh pr merge <PR-number> --squash
   ```
4. Tag release from `main`:
   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

## Path to 1.0.0

### Stability Criteria

ContentEngineAI will reach 1.0.0 when:

- ✅ **Core Pipeline**: End-to-end video production pipeline stable
- ✅ **Multi-Platform Support**: Amazon + 2 additional platforms
- ✅ **API Stability**: Public APIs finalized with backward compatibility
- ✅ **Documentation**: Complete user and developer documentation
- ✅ **Test Coverage**: >95% code coverage with comprehensive integration tests
- ✅ **Performance**: Consistent sub-5-minute video generation
- ✅ **Community**: Active contributor base and issue resolution
- ✅ **Production Use**: Successfully deployed in production environments

### Post-1.0.0 Promise

After 1.0.0 release:
- **Semantic Versioning**: Strict SemVer compliance
- **Backward Compatibility**: Breaking changes only in major versions
- **LTS Support**: Long-term support for major versions
- **Migration Guides**: Comprehensive upgrade documentation

## Version Support

### Current Support Policy

- **Latest Version**: Full support with new features and bug fixes
- **Previous Minor**: Bug fixes and security patches for 3 months
- **Pre-1.0.0**: Best-effort support, focus on latest version

### Post-1.0.0 Support Policy

- **Current Major**: Full feature development and bug fixes
- **Previous Major**: Security patches and critical bug fixes for 12 months
- **LTS Versions**: Extended support for enterprise users

## Release Notes

Each release includes:

- **Summary**: High-level overview of changes
- **Added**: New features and capabilities
- **Changed**: Modifications to existing functionality
- **Deprecated**: Features being phased out
- **Removed**: Discontinued features (major versions only)
- **Fixed**: Bug fixes and performance improvements
- **Security**: Security-related changes

## Contributing to Releases

### Feature Requests

- Submit feature requests via GitHub Issues
- Use the "enhancement" label
- Provide use cases and expected behavior
- Community voting helps prioritize features

### Bug Reports

- Report bugs via GitHub Issues
- Include reproduction steps and environment details
- Critical bugs may trigger patch releases
- Use appropriate severity labels

### Release Testing

- Beta versions available for testing
- Community feedback incorporated before final release
- Release candidates published for major versions

---

**Questions?** Open an issue or discussion on GitHub for version-related questions.