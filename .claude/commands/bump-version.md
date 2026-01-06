Read project versioning guidelines and determine if version bump is needed.

## Process
1. Read `CONTRIBUTING.md` and `docs/versioning.md` for versioning rules
2. Review all branch changes: `git log main..HEAD --oneline`
3. Determine version bump type based on changes
4. Update version in all locations
5. Update `CHANGELOG.md`

## Version Locations
- `pyproject.toml` - main project version
- `src/scraper/amazon/__init__.py` - scraper module version (if scraper changes)

## Semantic Versioning (X.Y.Z)
- **Major (X)**: Breaking API changes
- **Minor (Y)**: New features (backward compatible)
- **Patch (Z)**: Bug fixes only

## Changelog Format (Keep a Changelog)
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features
```

## Rules
- Only bump ONE version number per release (highest priority wins)
- Never modify released version contents
- Document breaking changes clearly
- Use conventional commit messages to determine bump type:
  - `feat:` → Minor bump
  - `fix:` → Patch bump
  - `BREAKING CHANGE:` or `!` → Major bump
