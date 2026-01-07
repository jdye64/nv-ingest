# Release Process Guide

This document outlines the GitHub hygiene, processes, and conventions for releasing NeMo Retriever Extraction (nv-ingest). Following this guide ensures consistency, quality, and predictability across all releases.

## Table of Contents

1. [Release Overview](#release-overview)
2. [Versioning Strategy](#versioning-strategy)
3. [Release Branch Workflow](#release-branch-workflow)
4. [Release Candidate Process](#release-candidate-process)
5. [Final Release Process](#final-release-process)
6. [Post-Release Activities](#post-release-activities)
7. [Hotfix Process](#hotfix-process)
8. [Automated Release Workflows](#automated-release-workflows)
9. [Checklist Templates](#checklist-templates)

---

## Release Overview

### Release Philosophy

NeMo Retriever Extraction follows a structured release process that:

- **Isolates release work** from ongoing development via dedicated `release/` branches
- **Uses tags for immutability** to mark specific points in time (RCs and final releases)
- **Promotes stability** through iterative release candidates before final release
- **Maintains multiple packages** (api, client, and core service) with synchronized versions

### Release Types

| Type | Description | Branch | Tag Format | Frequency |
|------|-------------|--------|------------|-----------|
| **Nightly** | Automated dev builds from `main` | `main` | None | Daily (11:30 PM UTC) |
| **Release Candidate** | Pre-release for testing | `release/X.Y.Z` | `vX.Y.Z-rcN` | As needed |
| **Final Release** | Production-ready release | `release/X.Y.Z` | `vX.Y.Z` | As scheduled |
| **Hotfix** | Critical bug fix | `release/X.Y.Z` | `vX.Y.Z+1` | Emergency only |

---

## Versioning Strategy

### Version Format

NeMo Retriever Extraction uses **calendar versioning** with semantic extensions:

```
YY.M.PATCH[-rcN]
```

- **YY**: Two-digit year (e.g., `25` for 2025)
- **M**: Month without leading zero (e.g., `1` for January, `12` for December)
- **PATCH**: Incremental patch number starting at 0
- **-rcN**: Optional release candidate suffix (e.g., `-rc1`, `-rc2`)

**Examples:**
- `25.1.0` - January 2025 release
- `25.4.1` - April 2025, first patch
- `26.1.0-rc3` - January 2026, third release candidate

### Version File Management

The project maintains a single source of truth for versions:

- **`version.txt`**: Root-level version file (e.g., `26.1.0`)
- **Package versions**: Dynamically generated via `get_version()` in:
  - `src/nv_ingest/version.py`
  - `api/src/version.py`
  - `client/src/version.py`

Version generation is controlled by environment variables:

```bash
# Development version (default)
NV_INGEST_RELEASE_TYPE=dev NV_INGEST_VERSION=26.1.0
# Output: 26.1.0.dev20260107

# Release version
NV_INGEST_RELEASE_TYPE=release NV_INGEST_VERSION=26.1.0
# Output: 26.1.0
```

---

## Release Branch Workflow

### 1. Creating a Release Branch

Release branches are created from `main` when preparing for a new release.

**Branch Naming Convention:**
```
release/X.Y.Z
```

**Steps:**

```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create release branch
git checkout -b release/26.1.0

# 3. Update version.txt
echo "26.1.0" > version.txt

# 4. Update CHANGELOG.md
# Add release date and finalize release notes
# Format: # NVIDIA Ingest YY.M.PATCH

# 5. Commit version changes
git add version.txt CHANGELOG.md
git commit -m "Prepare release 26.1.0"

# 6. Push release branch
git push origin release/26.1.0
```

### 2. Release Branch Protection

Once created, release branches should be protected:

- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Include administrators in restrictions
- ✅ Restrict force pushes
- ❌ Do not allow branch deletion

### 3. Working on Release Branches

**Allowed Changes:**
- Bug fixes
- Documentation updates
- Version/changelog updates
- Build/packaging fixes

**Prohibited Changes:**
- New features
- Refactoring
- Non-critical improvements

**Process:**
1. Create feature branch from release branch: `git checkout -b fix/issue-123 release/26.1.0`
2. Make changes and commit
3. Open PR targeting the release branch (not `main`)
4. After merge to release branch, cherry-pick to `main` if needed

---

## Release Candidate Process

### Purpose

Release candidates (RCs) allow for:
- Integration testing
- Performance validation
- User acceptance testing
- Bug identification before final release

### Creating a Release Candidate

**Tag Naming Convention:**
```
vX.Y.Z-rcN
```

Where `N` starts at 1 and increments for each subsequent RC.

**Steps:**

```bash
# 1. Ensure release branch is ready
git checkout release/26.1.0
git pull origin release/26.1.0

# 2. Run pre-release checks (see checklist below)
# - All tests pass
# - Documentation is up to date
# - CHANGELOG.md is complete for this RC

# 3. Create and push the RC tag
git tag -a v26.1.0-rc1 -m "Release candidate 1 for version 26.1.0"
git push origin v26.1.0-rc1
```

### Testing Release Candidates

Each RC should undergo:

1. **Automated Testing**
   - All unit tests pass
   - Integration tests pass
   - Docker image builds successfully

2. **Manual Testing**
   - Deploy to staging environment
   - Run smoke tests
   - Validate all supported file types
   - Performance benchmarking

3. **Documentation Review**
   - Installation instructions work
   - Examples run without errors
   - API documentation is accurate

### Iterating on Release Candidates

If bugs are found:

```bash
# 1. Fix bugs on release branch via PRs
git checkout release/26.1.0
# ... make fixes, merge PRs ...

# 2. Create next RC
git tag -a v26.1.0-rc2 -m "Release candidate 2 for version 26.1.0"
git push origin v26.1.0-rc2

# 3. Repeat testing cycle
```

**Best Practices:**
- Document known issues in each RC's GitHub release notes
- Communicate RC availability to testers
- Set clear expectations: RCs are not production-ready
- Allow at least 2-3 days between RCs for testing

---

## Final Release Process

### Pre-Release Requirements

Before creating the final release, ensure:

- ✅ Latest RC has been thoroughly tested
- ✅ All critical bugs are resolved
- ✅ Documentation is complete and accurate
- ✅ CHANGELOG.md is finalized with release date
- ✅ All CI/CD workflows are passing
- ✅ Performance benchmarks meet expectations
- ✅ Security scanning shows no critical issues

### Creating the Final Release

**Tag Naming Convention:**
```
vX.Y.Z
```

**Steps:**

```bash
# 1. Final sanity checks on release branch
git checkout release/26.1.0
git pull origin release/26.1.0

# 2. Update CHANGELOG.md with release date
# Change from "Unreleased" or date RC to actual release date
# Example: # NVIDIA Ingest 26.1.0 (January 15, 2026)

# 3. Commit and push final changes
git add CHANGELOG.md
git commit -m "Finalize 26.1.0 release"
git push origin release/26.1.0

# 4. Create and push release tag
git tag -a v26.1.0 -m "Release version 26.1.0"
git push origin v26.1.0
```

### Automated Release Triggers

Pushing the tag triggers automated workflows:

1. **Docker Images** (`.github/workflows/docker-release-publish.yml`)
   - Builds multi-platform images (amd64, arm64)
   - Publishes to NGC registry with tag `26.1.0`

2. **PyPI Packages** (`.github/workflows/pypi-nightly-publish.yml`)
   - Builds wheels for `nv-ingest`, `nv-ingest-api`, `nv-ingest-client`
   - Publishes to Artifactory

3. **Conda Packages** (`.github/workflows/conda-publish.yml`)
   - Builds conda packages
   - Publishes to Nvidia channel

### Manual Release Workflow Dispatch

For manual control over releases:

```bash
# Navigate to GitHub Actions → "Nv-Ingest Nightly PyPi Wheel Publish"
# Click "Run workflow"
# Set parameters:
#   - source branch: release/26.1.0
#   - VERSION: 26.1.0
#   - RELEASE_TYPE: release
```

### Creating GitHub Release

After tag is pushed and automated workflows complete:

1. Go to **Releases** → **Draft a new release**
2. **Tag:** `v26.1.0`
3. **Release title:** `NeMo Retriever Extraction 26.1.0`
4. **Description:** Copy from CHANGELOG.md and enhance with:
   - Overview/highlights
   - Breaking changes (if any)
   - Installation instructions
   - Links to documentation
   - Known issues
   - Contributors

5. **Artifacts:** Attach additional files if needed:
   - Standalone packages
   - Checksums
   - Signatures

6. Click **Publish release**

---

## Post-Release Activities

### 1. Merge Back to Main

Critical: Ensure release changes are integrated back to `main`:

```bash
# 1. Create PR from release branch to main
git checkout main
git pull origin main
git checkout -b merge/release-26.1.0-to-main
git merge origin/release/26.1.0

# 2. Resolve any conflicts (especially version.txt and CHANGELOG.md)
# - version.txt on main should be the NEXT planned version
# - CHANGELOG.md should have release notes at the top

# 3. Push and create PR
git push origin merge/release-26.1.0-to-main
# Create PR: merge/release-26.1.0-to-main → main
```

### 2. Update Documentation

- ✅ Update docs.nvidia.com with new version
- ✅ Update README.md if installation instructions changed
- ✅ Update example notebooks if API changed
- ✅ Verify all links in documentation

### 3. Communications

Announce the release through:
- GitHub Discussions
- Internal team channels
- NVIDIA AI forums
- Release blog post (for major releases)

Include:
- What's new
- Upgrade instructions
- Deprecation notices
- Known issues and workarounds

### 4. Prepare Next Version

```bash
# Update version.txt on main to next planned version
git checkout main
git pull origin main
echo "26.2.0" > version.txt
git add version.txt
git commit -m "Bump version to 26.2.0-dev"
git push origin main

# Create CHANGELOG.md section for next version
# Add new section at top of CHANGELOG.md:
# # NVIDIA Ingest 26.2.0
# 
# ## New Features
# - ...
```

### 5. Monitor Release

After release, monitor for:
- Docker image pull issues
- PyPI/Conda installation failures
- Bug reports
- Performance regressions
- Documentation errors

---

## Hotfix Process

### When to Use Hotfixes

Hotfixes are for **critical issues only**:
- Security vulnerabilities
- Data loss bugs
- Severe performance regressions
- Blocking production issues

### Hotfix Workflow

```bash
# 1. Work directly on existing release branch
git checkout release/26.1.0
git pull origin release/26.1.0

# 2. Create hotfix branch
git checkout -b hotfix/critical-security-fix

# 3. Make minimal changes to fix the issue
# - Keep changes focused
# - Add tests that verify the fix

# 4. Create PR targeting release branch
# PR: hotfix/critical-security-fix → release/26.1.0

# 5. After merge, update version
echo "26.1.1" > version.txt
git add version.txt
git commit -m "Bump version to 26.1.1 for hotfix"
git push origin release/26.1.0

# 6. Tag hotfix release
git tag -a v26.1.1 -m "Hotfix release 26.1.1: Fix critical security issue"
git push origin v26.1.1

# 7. Cherry-pick to main
git checkout main
git cherry-pick <hotfix-commit-sha>
git push origin main
```

### Hotfix Communication

Hotfixes require **immediate communication**:
- Mark GitHub release with "Security" or "Critical" label
- Send notifications to all users
- Update security advisories if applicable
- Document upgrade urgency

---

## Automated Release Workflows

### Nightly Builds

**Workflow:** `.github/workflows/docker-nightly-publish.yml`  
**Trigger:** Daily at 11:30 PM UTC, pushes to `main`  
**Output:** Docker image tagged with date (e.g., `2026.01.07`)

### Release Docker Images

**Workflow:** `.github/workflows/docker-release-publish.yml`  
**Trigger:** Creation of `release/*` branch  
**Output:** Docker image tagged with version from branch name

### PyPI Package Publishing

**Workflow:** `.github/workflows/pypi-nightly-publish.yml`  
**Trigger:** Manual dispatch  
**Inputs:**
- `environment`: Source branch (default: `main`)
- `VERSION`: Version string (e.g., `26.1.0`)
- `RELEASE_TYPE`: `dev` or `release`

**Output:** Wheels published to Artifactory:
- `nv-ingest-26.1.0-py3-none-any.whl`
- `nv-ingest-api-26.1.0-py3-none-any.whl`
- `nv-ingest-client-26.1.0-py3-none-any.whl`

### Conda Package Publishing

**Workflow:** `.github/workflows/conda-publish.yml`  
**Trigger:** Manual dispatch  
**Inputs:**
- `CONDA_CHANNEL`: `dev` or `main`
- `VERSION`: Version string

**Output:** Conda packages published to RapidsAI channel

---

## Checklist Templates

### Pre-Release Candidate Checklist

**Branch Preparation:**
- [ ] Release branch created from latest `main`
- [ ] `version.txt` updated to target version
- [ ] `CHANGELOG.md` updated with features/fixes/improvements
- [ ] All tests passing on release branch
- [ ] Documentation reviewed and updated

**Code Quality:**
- [ ] No merge conflicts with `main`
- [ ] Linting passes
- [ ] Code coverage meets standards
- [ ] Security scanning completed

**Dependencies:**
- [ ] All dependencies pinned to stable versions
- [ ] No critical CVEs in dependency tree
- [ ] Lock files updated (`uv.lock`)

**Testing:**
- [ ] Unit tests: 100% passing
- [ ] Integration tests: 100% passing
- [ ] Service tests: 100% passing
- [ ] Example notebooks run successfully

---

### Pre-Final Release Checklist

**RC Validation:**
- [ ] Latest RC tested for at least 48 hours
- [ ] All critical bugs resolved
- [ ] No known regressions from previous version
- [ ] Performance benchmarks meet SLAs

**Documentation:**
- [ ] README.md installation instructions verified
- [ ] API documentation complete and accurate
- [ ] CHANGELOG.md finalized with release date
- [ ] Example notebooks tested with RC

**Release Artifacts:**
- [ ] Docker images built and tested
- [ ] PyPI wheels built and validated
- [ ] Conda packages built and validated
- [ ] All packages reference same version number

**Compliance:**
- [ ] License files up to date
- [ ] NOTICE/CITATION files updated
- [ ] Third-party attribution complete
- [ ] Export compliance reviewed

**Communication:**
- [ ] Release notes drafted
- [ ] Breaking changes documented
- [ ] Upgrade guide prepared (if needed)
- [ ] Known issues documented

---

### Post-Release Checklist

**Integration:**
- [ ] Release tag created and pushed
- [ ] GitHub release published
- [ ] Release branch merged back to `main`
- [ ] `main` version bumped to next version
- [ ] CHANGELOG.md on `main` prepared for next version

**Validation:**
- [ ] Docker images available in NGC
- [ ] PyPI packages installable
- [ ] Conda packages installable
- [ ] Documentation site updated

**Monitoring:**
- [ ] Release announcement posted
- [ ] GitHub Discussions monitoring enabled
- [ ] Bug reports triaged
- [ ] Performance metrics baseline captured

**Housekeeping:**
- [ ] Old release branches archived (keep latest 3 releases)
- [ ] Deprecated features flagged for removal
- [ ] Technical debt items documented
- [ ] Lessons learned documented

---

## Best Practices

### Do's ✅

- **Create release branches early** to stabilize code while development continues
- **Use descriptive tag messages** that explain the release or RC purpose
- **Test RCs in realistic environments** that mirror production
- **Communicate proactively** about release timelines and known issues
- **Automate what you can** to reduce human error
- **Document everything** so the process is repeatable

### Don'ts ❌

- **Don't force push** to release branches or tags
- **Don't add new features** to release branches
- **Don't skip RCs** even if you think the code is ready
- **Don't rush releases** due to external pressure
- **Don't forget** to merge release changes back to `main`
- **Don't reuse tag names** even if you delete a tag

---

## Troubleshooting

### Q: A release workflow failed. What do I do?

**A:** 
1. Check the GitHub Actions logs for specific errors
2. Common issues:
   - Authentication failures: Verify secrets are set
   - Build failures: Ensure dependencies are available
   - Network timeouts: Retry the workflow
3. If workflow must be rerun after fixes, use "Re-run failed jobs"

### Q: I need to change a tag. How?

**A:** 
```bash
# Don't do this on published releases!
# If absolutely necessary (e.g., typo in RC):

# Delete local tag
git tag -d v26.1.0-rc1

# Delete remote tag
git push origin :refs/tags/v26.1.0-rc1

# Create corrected tag
git tag -a v26.1.0-rc1 -m "Corrected tag message"
git push origin v26.1.0-rc1
```

**Important:** Never modify tags for final releases that users may have already deployed.

### Q: How do I handle a broken release?

**A:** 
1. **Assess impact:** How many users affected?
2. **Immediate action:** 
   - Mark GitHub release as "pre-release" or delete if not widely used
   - Post incident notice
3. **Fix options:**
   - **Hotfix:** For critical issues, create hotfix version (e.g., 26.1.1)
   - **Yank:** For package managers, yank the broken version
4. **Communication:** 
   - Clearly explain the issue and impact
   - Provide workarounds if available
   - Set expectations for fix timeline

### Q: Multiple packages have different version numbers. Is that okay?

**A:** 
No. All three packages (`nv-ingest`, `nv-ingest-api`, `nv-ingest-client`) should always share the same version number for a given release. If they differ:
- Verify the `NV_INGEST_VERSION` environment variable was set correctly during build
- Rebuild packages with correct version
- Do not publish mismatched versions

---

## Appendix: Version History Examples

### Successful Release Timeline

```
Jan 2, 2026:  Create release/26.1.0 branch
Jan 3, 2026:  Tag v26.1.0-rc1
Jan 6, 2026:  Bug found, fix merged
Jan 7, 2026:  Tag v26.1.0-rc2
Jan 10, 2026: RC2 validated, tag v26.1.0
Jan 10, 2026: Publish GitHub release
Jan 11, 2026: Merge release/26.1.0 → main
Jan 12, 2026: Monitor release, respond to issues
```

### Release with Hotfix

```
Jan 10, 2026: Release v26.1.0
Jan 15, 2026: Critical bug discovered
Jan 15, 2026: Hotfix merged to release/26.1.0
Jan 15, 2026: Tag v26.1.1
Jan 15, 2026: Publish GitHub release with security warning
Jan 15, 2026: Cherry-pick hotfix to main
```

---

## Questions or Suggestions?

If you have questions about the release process or suggestions for improvements, please:
- Open an issue with the `process` label
- Discuss in GitHub Discussions under "Releases"
- Contact the release manager

**Document Version:** 1.0  
**Last Updated:** January 7, 2026  
**Maintained By:** NV-Ingest Engineering Team
