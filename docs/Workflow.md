# Contribution Workflow (Solo/OSS)

## Branching
- Create a feature branch from `main` for each change.
- Push the feature branch to the remote to keep work backed up and enable PRs.

## Pull Requests
- Open a PR from the feature branch into `main`.
- Include a brief summary and testing notes.
- Run CI/manual checks before merging when possible.

## Merging
- Merge the PR (squash or regular) once satisfied.
- After merge, delete the feature branch locally and remotely:
  - `git branch -d <branch>`
  - `git push origin --delete <branch>`

## Changelog
- Keep a `Unreleased/Next` section for ongoing changes.
- On a release, move entries into a dated/versioned section.

## Releases (when ready)
- Pick a version (e.g., v0.1.0).
- Move changelog entries under that version with the release date.
- Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.

## Notes
- Use `.env` to set config (e.g., `GROQ_API_KEY`, `BIG_MOVES_DEBUG_PROMPTS`).
- Keep `main` stable; branch for new work.
