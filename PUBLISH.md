# Publish

This repo is the origin for the Hermes SoL-Pi plugin. Do **not** open a PR on `NousResearch/hermes-agent`.

If this tree was produced under `/tmp/sol-pi-hermes` without GitHub write as `kvnloo`:

```bash
gh auth login   # as kvnloo
git -C /tmp/sol-pi-hermes remote add origin git@github.com:kvnloo/sol-pi-hermes.git
# local insteadOf must not rewrite github.com to the cloud agent token
git -C /tmp/sol-pi-hermes config --local --unset-all 'url.https://x-access-token:.insteadof' 2>/dev/null || true
gh repo create kvnloo/sol-pi-hermes --public --source=/tmp/sol-pi-hermes --remote origin --push
gh pr create --repo kvnloo/sol-pi-hermes --title "feat: SoL-Pi Hermes plugin (ObservationPack projection)" --body "Standalone Hermes plugin port. refs PER-1507. Does not merge to hermes-agent."
```
