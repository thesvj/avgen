#!/usr/bin/env bash
# Verify every non-merge commit in a range carries a Signed-off-by line matching
# its author.
#
# Lives in a script rather than inline in the workflow so it can be run locally
# before pushing — a contributor should not need a failed CI run to find out.
#
#   .github/scripts/check_dco.sh origin/main HEAD
set -euo pipefail

base="${1:?usage: check_dco.sh <base-ref> <head-ref>}"
head="${2:?usage: check_dco.sh <base-ref> <head-ref>}"

missing=0
# Merge commits are skipped: their sign-off belongs to the commits they bring
# in, and GitHub's own merge commits carry none.
while read -r sha; do
    [ -n "$sha" ] || continue
    author="$(git show -s --format='%an <%ae>' "$sha")"
    if git show -s --format='%B' "$sha" | grep -qiF "Signed-off-by: $author"; then
        continue
    fi
    subject="$(git show -s --format='%s' "$sha")"
    printf '::error::%s is not signed off by its author (%s): %s\n' \
        "$sha" "$author" "$subject"
    missing=$((missing + 1))
done < <(git rev-list --no-merges "$base..$head")

if [ "$missing" -gt 0 ]; then
    printf '\n%d commit(s) are missing a Signed-off-by line matching the author.\n\n' "$missing"
    printf 'avgen uses the Developer Certificate of Origin:\n'
    printf '  https://developercertificate.org/\n\n'
    printf 'Signing off certifies that you wrote the change, or have the right to\n'
    printf 'submit it under Apache-2.0. There is no CLA.\n\n'
    printf 'Sign a commit as you write it:\n'
    printf '  git commit -s -m "your message"\n\n'
    printf 'Sign every commit already on this branch:\n'
    printf '  git rebase --signoff origin/main\n'
    printf '  git push --force-with-lease\n\n'
    exit 1
fi

printf 'all commits in %s..%s are signed off\n' "$base" "$head"
