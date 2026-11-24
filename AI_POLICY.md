# AI policy

AI tools are welcome in jaxspec. They do not change what we expect of a contribution: it
has to be worth the time it takes to review, and reliable enough for the published results
that rely on jaxspec.

These rules apply to pull requests, issues and comments, from everyone, maintainers
included. Automation set up by the maintainers, such as Dependabot, is exempt.

## The rules

1. **Agree on the change first.** Open an issue or a discussion and wait for a maintainer
   to say yes before writing a pull request. Only a small fix can skip this step: a typo,
   a broken link, a bug fixed in a few lines.
2. **Say what you used.** Every pull request fills in the AI line of its template: `none`,
   or the tool, the model and what they did.
3. **Answer for your change.** You have read every line and run it, and you can explain it
   without the tool. Its licence is your responsibility too: a tool can reproduce code
   that cannot be distributed under the MIT licence.
4. **Talk to us yourself.** Answer reviews in your own words, and say so when a description
   or a comment was written by a tool. Using one to translate or to correct your language
   is fine.
5. **No autonomous agents.** A person decides to open each issue and pull request.

In commits, only people are authors or co-authors. Credit a tool with a trailer:

```text
Assisted-by: <tool>:<model>
```

## When a rule is not followed

The pull request or issue is closed with a link to this page, without a detailed review.
This says nothing about you or your idea: open an issue and we can start from there.

## Sources

This policy follows the *Moderate* sample of the
[Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/ai/),
with ideas from the policies of
[Astropy](https://github.com/astropy/astropy-project/blob/main/policies/ai-policy.md),
[SunPy](https://docs.sunpy.org/en/latest/dev_guide/contents/ai_usage.html) and
[X-PSI](https://xpsi-group.github.io/xpsi/ai_policy.html).
