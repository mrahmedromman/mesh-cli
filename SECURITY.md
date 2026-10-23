# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead: open the
repository's **Security** tab and choose **Report a vulnerability**.

Include the `mesh --version`, what you did, and what you expected. You can expect a reply within a week.

## What counts

`mesh` runs the `command` and `stop` lines of a `models.yaml` through `bash` with your user's permissions, by design.
A config file is a script, so running a config you do not trust is not a vulnerability in `mesh`.

In scope: a way to make `mesh` run something that is not in the config, write outside the log directory, or signal
processes that do not belong to the stack.

## Supported versions

Only the latest release receives fixes while the project is in alpha.
