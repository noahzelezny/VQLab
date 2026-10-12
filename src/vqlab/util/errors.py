"""UserError: a request that cannot be done as asked.

Raised by anything a person's request reaches (a GUI POST handler, the
recipe compiler) when the request itself is the problem; its message is
shown as is. `vqlab gui` answers it with 400 {"ok": false, "error"}. Any
other exception is a server fault (500), so a bug is never dressed up as a
refusal. Lives at the bottom layer so every stage can raise it.
"""


class UserError(ValueError):
    """A request that cannot be done as asked; the message is shown as is."""
