"""Allow `python -m mesh_cli` as an alternative to the `mesh` command."""

import sys

from .cli import main

sys.exit(main())
