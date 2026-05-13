"""Allow ``python -m moe_policylang`` to invoke the CLI."""

import sys

from moe_policylang.cli import main

sys.exit(main())
