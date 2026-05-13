"""Allow ``python -m moe_sched`` to invoke the CLI."""

import sys

from moe_sched.cli import main

sys.exit(main())
