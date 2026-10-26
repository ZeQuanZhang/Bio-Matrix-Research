"""OpenSSH askpass: the saved password is emitted only to ssh's private pipe."""
import json
import os
from pathlib import Path
import sys


def main():
    try:
        path = Path(os.environ['BIOMATRIX_SSH_CONFIG'])
        password = json.loads(path.read_text())['password']
        if not isinstance(password, str) or not password or '\n' in password or '\r' in password:
            raise ValueError('Unsupported password')
    except (OSError, KeyError, ValueError, TypeError):
        print('Bio-Matrix SSH: local authentication configuration is unavailable.', file=sys.stderr)
        return 1
    sys.stdout.write(password+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
