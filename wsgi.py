import sys

# This assumes your Flask app file is named 'server.py'
path = '/home/krishna342/mysite'
if path not in sys.path:
    sys.path.append(path)

from server import app as application
