import sys
import os
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from flask import Flask

# 1. Add folders to path so Python can find your apps
sys.path.append(os.path.abspath('./GNN/smart-contract-backend'))
sys.path.append(os.path.abspath('./RNN'))

# 2. Import your existing apps
# (Assuming your files are named 'app_gnn.py' and 'app_rnn.py' and the Flask app variable inside them is named 'app')
from app_gnn import app as gnn_app
from app_rnn import app as rnn_app

# 3. Create a simple "Root" app (just to say hello)
main_app = Flask(__name__)
@main_app.route('/')
def index():
    return "Backend is running! Use /gnn or /rnn endpoints."

# 4. Merge them!
# Requests to /gnn/... go to the GNN app
# Requests to /rnn/... go to the RNN app
application = DispatcherMiddleware(main_app, {
    '/gnn': gnn_app,
    '/rnn': rnn_app
})

if __name__ == "__main__":
    run_simple('0.0.0.0', 5000, application)