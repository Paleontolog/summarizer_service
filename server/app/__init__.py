import os

from app.config import config, init_config
from app.routes import route
from flask import Flask


def create_flask_app():
    app = Flask(__name__)

    route(app)

    path = os.environ.get('CONFIG_PATH') if os.environ.get('CONFIG_PATH') else "./settings.ini"
    init_config(path)

    try:
        base_config = dict(
            SECRET_KEY=str(config['FLASK_APP']['FLASK_APP_SECRET_KEY'])
        )

        app.config.update(base_config)

    except KeyError:
        print(f"Incorrect config path: {path}")

    return app
