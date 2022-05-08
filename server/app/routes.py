from app.controllers.main_controller import server_controller


def route(app):
    app.register_blueprint(server_controller)
