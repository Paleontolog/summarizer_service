from app.controllers.main_controller import main_controller


def route(app):
    app.register_blueprint(main_controller)
