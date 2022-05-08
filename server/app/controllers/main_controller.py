from flask import Blueprint, jsonify, request

from ..server_model import ServerModel

server_controller = Blueprint('server-controller', __name__)

model = ServerModel()


@server_controller.route("/api/generate", methods=["POST"])
def process_all():
    data = request.get_data()

    response = model.process(data)

    return jsonify({"result": response}), 200
