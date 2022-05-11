import os

from flask import Blueprint, render_template, jsonify, request

from ..client.client_type import ClientType
from ..client.summ_client import Client
from ..request import SummarizationRequest, GenerationRequest

main_controller = Blueprint('main-controller', __name__)

classifier = Client.create_from_url(client_type=ClientType[os.environ["BINARY_CLASSIFIER_TYPE"]],
                                    base_url=os.environ["BINARY_CLASSIFIER"])

summarizer = Client.create_from_url(client_type=ClientType[os.environ["SUMMARIZER_TYPE"]],
                                    base_url=os.environ["SUMMARIZER"])


@main_controller.route("/", methods=["GET"])
def get_page():
    return render_template('index.html')


@main_controller.route("/api/processing/all", methods=["POST"])
def process_all():
    data = request.get_data()

    generation_request = GenerationRequest.from_json(data)
    summarization_request = SummarizationRequest(data)

    summarization_request.input_text = classifier.process(generation_request)

    response = summarizer.process(summarization_request)

    return jsonify({"result": response.result}), 200


@main_controller.route("/api/processing/bert", methods=["POST"])
def process_bert():
    data = request.get_data()
    generation = GenerationRequest.from_json(data)
    response = classifier.process(generation)
    return jsonify({"result": response.result}), 200


@main_controller.route("/api/processing/bart", methods=["POST"])
def process_bart():
    data = request.get_data()
    summarization_request = SummarizationRequest.from_json(data)
    response = summarizer.process(summarization_request)
    return jsonify({"result": response.result}), 200
