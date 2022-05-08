import os

from flask import Blueprint, render_template, jsonify, request

from ..client.summ_client import Client
from ..request import SummarizationRequest, GenerationRequest

main_controller = Blueprint('main-controller', __name__)

classifier = Client(url=os.environ["BINARY_CLASSIFIER"])
summarizer = Client(url=os.environ["SUMMARIZER"])


@main_controller.route("/", methods=["GET"])
def get_page():
    return render_template('index.html')


@main_controller.route("/api/processing/all", methods=["POST"])
def process_all():
    data = request.get_data()

    generation_request = GenerationRequest.from_json(data)
    summarization_request = SummarizationRequest(data)

    summarization_request.input_text = classifier.generate(generation_request)

    response = summarizer.generate(summarization_request)

    return jsonify({"result": response.result}), 200


@main_controller.route("/api/processing/bert", methods=["POST"])
def process_bert():
    data = request.get_data()
    generation = GenerationRequest.from_json(data)
    response = classifier.generate(generation)
    return jsonify({"result": response.result}), 200


@main_controller.route("/api/processing/bart", methods=["POST"])
def process_bart():
    data = request.get_data()
    summarization_request = SummarizationRequest.from_json(data)
    response = summarizer.generate(summarization_request)
    return jsonify({"result": response.result}), 200
