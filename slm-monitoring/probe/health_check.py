
import json
import time
import requests

from prometheus_client import (
    start_http_server,
    Gauge
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"

DATASET = "probe/golden_dataset.json"

INTERVAL = 300  # 5 minutes

PASS_RATE = Gauge(
    "slm_eval_pass_rate",
    "Fraction of golden tests that passed"
)

TOTAL_TESTS = Gauge(
    "slm_eval_total_tests",
    "Total number of golden tests"
)

FAILED_TESTS = Gauge(
    "slm_eval_failed_tests",
    "Number of failed golden tests"
)

PROBE_SUCCESS = Gauge(
    "slm_eval_probe_success",
    "1 if the latest evaluation completed, otherwise 0"
)

def load_dataset():
    with open(DATASET, "r") as file:
        return json.load(file)

def ask_model(question):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": question,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"].lower()

def run_evaluation():

    dataset = load_dataset()

    passed = 0
    failed = 0

    for test in dataset:

        question = test["question"]
        expected = test["expected_contains"]

        try:
            answer = ask_model(question)

            test_passed = all(
                phrase.lower() in answer
                for phrase in expected
            )

            if test_passed:
                passed += 1
                print(test["id"], "PASS")
            else:
                failed += 1
                print(test["id"], "FAIL")
                print("Answer:", answer)

        except Exception as error:
            failed += 1
            print(test["id"], "ERROR:", error)

    total = len(dataset)

    rate = passed / total if total else 0

    PASS_RATE.set(rate)
    TOTAL_TESTS.set(total)
    FAILED_TESTS.set(failed)

    print("--------------------------------")
    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Pass rate:", round(rate * 100, 2), "%")
    print("--------------------------------")

def main():

    start_http_server(9101)

    print("SLM health probe started")
    print("Prometheus metrics: port 9101")

    while True:

        try:
            PROBE_SUCCESS.set(0)
            run_evaluation()
            PROBE_SUCCESS.set(1)

        except Exception as error:
            print("Evaluation failed:", error)

        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
