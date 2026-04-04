# eval_script.py
"""
Run with: python eval_script.py
Fails (exit code 1) if faithfulness score < THRESHOLD.
Wire this into your CI as a step.
"""

import json
import sys
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from rag import query_drug, index_data

THRESHOLD = 0.7  # fail CI if score drops below this

def run_eval():
    index_data()  # ensure index is ready

    with open("golden_eval.json") as f:
        golden = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        result = query_drug(item["question"], source=item["source"], persona="clinician")
        questions.append(item["question"])
        answers.append(result.get("answer", ""))
        contexts.append([s.get("drug", "") for s in result.get("sources", [])])
        ground_truths.append(item["expected_answer"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(f"\nEvaluation results:\n{scores}")

    avg_faithfulness = scores["faithfulness"]
    if avg_faithfulness < THRESHOLD:
        print(f"\nFAILED: faithfulness {avg_faithfulness:.2f} < threshold {THRESHOLD}")
        sys.exit(1)

    print(f"\nPASSED: faithfulness {avg_faithfulness:.2f}")

if __name__ == "__main__":
    run_eval()