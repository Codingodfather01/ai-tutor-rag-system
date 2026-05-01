import json
import pickle

with open("data/data.json", "r") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

with open("data/questions.pkl", "wb") as f:
    pickle.dump(questions, f)

with open("data/answers.pkl", "wb") as f:
    pickle.dump(answers, f)

print("Training complete.")
