# Analysis script for Q1 results
import json

def load_results():
    with open('results_enhanced/q1_all_results.json', 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    print("Analysis script ready!")
