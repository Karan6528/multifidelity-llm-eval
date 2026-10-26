# Multi-fidelity evaluation for an in-car LLM assistant

![ci](https://github.com/Karan6528/multifidelity-llm-eval/actions/workflows/ci.yml/badge.svg)

How do you test an LLM voice assistant for a car without running a big judge model on every single test?

That is the question behind this project. When I built RAG systems and chatbots during my internship at ScikIQ, the building part was the easy bit. The hard part was knowing whether an answer was actually good. Reading every answer yourself doesn't scale. Asking a large LLM to grade every answer works much better, but it is slow and it gets expensive once you have hundreds of tests and run them after every change.

So I built a small framework that scores each answer at three levels of cost, and then checks which mix of levels catches the most failures for the least compute.

## The three levels

| Level | What it does | Cost | Good at | Blind to |
|---|---|---|---|---|
| **L1 Rules** | Checks the JSON output: right intent, right slot values, numbers repeated back, short enough for a driver, asks before sending a message | microseconds | anything you can check mechanically | whether the answer makes sense |
| **L2 Similarity** | Compares the spoken answer with a reference answer using sentence embeddings ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), PyTorch) | milliseconds | off-topic and unhelpful answers | small factual errors ("312 km" vs "213 km" look almost the same) |
| **L3 LLM judge** | A bigger model grades the answer from 1 to 5 against a written rubric, seeing the vehicle state and the expected behaviour | seconds | almost everything | it is slow, and it can be wrong too, so I check it against my own labels |

The low levels run on every test. The question is which tests deserve the expensive one.

```
186 test requests ──► assistant ──► answers
                                     │
               ┌─────────────────────┼─────────────────────┐
               ▼                     ▼                     ▼
          L1 rules             L2 similarity            L3 judge
          (all tests)          (rule passes)         (only some tests?)
               └──────────► selection strategy ◄──────────┘
                                     │
                       failures caught vs. compute cost
```

## Test suite

186 requests a driver might make, generated from templates with a fixed seed (`data/testsuite.jsonl`):

- climate, seat heating, navigation, music, calls and messages, in normal wording, paraphrased and as sloppy speech-to-text ("hey car navigate to MBW Welt")
- questions about the car (range, battery, tyre pressure, outside temperature) that must be answered with the numbers from the vehicle state
- requests a car assistant can't do ("book me a flight") and requests it must refuse ("play a movie while I'm driving")

Each test has the expected intent, the slots, a reference answer and the vehicle state.

## Selection strategies I compared

All strategies are simulated on the same scores (every test is scored at every level once), so the comparison is fair:

1. **Rules only**: just L1.
2. **Rules + similarity**: L1, then fail anything below a similarity threshold. No judge at all.
3. **Judge on everything**: the expensive reference, 100% cost. Plus **rules + judge on everything** (101%), which catches whatever either of them catches.
4. **Rules + judge on a random sample**: the obvious baseline for a fixed budget.
5. **Cascade**: similarity decides the clear cases, and the judge only sees the grey zone in between.
6. **Budget-aware**: given a judge budget, spend it on the answers that are least similar to the reference first.

Cost is measured relative to running the judge on every test, with L1 = 1, L2 = 5 and L3 = 100 units per test (`configs/default.yaml`). A strategy only counts if at least 80% of what it flags is a real failure. Otherwise "flag everything" would look perfect.

## Results so far (simulated assistant)

Before plugging in a real model, I tested the framework on a simulated assistant (`MockAssistant`). It answers correctly most of the time, in varied wording like a real model, but injects ten kinds of faults on purpose: wrong intent, wrong slot value, broken JSON, rambling, off-topic answers, sending a message without asking, wrong numbers about the car, agreeing to unsafe requests, and so on. Because I know exactly which answers are broken, I can measure precisely what each level catches. The simulated judge also makes mistakes on purpose: it misses some subtle faults and sometimes fails a correct answer.

46 of the 186 answers were broken. With the basic rules:

![Cost vs recall, basic rules](runs/mock-basic-rules/frontier.png)

| Strategy | Cost (% of full judge) | Failures caught |
|---|---|---|
| Rules only | 1% | 80.4% |
| Rules + similarity | 5% | 87.0% |
| Budget-aware, judge 5% of rule passes | 8.8% | 87.0% |
| Random sample, same budget | 9.1% | 82.3% |
| Cascade, grey zone 0.1–0.7 | 46.9% | 91.3% |
| Cascade, grey zone 0.1–0.9 | 57.7% | 100% |
| Judge on everything | 100% | 100% |

What I learned from this:

- **Rules are underrated.** They cost almost nothing and caught 80% of the failures, including every wrong slot value, missing slot and unconfirmed message.
- **Ranking by similarity beats random sampling at small budgets.** At about 9% of the cost it caught 87% of failures, against 82% for a random sample. Then it plateaus.
- **The plateau is wrong numbers.** "You have 213 km of range left" when the car says 312 km looks almost identical to a correct answer, so similarity puts it at the very end of the queue. In this run only the judge caught them (L1 0%, L2 17%, L3 100%).
- **So I added one more cheap rule**: for questions about the car, the number in the answer must match the vehicle state. With that rule, rules alone catch 93.5% at 1% of the cost, and rules + similarity catch all 46 failures at 4.8% of the cost ([report](runs/mock/report.md)). It is a good reminder that the cheapest improvement is often a better low-fidelity check, not a smarter way to call the expensive one.

Honest caveat: the simulated assistant only makes the mistakes I designed it to make, so the cheap levels look better here than they will on a real model. That is why the next step is a real run.

## Running it on a real model

The real setup runs fully locally with [Ollama](https://ollama.com): a small model as the assistant and a bigger one as the judge.

```bash
ollama pull llama3.2:3b     # the assistant under test
ollama pull qwen2.5:7b      # the judge

mfeval all --name ollama-llama3.2-3b --backend ollama --model llama3.2:3b \
           --embedder st --judge ollama --judge-model qwen2.5:7b
```

For real runs there is no exact ground truth. An answer counts as broken if it breaks a hard rule (a wrong slot value is a bug whatever the judge thinks) or the judge fails it. Since that leans heavily on the judge, I check the judge against my own labels:

```bash
mfeval label --name ollama-llama3.2-3b --n 50   # label 50 answers yourself: p = pass, f = fail
mfeval report --name ollama-llama3.2-3b         # adds agreement (Cohen's kappa) per level
```

The report then also shows the measured time per test at each level, so you can replace my guessed cost units with real ones.

## Quick start

Needs Python 3.10+.

```bash
git clone https://github.com/Karan6528/multifidelity-llm-eval.git
cd multifidelity-llm-eval
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,embeddings]"     # leave out ",embeddings" to skip PyTorch

mfeval all --name mock                 # simulated run, no downloads, takes seconds
pytest                                 # 18 tests
```

The results land in `runs/<name>/`: `report.md`, `frontier.png`, `scores.csv` (every answer with its score at every level) and `strategies.csv`.

**VS Code:** open the folder and install the recommended extensions when asked. Select the `.venv` interpreter, and the Run and Debug panel has ready-made configurations for the simulated run, the real run and labelling. Tests show up in the Testing panel.

## Project structure

```
src/mfeval/
  testsuite.py          the 186 test requests
  assistants.py         MockAssistant (injected faults) and OllamaAssistant
  evaluators/
    rules.py            L1
    semantic.py         L2 (sentence-transformers, or TF-IDF for CI)
    judge.py            L3 (Ollama judge with rubric, or a noisy mock judge)
  selection.py          the six strategies and the Pareto front
  metrics.py            agreement and Cohen's kappa
  pipeline.py, report.py, cli.py
tests/                  pytest, run in GitHub Actions on every push
runs/                   saved runs with reports and charts
```

## Limitations and what I'd do next

- 186 tests and one simulated run is a small sample. The differences of a few points between strategies are within noise. Next I want repeated runs with different seeds and confidence intervals.
- The cost units are estimates. After the real run I'll replace them with measured times.
- Similarity to a single reference answer punishes correct answers that are worded very differently. Several references per test, or comparing against the intent instead of one sentence, would help.
- A natural next level between L1 and L3 is a small, cheap LLM judge, which would make it a true multi-fidelity ladder rather than three fixed steps.
- Learning the escalation threshold from past runs, instead of sweeping it, is the obvious step towards proper test optimisation.

## License

MIT
