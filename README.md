# Testing an in-car LLM assistant without paying for a judge on every test

![ci](https://github.com/Karan6528/multifidelity-llm-eval/actions/workflows/ci.yml/badge.svg)

During my internship at ScikIQ I built RAG systems and chatbots. Getting them to answer was not the hard part. The hard part was knowing if an answer was actually good. I ended up reading a lot of answers by hand, and that obviously doesn't scale.

The usual fix is to let a bigger LLM grade the answers ("LLM-as-judge"). That works well, but it is slow. If you have a few hundred tests and you want to run them after every change, it adds up quickly.

So for this project I tried something in between. Every answer can be checked at three levels, from cheap to expensive, and I wanted to find out how much of the expensive checking you can skip without missing failures.

I picked an in-car voice assistant as the use case ("set the temperature to 21", "how far can I still drive?"), because there the mistakes really matter: a wrong number about your battery or a "sure!" to an unsafe request is not just a bad answer.

## The three levels

**L1, rules.** Plain Python, no model. Did the assistant return valid JSON, the right intent, the right values (21 degrees, not 23)? Is the answer short enough to listen to while driving? Does it ask before sending a message? For questions about the car, does the number match the vehicle data? This takes microseconds.

**L2, similarity.** I embed the answer and a reference answer with sentence-transformers ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), PyTorch) and compare them. This catches answers that pass the rules but talk about something else, or just say "I'm not sure what you mean". Takes a few milliseconds.

**L3, LLM judge.** A bigger model gets the request, the vehicle data, the expected behaviour and the answer, and gives a score from 1 to 5 using a rubric (`src/mfeval/evaluators/judge.py`). Best quality, but seconds per answer. It can also be wrong, so there is a small tool to label answers myself and check how much I agree with it.

Rules always run because they cost nothing. The real question is which answers should go to the judge.

## The tests

186 requests, generated from templates with a fixed seed so everyone gets the same file (`data/testsuite.jsonl`). Climate, seat heating, navigation, music, calls, messages and questions about the car. Each one comes in normal wording, reworded, and as messy speech-to-text like "hey car navigate to MBW Welt". There are also things the assistant can't do ("book me a flight") and things it should refuse ("play a movie while I'm driving").

## What I compared

I score every answer at all three levels once, and then replay different strategies on those scores. That way every strategy is judged on exactly the same answers.

- rules only
- rules + a similarity threshold (no judge)
- judge on everything (the expensive baseline)
- rules, then the judge on a random sample
- a cascade: similarity decides the obvious cases, the judge only looks at the unclear middle
- budget-aware: you have a fixed judge budget and spend it on the least similar answers first

Cost is relative to running the judge on every test. I used 1 / 5 / 100 units per test for L1 / L2 / L3 (`configs/default.yaml`). These are guesses for now, the real run measures the actual times. I also ignore any strategy where less than 80% of what it flags is really broken, otherwise "flag everything" would win.

## First results (simulated assistant)

I haven't run a real model through this yet. To check the framework itself works, I wrote a fake assistant (`MockAssistant`) that answers correctly most of the time, with different wording each time, but makes ten kinds of mistakes on purpose: wrong intent, wrong values, broken JSON, rambling, going off topic, sending a message without asking, wrong numbers about the car, agreeing to unsafe requests, and a few more. Because I know which answers are broken, I can see exactly what each level catches. The fake judge also makes mistakes now and then, so it isn't perfect either.

46 of the 186 answers were broken. First with only the basic rules:

![Cost vs recall, basic rules](runs/mock-basic-rules/frontier.png)

| Strategy | Cost (% of judging everything) | Failures caught |
|---|---|---|
| Rules only | 1% | 80.4% |
| Rules + similarity | 5% | 87.0% |
| Judge the least similar 5% | 8.8% | 87.0% |
| Judge a random sample (same budget) | 9.1% | 82.3% |
| Cascade, unclear zone 0.1–0.7 | 46.9% | 91.3% |
| Cascade, unclear zone 0.1–0.9 | 57.7% | 100% |
| Judge everything | 100% | 100% |

A few things surprised me.

The rules did much better than I expected. 80% of the failures for basically no cost. Wrong values, missing values and unconfirmed messages were all caught here.

Picking the least similar answers for the judge was better than picking randomly, at least with a small budget (87% vs 82% at about 9% of the cost). But after that it got stuck.

The reason was wrong numbers. "Around 213 km of range left" when the car says 312 km reads almost the same as the correct answer, so the similarity score is high and these answers end up last in the queue. There were 6 of them in this run. The rules caught none, similarity caught 1, and the judge caught all 6.

So I added one more rule: for questions about the car, the number in the answer has to match the vehicle data. That one rule changed a lot. Rules alone went to 93.5% at 1% of the cost, and rules + similarity caught all 46 failures at 4.8% ([full report](runs/mock/report.md)). Before building smarter ways to call the expensive model, it was worth making the cheap checks better.

To be clear, the fake assistant only makes the mistakes I gave it, so the cheap checks probably look better here than they will with a real model. So I tried a real one.

## Real model: Llama 3.2 3B, judged by Qwen 2.5 7B

Same 186 tests, everything running locally on my laptop with [Ollama](https://ollama.com): `llama3.2:3b` as the assistant, `qwen2.5:7b` as the judge, all-MiniLM-L6-v2 for similarity.

**Time per answer**

| Level | Time |
|---|---|
| L1 rules | under 0.1 ms |
| L2 similarity | 12 ms |
| L3 judge | 11.4 s on average, up to 100 s |

The judge took 35 minutes for the whole run, roughly 900 times slower than similarity. My guessed cost units (1 / 5 / 100) were far too kind to it.

**First the rules embarrassed me.** On the first pass they failed 71 answers, the judge only 17. When I looked, most rule failures were my fault: the model wrote `range_km` instead of `range`, `everyone` instead of `all`, `office` instead of `the office`. Perfectly fine answers, but my rules compared exact strings, and my prompt never told the model which values were allowed. I fixed both. I also found that my confirmation rule only checked for a question mark, and the model kept writing things like "I've sent a message to Anna. Would you like me to confirm?". Question mark, message already gone. I fixed that too.

**Then I labelled answers myself.** For a real model nobody knows which answers are wrong, so I went through 63 of them myself (`mfeval label`). The tool shows me the ones where rules and judge disagree first, plus some random ones where they agree, and doesn't show me what they said. My labels are the ground truth. For the other 123 answers I trust rules and judge where they agree.

That gives 21 broken answers out of 186 (11%). Every one of the 18 "send a message" tests failed, because the model always sends first (or says "sending…") and asks afterwards. The other three: YouTube on the centre display while driving (twice), and a poem request turned into a "poetry playlist".

**How each level did**

| Level | Agrees with my labels | Failures caught (of 21) | False alarms |
|---|---|---|---|
| L1 rules | 84% (kappa 0.65) | 21 | 10 of 31 flags |
| L2 similarity | 32% (kappa −0.33) | 8 | 30 of 38 flags |
| L3 judge | 87% (kappa 0.63) | 15 | 2 of 17 flags |

- **The rules catch everything, but a third of what they flag is fine.** Four polite refusals the model labelled "unsupported" instead of "refuse_unsafe", three "petrol station" destinations without "nearest", and two speech-to-text names I was OK with.
- **The judge is precise but misses things.** All 6 misses were the same pattern: "I'm sending a message to Mom. Please confirm before sending." The judge read that as asking for confirmation. I didn't, because the driver can't tell if the message is already on its way.
- **Similarity is no help here.** It disagrees with me more often than it agrees.

**What this means for test selection.** Flagging whatever the rules fail gives too many false alarms, so on this data every strategy that does that is ruled out. What works is using the rules as triage: only answers the rules fail go to the judge, and the judge decides. That catches the same 15 of 21 failures as judging every answer, with no false alarms, at **18% of the cost**. Adding similarity-ranked answers on top didn't catch anything extra.

![Cost vs recall, real model](runs/ollama-llama3.2-3b/frontier.png)

So the simulation and the real run disagree on one big point: with the fake assistant the cheap checks were almost enough on their own, with a real model they are a good filter but not a verdict. Full numbers in the [report](runs/ollama-llama3.2-3b/report.md).

**Honest caveats.** I tightened the confirmation rule after labelling, based on how I labelled, so the rules' 21 of 21 is fitted to this data and will look worse on new tests. The 123 answers I didn't label count as correct when rules and judge agree, which could hide mistakes both of them miss. And 21 failures is a small number to compare strategies on.

## Running it with a real model

```bash
ollama pull llama3.2:3b     # assistant
ollama pull qwen2.5:7b      # judge

mfeval all --name ollama-llama3.2-3b --backend ollama --model llama3.2:3b --embedder st --judge ollama --judge-model qwen2.5:7b
mfeval label --name ollama-llama3.2-3b       # label answers yourself, disagreements first
mfeval report --name ollama-llama3.2-3b
```

Until you have labelled something, the report only shows how the levels agree with each other and makes no claims about failures caught. After changing a rule, `mfeval rescore --name ...` re-runs the cheap levels in seconds and keeps the judge's verdicts.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/Karan6528/multifidelity-llm-eval.git
cd multifidelity-llm-eval
python -m venv .venv
.venv\Scripts\activate            # on Mac/Linux: source .venv/bin/activate
pip install -e ".[dev,embeddings]"   # drop ",embeddings" if you don't want PyTorch

mfeval all --name mock    # the simulated run, takes a few seconds
pytest                    # 27 tests
```

Results go to `runs/<name>/`: a report, the chart, every answer with its scores, and the strategy comparison.

If you use VS Code, the folder has run configurations for the simulated run, the real run and labelling, and the tests show up in the Testing tab.

## Files

```
src/mfeval/
  testsuite.py        the 186 requests
  assistants.py       fake assistant + Ollama assistant
  evaluators/         rules.py (L1), semantic.py (L2), judge.py (L3)
  selection.py        the strategies
  metrics.py          agreement, kappa
  pipeline.py, report.py, cli.py
tests/                run on every push with GitHub Actions
runs/                 saved results
```

## What's missing / next

- A fresh set of tests, so I can check the rules on data they weren't tuned on.
- More labels, and repeated runs with different seeds, so the numbers come with confidence intervals instead of a single run with 21 failures.
- Use the measured times as costs instead of my guesses.
- Comparing with one reference sentence is unfair to correct answers worded very differently, which is part of why similarity did so badly. Several references per test, or comparing intents instead of sentences, would help.
- A small, cheap LLM judge between the rules and the big judge would make it a proper ladder.
- Right now I pick thresholds by hand. Learning when to escalate from earlier labelled runs is the more interesting version of test optimisation.

MIT license.
