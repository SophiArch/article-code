# SophiArch article code

The experiments behind SophiArch's published articles, one folder per article. Each script
writes the results file next to it, so you can rerun it, change a model or a setting, and
compare your numbers with ours.

| Folder | Article | Run |
|---|---|---|
| [agent-tool-installed-on-purpose](agent-tool-installed-on-purpose/) | Your Agent's Biggest Security Hole Is a Tool You Installed on Purpose | `uv run my_agent.py` |
| [churn-model-intervention-gap](churn-model-intervention-gap/) | Your Churn Model Works. Your Churn Intervention Doesn't. | see below |
| [first-model-should-be-embarrassing](first-model-should-be-embarrassing/) | Your First Model Should Be Embarrassing | see below |
| [vector-database-not-covered](vector-database-not-covered/) | Your Vector Database Thinks "Not Covered" Means "Covered" | `uv run negation.py` |

All scripts run with [uv](https://docs.astral.sh/uv/).

## agent-tool-installed-on-purpose

A small harness that replaces an agent's model with scripted attacks and checks what your tools let through. It uses no LLM, no API key and no network. Standard library only. The [folder's README](agent-tool-installed-on-purpose/) has a diagram showing where your tools plug in.

```bash
cd agent-tool-installed-on-purpose
uv run my_agent.py
```

## churn-model-intervention-gap

Risk targeting vs. uplift targeting on Kevin Hillstrom's randomised MineThatData e-mail
campaign. Download the data first (it isn't redistributed here):

```bash
cd churn-model-intervention-gap
curl -o hillstrom.csv http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
uv run --with pandas --with scikit-learn python risk_vs_uplift.py hillstrom.csv 0 1 2 3 4 5 6 7 8 9 > results.jsonl
```

Each line of `results.jsonl` is one random 50/50 split (seed 0–9).

## first-model-should-be-embarrassing

A baseline ladder (majority class → logistic regression → default gradient boosting → tuned
gradient boosting) on six PMLB datasets. Results for seeds 0–3 are in `results.jsonl`:

```bash
cd first-model-should-be-embarrassing
uv run --with pmlb --with scikit-learn --with scipy python ladder.py adult,churn,spambase,magic,hypothyroid,phoneme 0
```

## vector-database-not-covered

How six open embedding models and two cross-encoder re-rankers handle negation, on 40 invented
policy facts (`items.py`). Downloads about 3 GB of models on first run.

```bash
cd vector-database-not-covered
uv run negation.py
```

Learn more at [SophiArch](https://lms.sophiarch.com/?utm_source=github&utm_medium=referral&utm_campaign=article-code).

## Licence

MIT. See [LICENSE](LICENSE).
