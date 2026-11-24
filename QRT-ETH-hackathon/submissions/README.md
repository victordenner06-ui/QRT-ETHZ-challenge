# QRT x ETH Hackathon: setup

1. Unzip the Kaggle zip (`qrt-x-eth-hackathon-24-11-2026.zip`, in Downloads) into `data/`.
2. In the VS Code terminal, from the root of this folder:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python -m src.train      # benchmark CV + LightGBM CV + submission in submissions/
   ```
   (Mac M1/M2: if lightgbm fails → `brew install libomp`.)
3. Select the `.venv` interpreter in VS Code (Cmd+Shift+P → "Python: Select Interpreter").
4. Claude Code: install the "Claude Code" extension (Anthropic) from the Extensions tab, or `claude` in the terminal. It reads `CLAUDE.md` automatically.
5. Submit: Kaggle → "Submit Prediction" → upload the CSV from `submissions/`.
