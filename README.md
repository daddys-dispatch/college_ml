**Step 1:** Make sure the system has everything installed.

```bash
sudo apt update && sudo apt install -y git python3 python3-pip python3-venv python-is-python3
```

**Step 2:** Set up the dev environment.

```bash
git clone https://github.com/daddys-dispatch/college_ml.git --depth=1 && cd college_ml && python3 -m venv .venv && source .venv/bin/activate && pip install -U -r requirements.txt
```

This will set up the development environment and install all the required packages.

> **Important:** The repository contains a folder named `python`. Use the files inside this folder **only for viewing/copying their contents from the terminal** (for example, using `cat filename.py`). Do not modify these files unless explicitly instructed. Copy the required code from the terminal and use it as needed.

This will print the file contents to the terminal so you can copy them.
