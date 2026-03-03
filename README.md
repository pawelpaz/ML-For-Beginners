# SKM's Machine Learning Workshops 🤖

Welcome to the Machine Learning for Beginners course! This curriculum is based on the [Microsoft ML for Beginners](https://github.com/microsoft/ML-For-Beginners) program, but it has been modified and condensed to fit our 8-week workshop format.

## 🗓️ Course Structure

- **9 Meetings:** Lesson 0 (Setup) + 8 Content Sessions
- **Format:** Weekly live sessions on Discord every Tuesday, followed by team-based project work throughout the week
- **Language:** All materials, coding, and documentation are in **English**
- **Final Project:** The last session is dedicated to presenting your team's custom ML project

## Prerequisites & Setup

### 1. Python 3.10+

Python is our core programming language.

- Download the installer from [python.org](https://www.python.org/downloads/)
- **⚠️ IMPORTANT (Windows Users):** During installation, you **must** check the box: **"Add Python to PATH"**. If you skip this, your terminal will not recognize Python commands.

### 2. Git

Git is required to download assignments and push your code to GitHub.

- Download and install from [git-scm.com](https://git-scm.com/downloads)
- Use the default settings during the installation process

### 3. Visual Studio Code & Extensions

VS Code is our primary editor.

- Download and install [VS Code](https://code.visualstudio.com/)
- Open VS Code, go to the **Extensions** view (Ctrl+Shift+X), and install:
    - **Python** (by Microsoft)
    - **Jupyter** (by Microsoft)

### 4. Virtual Environment (venv)

We use virtual environments to keep project-specific libraries isolated and prevent conflicts.

1. Open your project folder in VS Code
2. Open the terminal (`Ctrl` + `` ` ``) and run:
     ```bash
     python -m venv .venv
     ```
3. **Activate it:**
     - **Windows:** `.venv\Scripts\activate`
     - **Mac/Linux:** `source .venv/bin/activate`
     - *Note: You should see `(venv)` appearing at the beginning of your terminal prompt*

### 5. Install ML Libraries

Once your virtual environment is active, install all required dependencies:

```bash
pip install -r requirements.txt
```

## Lesson Components

Each lesson includes:

- Optional sketchnote
- Optional supplemental video
- Video walkthrough (some lessons only)
- [Pre-lecture warmup quiz](https://ff-quizzes.netlify.app/en/ml/)
- Written lesson
- Step-by-step project guides (project-based lessons)
- Knowledge checks
- Challenge
- Supplemental reading
- Assignment
- [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

| Lesson | Topic | Link |
| :---: | :--- | :--- |
| **0** | Introduction | [README.md](./0-Introduction/README.md) |
| **1** | Data | [README.md](./1-Data/README.md) |
| **2** | Regression-1 | [README.md](./2-Regression-1/README.md) |
| **3** | Regression-2 | [README.md](./3-Regression-2/README.md) |
| **4** | Classification-1 | [README.md](./4-Classification-1/README.md) |
| **5** | Classification-2 | [README.md](./5-Classification-2/README.md) |
| **6** | Clustering | [README.md](./6-Clustering/README.md) |
| **7** | Praca nad projektem | [README.md](./7-Project-1/README.md) |
| **8** | Prezentacja projektów | [README.md](./8-Project-2/README.md) |
