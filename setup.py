from setuptools import setup, find_packages
setup(
    name="agentic_ai_wf",
    packages=find_packages(),
)
```

Save. Then update the **Build Command** in Render settings to:
```
pip install -r requirements.txt && pip install -e .