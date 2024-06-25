# Command Line Tool

This is a simple python CLI tool that will return some information from a public API service.


### HOW TO USE THIS TOOL
PreRequistes:
 Python 3.11
 Windows or Unix Machine


```bash
git clone <>
set API_KEY=XXXX-XXXX-XXXX-XXXX  (Use export in Linux Instead and value should used as Secret/Masked if needs to be used in Pipelines)
pip install -r requirements.txt
python app.py <Word to be Queried>
eg:
 python app.py exercise
 INFO:root:Connecting to Merriam-Webster Dictionary
 INFO:root:Connection successful to MW
 Result: the act of putting into use, action, or practice
```