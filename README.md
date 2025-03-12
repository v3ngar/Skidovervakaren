# Skidovervakaren
 
 This guide walks you through setting up the Python virtual environment and installing dependencies.

## Setting Up the Virtual Environment

**Ensure Python is Installed**  
Check if Python is installed by running:

```
python --version
```
or

```
python3 --version
```

> [!IMPORTANT]
> Python 3.6+ is required.

### Create a Virtual Environment

Run the following command in your project directory:

``` 
python -m venv venv 
```

or

```
python3 -m venv venv 
```

### Activate the Virtual Environment

Windows (Command Prompt / PowerShell)

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```
Once activated, you should see (venv) in your terminal.

### Installing Dependencies

After activating the virtual environment, install required dependencies with:

```
pip install -r requirements.txt
```
To add new dependencies and update requirements.txt, run:

```
pip freeze > requirements.txt
```
### Deactivating the Environment

When finished, deactivate the virtual environment by running:

```
deactivate
```