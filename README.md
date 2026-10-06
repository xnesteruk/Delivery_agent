# **User manual**

Open the Delivery\_agent project in IDE. Run terminal commands from the project folder containing main.py, requirements.txt, configs and tests.

Install the required packages with **pip**

The requirements.txt file lists the required packages, including matplotlib and pytest. Install them with:

##### `python -m pip install -r requirements.txt`

### **Run a configuration**

###### `python main.py --config configs/map_10.json`

###### `python main.py --config configs/map_15.json`

###### `python main.py --config configs/map_20.json`

Run one command at a time. Each supplied experiment preset runs 20 scenarios. The program prints the results and creates a new results/experiment\_NNN folder.

Run the default configuration with the button

Open main.py in IDE and click the green Run triangle next to the main entry point, or right-click main.py and select Run. Leave script parameters empty. The program then uses configs/default.json. The equivalent terminal command is:


### Tests

To run tests:
#### `python -m pytest tests -v`
