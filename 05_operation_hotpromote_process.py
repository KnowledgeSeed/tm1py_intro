import keyring
import pandas as pd
import configparser
from TM1py import TM1Service

config = configparser.ConfigParser()
config.read(r'.\config.ini')
#read info from config
INSTANCE = "tm1srv_24retail_dev_with_secret"
INSTANCE_LOCAL = "local_24ret"
address = config[INSTANCE]["address"]
port = config[INSTANCE]["port"]
ssl = config[INSTANCE]["ssl"]
user = config[INSTANCE]["user"]
# interact with Windows Credential Manager through the keyring library
password = keyring.get_password(INSTANCE, user)
config[INSTANCE]["password"] = password

process_name = "_TM1py_Oktatas"

with TM1Service(**config[INSTANCE]) as tm1:
    tm1_version = tm1.server.get_product_version()
    print(tm1_version)

    process = tm1.processes.get(process_name)
    print(process)

address = config[INSTANCE_LOCAL]["address"]
port = config[INSTANCE_LOCAL]["port"]
ssl = config[INSTANCE_LOCAL]["ssl"]
user = config[INSTANCE_LOCAL]["user"]
password = config[INSTANCE_LOCAL]["password"]

with TM1Service(**config[INSTANCE_LOCAL]) as tm1:
    tm1_version = tm1.server.get_product_version()
    print(tm1_version)
    if tm1.processes.exists(process_name):
        tm1.processes.delete(process_name)
    else:
        tm1.processes.create(process)
        tm1.processes.execute(process_name)


