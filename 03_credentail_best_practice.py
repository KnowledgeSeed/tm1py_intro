import configparser
from getpass import getpass

import keyring
from TM1py import TM1Service

config = configparser.ConfigParser()
config.read(r'.\config.ini')

# get instance parameters from ini, and password safely
INSTANCE = "riskmodel"
address = config[INSTANCE]["address"]
port = config[INSTANCE]["port"]
ssl = config[INSTANCE]["ssl"]
user = config[INSTANCE]["user"]
namespace = config[INSTANCE]["namespace"]
password = keyring.get_password(INSTANCE, user)

# login in with AuthMode = 5 wo SSO
with TM1Service(address=address, port=port, user=user,password=password,ssl=False, namespace=namespace) as tm1_cam_wo_ssso:
    print(tm1_cam_wo_ssso.server.get_product_version())

INSTANCE = "tm1srv_24retail_dev_with_secret"
address = config[INSTANCE]["address"]
port = config[INSTANCE]["port"]
ssl = config[INSTANCE]["ssl"]
user = config[INSTANCE]["user"]
# interact with Windows Credential Manager through the keyring library

password = keyring.get_password(INSTANCE, user)

# if there are no pass stored ask for it
if password is None:
    password = getpass(prompt= f"Please insert password for user '{user}' and instance '{INSTANCE}':")
    keyring.set_password(INSTANCE, user, password)

config[INSTANCE]["password"] = password
## pass as kwargs
with TM1Service(**config[INSTANCE]) as tm1:
    tm1_version = tm1.server.get_product_version()
    print(tm1_version)

INSTANCE = "KSDEMO_Saas_v12"
baseurl = config[INSTANCE]["base_url"]
ssl = config[INSTANCE]["ssl"]
user = config[INSTANCE]["user"]
password = keyring.get_password(INSTANCE, user)

# login in with AuthMode = 5 wo SSO
with TM1Service(base_url=baseurl, user=user,password=password,ssl=ssl) as tm1_v12_paas:
    print(tm1_v12_paas.server.get_product_version())