from TM1py.Services import TM1Service
#CREATE A CONSTANT for Server information
SERVER_ADDRESS = {'address':'dev.knowledgeseed.ch', 'port':5384}


# login in with AuthMode = 1
with TM1Service(address=SERVER_ADDRESS.get('address'), port=SERVER_ADDRESS.get('port'), user='admin', password='apple', ssl=False) as tm1:
    print(tm1.server.get_product_version())