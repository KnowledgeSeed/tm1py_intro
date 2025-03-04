from TM1py.Services import TM1Service

# login in with AuthMode = 1
with TM1Service(address='dev.knowledgeseed.ch', port=5384, user='admin', password='apple', ssl=False) as tm1:
    print(tm1.server.get_product_version())
