from TM1py.Services import TM1Service
from TM1py.Utils import Utils

#CREATE A CONSTANT for Server information
SERVER_ADDRESS = {'address':'dev.knowledgeseed.ch', 'port':5384}

# login in with AuthMode = 1
with TM1Service(address=SERVER_ADDRESS.get('address'), port=SERVER_ADDRESS.get('port'), user='admin', password='apple', ssl=False) as tm1:
    print(tm1.server.get_product_version())
    # define MDX Query
    mdx ="SELECT " \
          "NON EMPTY {[Month].[Month].[Jan],[Month].[Month].[Feb],[Month].[Month].[Mar]} " \
          "ON COLUMNS ," \
          "NON EMPTY " \
          "{[Account].[Account].[6000],[Account].[Account].[6005]," \
          "[Account].[Account].[6010],[Account].[Account].[6015]," \
          "[Account].[Account].[6020]}" \
          "ON ROWS "\
          "FROM [GLTransactions] " \
          "WHERE ([TrxID].[TrxID].[Total]," \
          "[organization].[organization].[Total Company]," \
          "[Year].[Year].[Y2]," \
          "[TrxMeasures].[TrxMeasures].[Amount])"

    # Get data from GLTransaction cube through MDX
    tran_data = tm1.cubes.cells.execute_mdx(mdx)

    # Build pandas DataFrame fram raw cellset data
    df = Utils.build_pandas_dataframe_from_cellset(tran_data)

    print(df)

    # Calculate Statistical measures for dataframe
    print(df.describe())