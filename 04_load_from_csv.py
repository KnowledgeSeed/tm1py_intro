import keyring
import pandas as pd
import configparser
from TM1py import TM1Service

config = configparser.ConfigParser()
config.read(r'.\config.ini')
#read info from config
INSTANCE = "tm1srv_24retail_dev_with_secret"
address = config[INSTANCE]["address"]
port = config[INSTANCE]["port"]
ssl = config[INSTANCE]["ssl"]
user = config[INSTANCE]["user"]
# interact with Windows Credential Manager through the keyring library
password = keyring.get_password(INSTANCE, user)
config[INSTANCE]["password"] = password

## pass as kwargs
with TM1Service(**config[INSTANCE]) as tm1:
    tm1_version = tm1.server.get_product_version()
    print(tm1_version)
    # define cube
    cube = 'GLTransactions'
    #define clear mdx expression
    mdx = """
        SELECT 
        NON EMPTY 
        {[TrxMeasures].[TrxMeasures].Members} 
        ON COLUMNS , 
        NON EMPTY 
        {[Account].[Account].[6000],
        [Account].[Account].[6005],
        [Account].[Account].[6010],
        [Account].[Account].[6015],
        [Account].[Account].[6020]}
        * TM1ToggleExpandMode({TM1FILTERBYLEVEL({TM1SUBSETALL([TrxID].[TrxID])}, 0)},EXPAND_ABOVE)
        * {TM1SubsetToSet([organization].[organization], "L0 Organization")} 
        * {[Month].[Month].[Jan],[Month].[Month].[Feb],[Month].[Month].[Mar]} 
        PROPERTIES [organization].[organization].[Caption_Default]  ON ROWS 
        FROM [GLTransactions] 
        WHERE 
        (
        [Year].[Year].[Y3]
        )
    """
    #read data from csv to pandas df
    df = pd.read_csv('trx_upload.csv')
    print(df)
    #make sure all column for dimensions are string
    for column in df.columns:
        if column != 'Value':
            df[column] = df[column].astype(str)
    #clear target
    tm1.cells.clear_with_mdx(cube=cube,mdx=mdx)
    #write df to cube
    tm1.cells.write_dataframe(data=df,
                              cube_name=cube,
                              deactivate_transaction_log=True,
                              increment=False,
                              sum_numeric_duplicates=True)
    #clear target after load
    #tm1.cells.clear_with_mdx(cube=cube,mdx=mdx)