import configparser

class CongigerLoader:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance
    
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_database_parameter(self, name):
        if name == "API_KEY":
            api_key_path = self.config["DataBase"]["API_KEY"]
            with open(api_key_path, "r") as f:
                result = f.read()

        elif name == "URL":
            result = self.config["DataBase"]["URL"]

        elif name == "COLLECTION_NAME":
            result = self.config["DataBase"]["COLLECTION_NAME"]


        return result
    

    def get_data_process_parameter(self, name):
        return self.config["DataProcess"][name]




CONFIG_PATH = "configure/configure.ini"

 #load config
config = CongigerLoader(CONFIG_PATH)  

# if __name__ == "__main__":

#     CONFIG_PATH = "configure/configure.ini"

#     #load config
#     config = CongigerLoader(CONFIG_PATH)
            
        