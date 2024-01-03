import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\proprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features) 
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)




#Responsible for mapping the inputs that we are giving in HTML to  the "backend"
class CustomData:
    def __init__(self, duration: int,protocol_type: str,service: str,flag: str,src_bytes: int,dst_bytes: int,land: int,wrong_fragment: int,urgent: int,hot: int,num_failed_logins: int,logged_in: int,num_compromised: int,root_shell: int,su_attempted: int,num_file_creations: int,num_shells: int,num_access_files: int,is_guest_login: int,count: int,srv_count: int,dst_host_srv_count: int,serror_rate: float,rerror_rate: float,same_srv_rate: float,diff_srv_rate: float,srv_diff_host_rate: float,dst_host_count: int,dst_host_diff_srv_rate: float,dst_host_same_src_port_rate:float, dst_host_srv_diff_host_rate: float):

        self.duration = duration
        self.protocol_type = protocol_type
        self.service = service
        self.flag = flag
        self.src_bytes = src_bytes
        self.dst_bytes = dst_bytes
        self.land = land
        self.wrong_fragment = wrong_fragment
        self.urgent = urgent
        self.hot = hot
        self.num_failed_logins = num_failed_logins
        self.logged_in = logged_in
        self.num_compromised = num_compromised
        self.root_shell = root_shell
        self.su_attempted = su_attempted
        self.num_file_creations = num_file_creations
        self.num_shells = num_shells
        self.num_access_files = num_access_files
        self.is_guest_login = is_guest_login
        self.count = count
        self.srv_count = srv_count
        self.serror_rate = serror_rate
        self.rerror_rate = rerror_rate
        self.same_srv_rate = same_srv_rate
        self.diff_srv_rate = diff_srv_rate
        self.srv_diff_host_rate = srv_diff_host_rate
        self.dst_host_count = dst_host_count
        self.dst_host_diff_srv_rate = dst_host_diff_srv_rate
        self.dst_host_same_src_port_rate = dst_host_same_src_port_rate
        self.dst_host_srv_count=dst_host_srv_count
        self.dst_host_srv_diff_host_rate=dst_host_srv_diff_host_rate

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "duration": [self.duration],
                "protocol_type": [self.protocol_type],
                "service": [self.service],
                "flag": [self.flag],
                "src_bytes": [self.src_bytes],
                "dst_bytes": [self.dst_bytes],
                "land": [self.land],
                "wrong_fragment": [self.wrong_fragment],
                "urgent": [self.urgent],
                "hot": [self.hot],
                "num_failed_logins": [self.num_failed_logins],
                "logged_in": [self.logged_in],
                "num_compromised": [self.num_compromised],
                "root_shell": [self.root_shell],
                "su_attempted": [self.su_attempted],
                "num_file_creations": [self.num_file_creations],
                "num_shells": [self.num_shells],
                "num_access_files": [self.num_access_files],
                "is_guest_login": [self.is_guest_login],
                "count": [self.count],
                "srv_count": [self.srv_count],
                "serror_rate": [self.serror_rate],
                "rerror_rate": [self.rerror_rate],
                "same_srv_rate": [self.same_srv_rate],
                "diff_srv_rate": [self.diff_srv_rate],
                "srv_diff_host_rate": [self.srv_diff_host_rate],
                "dst_host_count": [self.dst_host_count],
                "dst_host_diff_srv_rate": [self.dst_host_diff_srv_rate],
                "dst_host_same_src_port_rate": [self.dst_host_same_src_port_rate],
                "dst_host_srv_count": [self.dst_host_srv_count],
                "dst_host_srv_diff_host_rate": [self.dst_host_srv_diff_host_rate],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

       
