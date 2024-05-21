import os
from datetime import datetime
import time

def initialize_storage():
  pathname = os.path.join("attendance-data")  
  os.makedirs(pathname, exist_ok=True)
  

def get_time_filename():
  date = datetime.now().strftime('%d-%m-%Y')
  filename = os.path.join("attendance-data", "%s.txt" % (date))   
  return filename

def check_attendance(name: str):
  with open(get_time_filename(), 'a') as file:
    file.write("%s\t%s\r\n" % (name, round(time.time())))
    
        