import sys
import os
import json
import logging
logging.basicConfig(level = logging.INFO)
import requests


def connect_to_mw_dict(key, dict):
  logging.info("Connecting to Merriam-Webster Dictionary")
  url = "https://dictionaryapi.com/api/v3/references/sd3/json/"+dict+"?key="+key
  attr = {'word': dict,'key': key}
  req = requests.get(url = url, params = attr)

  if req.status_code == 200:
    logging.info("Connection successful to MW ")
    return True, req.json()
  else:
    print("Connection failed to MW.")

def main():
  key=os.environ["API_KEY"]
  dict= sys.argv[1] 
  outcome, result_json = connect_to_mw_dict(key, dict)

  if outcome == True:
    txt= str(result_json[0] ['def'][0] ['sseq'][0][0][1] ['dt'][0][1]).replace('{bc}','')
    print ("Result:",txt)
  else:
    print ("Results not found")

if __name__ == "__main__":
    main()