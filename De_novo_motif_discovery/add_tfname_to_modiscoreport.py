#!/groups/stark/nikolaus.mandlburger/.conda/envs/scrape/bin/python3.11

import argparse
import requests
import re
import os

#generate a parser
parser = argparse.ArgumentParser(prog='ProgramName', description="Add tf names for jaspar IDs to tf modisco report")

#add arguments
parser.add_argument("-r","--html_report",type=str, help="The modisco html report file for which the tf names should be added. This is not modified but a modiefied copy is made")
                                
#parse and retrieve arguments
args=parser.parse_args()
                            
    
if args.html_report != None:
    in_html = args.html_report
    out_html = os.path.join(os.path.dirname(in_html),os.path.basename(in_html).replace('.html','_tfnames.html'))
else:
    raise ValueError("No html report file stated")



def get_transcription_factor_name(jaspar_id):
    url = f"http://jaspar.genereg.net/api/v1/matrix/{jaspar_id}/"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        tf_name = data.get('name')
        return tf_name
    else:
        return None


pattern = r'(?<=<td>)MA\d+\.\d+(?=</td>)'

with open(in_html, 'r') as infile:
    with open(out_html,'w') as outfile:
        lines = infile.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            if len(matches)==1:
                jaspar_id=matches[0]
                #look up id in database
                tf_name = get_transcription_factor_name(jaspar_id)
                new_entry=f"{jaspar_id} - {tf_name}"
                newline = line.replace(jaspar_id, new_entry)
                print(newline)
            else:
                newline = line
            outfile.write(newline)
            