import os
import xml.etree.ElementTree as ET
from pathlib import Path
PATH = "xmls/"

if __name__ == '__main__':
    ret = {}

    path_list = Path(PATH).rglob('*.xml')  # make sure that this is recursive
    for xml_path in path_list:
        xml_path = str(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        spikes = root.find('spikeDetection')
        groups = spikes.find('channelGroups').findall('group')
        file_name = xml_path.split('_')[-1][:-4]

        for i, group in enumerate(groups):
            channels = group.find('channels')
            num_channels = len(channels.findall('channel'))
            if num_channels != 8:
                print(f"File {file_name} on shank {i + 1} has only {num_channels} recording sites")
            ret[f"{file_name}_{i + 1}"] = num_channels

        return ret
